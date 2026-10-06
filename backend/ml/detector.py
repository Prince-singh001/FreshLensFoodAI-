import cv2
import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional

try:
    from config import CONFIDENCE_THRESHOLD, FOOD_CONFIDENCE_THRESHOLD
    from ml.loader import get_detector_model
    from ml.freshness import classify_food_crop
    from utils.logger import logger
except ImportError:
    from backend.config import CONFIDENCE_THRESHOLD, FOOD_CONFIDENCE_THRESHOLD
    from backend.ml.loader import get_detector_model
    from backend.ml.freshness import classify_food_crop
    from backend.utils.logger import logger

NON_FOOD_REJECTION_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
}

COCO_FOOD_CLASSES = {
    "banana": "Banana",
    "apple": "Apple",
    "orange": "Orange",
    "carrot": "Carrot",
    "broccoli": "Broccoli",
    "pizza": "Pizza",
    "sandwich": "Sandwich",
    "hot dog": "Hot Dog",
    "donut": "Donut",
    "cake": "Cake"
}


def _box_area(box: Dict[str, int]) -> int:
    return max(0, box["x2"] - box["x1"]) * max(0, box["y2"] - box["y1"])


def _clip_box(box: Dict[str, int], width: int, height: int) -> Dict[str, int]:
    x1 = max(0, min(int(box["x1"]), width - 1))
    y1 = max(0, min(int(box["y1"]), height - 1))
    x2 = max(x1 + 1, min(int(box["x2"]), width))
    y2 = max(y1 + 1, min(int(box["y2"]), height))
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _iou(a: Dict[str, int], b: Dict[str, int]) -> float:
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    intersection = w * h
    if intersection <= 0:
        return 0.0

    union = _box_area(a) + _box_area(b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _intersection_over_smaller(a: Dict[str, int], b: Dict[str, int]) -> float:
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    intersection = w * h
    if intersection <= 0:
        return 0.0

    smaller = min(_box_area(a), _box_area(b))
    if smaller <= 0:
        return 0.0
    return intersection / smaller


def _deduplicate_candidates(candidates: List[Dict[str, Any]], iou_thresh: float = 0.45, max_candidates: int = 10) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    # Sort descending by confidence
    candidates = sorted(candidates, key=lambda c: c.get("confidence", 0.0), reverse=True)
    selected: List[Dict[str, Any]] = []

    for cand in candidates:
        box = cand["bbox"]
        duplicate = False
        for ex in selected:
            ex_box = ex["bbox"]
            iou_val = _iou(box, ex_box)
            ios_val = _intersection_over_smaller(box, ex_box)
            if iou_val >= iou_thresh or ios_val >= 0.70:
                duplicate = True
                break

        if not duplicate:
            selected.append(cand)
            if len(selected) >= max_candidates:
                break

    # Order top-to-bottom, left-to-right
    selected.sort(key=lambda c: (c["bbox"]["y1"], c["bbox"]["x1"]))
    for idx, c in enumerate(selected, start=1):
        c["id"] = idx
    return selected


def _foreground_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, s, v = cv2.split(hsv)

    colorful = (s > 25) & (v > 30)
    non_white = gray < 235
    non_black = gray > 20

    mask = np.where((colorful | non_white) & non_black, 255, 0).astype(np.uint8)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
    return mask


def find_salient_food_regions(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """
    Lightweight computer vision localization fallback:
    Extracts distinct foreground produce regions using color/luminance segmentation
    and distance-transform component analysis.
    Assigns food identity using learned classifier rather than shape rules.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    height, width = image_bgr.shape[:2]
    total_area = height * width
    mask = _foreground_mask(image_bgr)

    candidates: List[Dict[str, Any]] = []

    # Distance transform to separate touching produce items
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_max = dist.max()

    if dist_max > 0:
        # High confidence cores
        _, sure_fg = cv2.threshold(dist, 0.30 * dist_max, 255, cv2.THRESH_BINARY)
        sure_fg = np.uint8(sure_fg)

        contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < total_area * 0.008:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            # Expand bounding box slightly to capture full produce edges
            pad_x = int(w * 0.25)
            pad_y = int(h * 0.25)
            box = _clip_box({
                "x1": x - pad_x,
                "y1": y - pad_y,
                "x2": x + w + pad_x,
                "y2": y + h + pad_y
            }, width, height)

            bw = box["x2"] - box["x1"]
            bh = box["y2"] - box["y1"]
            if bw < 35 or bh < 35:
                continue

            # Classify crop using learned classifier
            crop = image_bgr[box["y1"]:box["y2"], box["x1"]:box["x2"]]
            if crop.size > 0:
                item_key, item_display, category, conf = classify_food_crop(crop)
                candidates.append({
                    "detected_label": item_key,
                    "display_name": item_display,
                    "category": category,
                    "confidence": round(float(conf), 4),
                    "bbox": box,
                    "source": "dt_contour"
                })

    # Also inspect external contours of full mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < total_area * 0.02 or area > total_area * 0.85:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        box = _clip_box({"x1": x, "y1": y, "x2": x + w, "y2": y + h}, width, height)
        bw = box["x2"] - box["x1"]
        bh = box["y2"] - box["y1"]
        if bw < 40 or bh < 40:
            continue

        crop = image_bgr[box["y1"]:box["y2"], box["x1"]:box["x2"]]
        if crop.size > 0:
            item_key, item_display, category, conf = classify_food_crop(crop)
            candidates.append({
                "detected_label": item_key,
                "display_name": item_display,
                "category": category,
                "confidence": round(float(conf), 4),
                "bbox": box,
                "source": "mask_contour"
            })

    # If no candidate was found, fallback to full image
    if not candidates:
        crop = image_bgr.copy()
        item_key, item_display, category, conf = classify_food_crop(crop)
        candidates.append({
            "detected_label": item_key,
            "display_name": item_display,
            "category": category,
            "confidence": round(float(conf), 4),
            "bbox": {"x1": 0, "y1": 0, "x2": width, "y2": height},
            "source": "full_image"
        })

    return _deduplicate_candidates(candidates, iou_thresh=0.45, max_candidates=10)


def detect_objects_in_image(
    image_bgr: np.ndarray
) -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
    """
    Detects and localizes food produce items in the image.
    Uses YOLO detector when available, with non-food rejection logic.
    Gracefully falls back to computer vision localization + learned crop classification.
    Returns:
        (detected_objects, is_rejected_non_food, rejection_message)
    """
    started_at = time.perf_counter()
    logger.info("[DETECTOR] Detection started.")

    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        return [], True, "Invalid image data."

    height, width = image_bgr.shape[:2]
    logger.info(f"[DETECTOR] Image shape: {height}x{width}")

    if height < 16 or width < 16:
        return [], True, "Image resolution is too small."

    detector = get_detector_model()

    if detector is not None:
        try:
            logger.info("[DETECTOR] Running YOLO inference...")
            results = detector.predict(
                source=image_bgr,
                conf=0.25,
                imgsz=640,
                max_det=12,
                device="cpu",
                verbose=False
            )

            food_detections: List[Dict[str, Any]] = []
            non_food_detections: List[Tuple[str, float]] = []

            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                names = getattr(result, "names", {})

                for box in boxes:
                    try:
                        cls_id = int(box.cls[0])
                        cls_name = str(names.get(cls_id, "")).lower().strip()
                        conf = float(box.conf[0])
                        coords = box.xyxy[0].tolist()
                        if len(coords) != 4:
                            continue

                        clipped = _clip_box({
                            "x1": int(coords[0]),
                            "y1": int(coords[1]),
                            "x2": int(coords[2]),
                            "y2": int(coords[3])
                        }, width, height)

                        if cls_name in NON_FOOD_REJECTION_CLASSES and conf >= 0.45:
                            non_food_detections.append((cls_name, conf))
                        elif cls_name in COCO_FOOD_CLASSES:
                            food_detections.append({
                                "detected_label": cls_name,
                                "display_name": COCO_FOOD_CLASSES[cls_name],
                                "confidence": round(conf, 4),
                                "bbox": clipped,
                                "source": "yolo"
                            })
                    except Exception as err:
                        logger.warning(f"[DETECTOR] Failed parsing box: {err}")
                        continue

            # Check if image is purely non-food
            if non_food_detections and not food_detections:
                prominent_non_food = max(non_food_detections, key=lambda x: x[1])
                label = prominent_non_food[0].replace("_", " ").title()
                conf_pct = int(prominent_non_food[1] * 100)
                logger.info(f"[DETECTOR] Non-food detected: {label} ({conf_pct}%)")
                return [], True, f"Detected non-food object ({label} - {conf_pct}%). Please upload a food item."

            if food_detections:
                deduped = _deduplicate_candidates(food_detections, iou_thresh=0.45, max_candidates=10)
                elapsed = (time.perf_counter() - started_at) * 1000
                logger.info(f"[DETECTOR] YOLO Candidate objects: {len(deduped)} in {elapsed:.2f} ms")
                for c in deduped:
                    logger.info(f"[DETECTOR] Candidate {c['id']} | {c['display_name']} | conf={c['confidence']} | bbox={c['bbox']}")
                return deduped, False, None

            logger.info("[DETECTOR] YOLO detected no produce food classes; checking produce localization fallback.")

        except Exception as e:
            logger.warning(f"[DETECTOR] YOLO inference error: {e}. Switching to produce localization fallback.")

    # Lightweight produce localization fallback
    logger.info("[DETECTOR] Running lightweight multi-object produce localization.")
    candidates = find_salient_food_regions(image_bgr)
    elapsed = (time.perf_counter() - started_at) * 1000
    logger.info(f"[DETECTOR] Candidate objects: {len(candidates)} in {elapsed:.2f} ms")
    for c in candidates:
        logger.info(f"[DETECTOR] Candidate {c['id']} | {c.get('display_name', c.get('detected_label'))} | conf={c['confidence']} | bbox={c['bbox']}")

    return candidates, False, None
