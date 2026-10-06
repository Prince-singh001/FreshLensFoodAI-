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

MAX_LOCALIZATION_SIZE = 1280
MAX_CANDIDATES = 6


def _box_area(box: Dict[str, int]) -> int:
    return max(0, box["x2"] - box["x1"]) * max(0, box["y2"] - box["y1"])


def _clip_box(box: Dict[str, int], width: int, height: int) -> Dict[str, int]:
    x1 = max(0, min(int(box["x1"]), width - 1))
    y1 = max(0, min(int(box["y1"]), height - 1))
    x2 = max(x1 + 1, min(int(box["x2"]), width))
    y2 = max(y1 + 1, min(int(box["y2"]), height))

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2
    }


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


def _deduplicate_candidates(
    candidates: List[Dict[str, Any]],
    iou_thresh: float = 0.45,
    max_candidates: int = MAX_CANDIDATES
) -> List[Dict[str, Any]]:

    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda c: c.get("confidence", 0.0),
        reverse=True
    )

    selected = []

    for candidate in candidates:
        box = candidate["bbox"]
        duplicate = False

        for existing in selected:
            existing_box = existing["bbox"]

            if _iou(box, existing_box) >= iou_thresh:
                duplicate = True
                break

            if _intersection_over_smaller(box, existing_box) >= 0.70:
                duplicate = True
                break

        if not duplicate:
            selected.append(candidate)

        if len(selected) >= max_candidates:
            break

    selected.sort(
        key=lambda c: (
            c["bbox"]["y1"],
            c["bbox"]["x1"]
        )
    )

    for index, candidate in enumerate(selected, start=1):
        candidate["id"] = index

    return selected


def _resize_for_localization(
    image: np.ndarray
) -> Tuple[np.ndarray, float, float]:

    height, width = image.shape[:2]
    max_dim = max(height, width)

    if max_dim <= MAX_LOCALIZATION_SIZE:
        return image, 1.0, 1.0

    scale = MAX_LOCALIZATION_SIZE / float(max_dim)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    scale_x = width / float(new_width)
    scale_y = height / float(new_height)

    return resized, scale_x, scale_y


def _scale_box(
    box: Dict[str, int],
    scale_x: float,
    scale_y: float,
    width: int,
    height: int
) -> Dict[str, int]:

    return _clip_box(
        {
            "x1": int(box["x1"] * scale_x),
            "y1": int(box["y1"] * scale_y),
            "x2": int(box["x2"] * scale_x),
            "y2": int(box["y2"] * scale_y)
        },
        width,
        height
    )


def _foreground_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, saturation, value = cv2.split(hsv)

    colorful = (saturation > 25) & (value > 30)
    non_white = gray < 235
    non_black = gray > 20

    mask = np.where(
        (colorful | non_white) & non_black,
        255,
        0
    ).astype(np.uint8)

    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    kernel_medium = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_small,
        iterations=1
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_medium,
        iterations=1
    )

    return mask


def _generate_localization_boxes(
    image: np.ndarray
) -> List[Dict[str, Any]]:

    height, width = image.shape[:2]
    total_area = height * width

    mask = _foreground_mask(image)

    candidates = []

    dist = cv2.distanceTransform(
        mask,
        cv2.DIST_L2,
        5
    )

    dist_max = float(dist.max())

    if dist_max > 0:

        _, sure_fg = cv2.threshold(
            dist,
            0.30 * dist_max,
            255,
            cv2.THRESH_BINARY
        )

        sure_fg = np.uint8(sure_fg)

        contours, _ = cv2.findContours(
            sure_fg,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < total_area * 0.006:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            pad_x = max(8, int(w * 0.20))
            pad_y = max(8, int(h * 0.20))

            box = _clip_box(
                {
                    "x1": x - pad_x,
                    "y1": y - pad_y,
                    "x2": x + w + pad_x,
                    "y2": y + h + pad_y
                },
                width,
                height
            )

            box_width = box["x2"] - box["x1"]
            box_height = box["y2"] - box["y1"]

            if box_width < 35 or box_height < 35:
                continue

            box_area = _box_area(box)

            area_ratio = box_area / float(total_area)

            confidence = min(
                0.99,
                max(
                    0.20,
                    0.55 + min(area_ratio, 0.35)
                )
            )

            candidates.append(
                {
                    "bbox": box,
                    "confidence": round(float(confidence), 4),
                    "source": "dt_contour"
                }
            )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < total_area * 0.015:
            continue

        if area > total_area * 0.90:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < 35 or h < 35:
            continue

        box = _clip_box(
            {
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h
            },
            width,
            height
        )

        box_area = _box_area(box)
        area_ratio = box_area / float(total_area)

        confidence = min(
            0.95,
            max(
                0.20,
                0.50 + min(area_ratio, 0.40)
            )
        )

        candidates.append(
            {
                "bbox": box,
                "confidence": round(float(confidence), 4),
                "source": "mask_contour"
            }
        )

    return _deduplicate_candidates(
        candidates,
        iou_thresh=0.45,
        max_candidates=MAX_CANDIDATES
    )


def find_salient_food_regions(
    image_bgr: np.ndarray
) -> List[Dict[str, Any]]:

    if image_bgr is None or image_bgr.size == 0:
        return []

    original_height, original_width = image_bgr.shape[:2]

    working_image, scale_x, scale_y = _resize_for_localization(
        image_bgr
    )

    working_height, working_width = working_image.shape[:2]

    logger.info(
        f"[DETECTOR] Localization image: "
        f"{working_width}x{working_height}"
    )

    candidates = _generate_localization_boxes(
        working_image
    )

    if not candidates:
        logger.info(
            "[DETECTOR] No segmented regions found. "
            "Using full-image fallback."
        )

        full_box = {
            "x1": 0,
            "y1": 0,
            "x2": working_width,
            "y2": working_height
        }

        candidates = [
            {
                "bbox": full_box,
                "confidence": 0.50,
                "source": "full_image"
            }
        ]

    scaled_candidates = []

    for candidate in candidates:

        scaled_box = _scale_box(
            candidate["bbox"],
            scale_x,
            scale_y,
            original_width,
            original_height
        )

        scaled_candidates.append(
            {
                "bbox": scaled_box,
                "confidence": candidate["confidence"],
                "source": candidate["source"]
            }
        )

    scaled_candidates = _deduplicate_candidates(
        scaled_candidates,
        iou_thresh=0.45,
        max_candidates=MAX_CANDIDATES
    )

    final_candidates = []

    for candidate in scaled_candidates:

        box = candidate["bbox"]

        crop = image_bgr[
            box["y1"]:box["y2"],
            box["x1"]:box["x2"]
        ]

        if crop is None or crop.size == 0:
            continue

        try:

            item_key, item_display, category, classifier_conf = (
                classify_food_crop(crop)
            )

            if not item_key or item_key == "unknown":
                continue

            final_confidence = float(classifier_conf)

            final_candidates.append(
                {
                    "detected_label": item_key,
                    "display_name": item_display,
                    "category": category,
                    "confidence": round(
                        final_confidence,
                        4
                    ),
                    "bbox": box,
                    "source": candidate["source"]
                }
            )

        except Exception as error:

            logger.warning(
                f"[DETECTOR] Crop classification failed: {error}"
            )

    if not final_candidates:

        crop = image_bgr.copy()

        try:

            item_key, item_display, category, conf = (
                classify_food_crop(crop)
            )

            if item_key and item_key != "unknown":

                final_candidates.append(
                    {
                        "detected_label": item_key,
                        "display_name": item_display,
                        "category": category,
                        "confidence": round(
                            float(conf),
                            4
                        ),
                        "bbox": {
                            "x1": 0,
                            "y1": 0,
                            "x2": original_width,
                            "y2": original_height
                        },
                        "source": "full_image"
                    }
                )

        except Exception as error:

            logger.error(
                f"[DETECTOR] Full-image classification failed: {error}"
            )

    final_candidates = _deduplicate_candidates(
        final_candidates,
        iou_thresh=0.45,
        max_candidates=MAX_CANDIDATES
    )

    return final_candidates


def detect_objects_in_image(
    image_bgr: np.ndarray
) -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:

    started_at = time.perf_counter()

    logger.info("[DETECTOR] Detection started.")

    if (
        image_bgr is None
        or not isinstance(image_bgr, np.ndarray)
        or image_bgr.size == 0
    ):
        return [], True, "Invalid image data."

    height, width = image_bgr.shape[:2]

    logger.info(
        f"[DETECTOR] Image shape: {height}x{width}"
    )

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

            food_detections = []
            non_food_detections = []

            for result in results:

                boxes = getattr(
                    result,
                    "boxes",
                    None
                )

                if boxes is None:
                    continue

                names = getattr(
                    result,
                    "names",
                    {}
                )

                for box in boxes:

                    try:

                        cls_id = int(box.cls[0])

                        cls_name = str(
                            names.get(
                                cls_id,
                                ""
                            )
                        ).lower().strip()

                        conf = float(
                            box.conf[0]
                        )

                        coords = box.xyxy[0].tolist()

                        if len(coords) != 4:
                            continue

                        clipped = _clip_box(
                            {
                                "x1": int(coords[0]),
                                "y1": int(coords[1]),
                                "x2": int(coords[2]),
                                "y2": int(coords[3])
                            },
                            width,
                            height
                        )

                        if (
                            cls_name in NON_FOOD_REJECTION_CLASSES
                            and conf >= 0.45
                        ):

                            non_food_detections.append(
                                (
                                    cls_name,
                                    conf
                                )
                            )

                        elif cls_name in COCO_FOOD_CLASSES:

                            food_detections.append(
                                {
                                    "detected_label": cls_name,
                                    "display_name": COCO_FOOD_CLASSES[cls_name],
                                    "confidence": round(
                                        conf,
                                        4
                                    ),
                                    "bbox": clipped,
                                    "source": "yolo"
                                }
                            )

                    except Exception as error:

                        logger.warning(
                            f"[DETECTOR] Failed parsing box: {error}"
                        )

            if (
                non_food_detections
                and not food_detections
            ):

                prominent = max(
                    non_food_detections,
                    key=lambda x: x[1]
                )

                label = prominent[0].replace(
                    "_",
                    " "
                ).title()

                confidence = int(
                    prominent[1] * 100
                )

                return (
                    [],
                    True,
                    f"Detected non-food object "
                    f"({label} - {confidence}%). "
                    f"Please upload a food item."
                )

            if food_detections:

                deduped = _deduplicate_candidates(
                    food_detections,
                    iou_thresh=0.45,
                    max_candidates=MAX_CANDIDATES
                )

                elapsed = (
                    time.perf_counter()
                    - started_at
                ) * 1000

                logger.info(
                    f"[DETECTOR] YOLO Candidate objects: "
                    f"{len(deduped)} in {elapsed:.2f} ms"
                )

                for candidate in deduped:

                    logger.info(
                        f"[DETECTOR] Candidate "
                        f"{candidate['id']} | "
                        f"{candidate['display_name']} | "
                        f"conf={candidate['confidence']} | "
                        f"bbox={candidate['bbox']}"
                    )

                return deduped, False, None

            logger.info(
                "[DETECTOR] YOLO detected no food classes. "
                "Using localization fallback."
            )

        except Exception as error:

            logger.warning(
                f"[DETECTOR] YOLO inference error: {error}. "
                f"Switching to localization fallback."
            )

    logger.info(
        "[DETECTOR] Running lightweight multi-object produce localization."
    )

    candidates = find_salient_food_regions(
        image_bgr
    )

    elapsed = (
        time.perf_counter()
        - started_at
    ) * 1000

    logger.info(
        f"[DETECTOR] Candidate objects: "
        f"{len(candidates)} in {elapsed:.2f} ms"
    )

    for candidate in candidates:

        logger.info(
            f"[DETECTOR] Candidate "
            f"{candidate['id']} | "
            f"{candidate.get('display_name', candidate.get('detected_label'))} | "
            f"conf={candidate['confidence']} | "
            f"bbox={candidate['bbox']}"
        )

    return candidates, False, None