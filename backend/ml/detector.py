import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

try:
    from ml.loader import get_detector_model
    from config import CONFIDENCE_THRESHOLD
    from utils.logger import logger
except ImportError:
    from backend.ml.loader import get_detector_model
    from backend.config import CONFIDENCE_THRESHOLD
    from backend.utils.logger import logger


COCO_FOOD_CLASSES = {
    "banana": "Banana",
    "apple": "Apple",
    "sandwich": "Sandwich",
    "orange": "Orange",
    "broccoli": "Broccoli",
    "carrot": "Carrot",
    "hot dog": "Hot Dog",
    "pizza": "Pizza",
    "donut": "Donut",
    "cake": "Cake",
    "bowl": "Food"
}


NON_FOOD_REJECTION_CLASSES = {
    "person",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "chair",
    "couch",
    "bed",
    "dining table",
    "toilet",
    "bench",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "airplane",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "dog",
    "cat",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "bird",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket"
}


def find_salient_food_regions(
    image_bgr: np.ndarray,
    min_area_ratio: float = 0.04
) -> List[Dict[str, int]]:

    if image_bgr is None or image_bgr.size == 0:
        return []

    try:
        h, w = image_bgr.shape[:2]

        if h <= 0 or w <= 0:
            return []

        total_area = h * w

        gray = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY
        )

        blurred = cv2.GaussianBlur(
            gray,
            (7, 7),
            0
        )

        _, otsu = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        edges = cv2.Canny(
            blurred,
            30,
            120
        )

        combined = cv2.bitwise_or(
            otsu,
            edges
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 9)
        )

        closed = cv2.morphologyEx(
            combined,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        proposals = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < total_area * min_area_ratio:
                continue

            if area > total_area * 0.95:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)

            if bw <= 0 or bh <= 0:
                continue

            aspect_ratio = bw / float(bh)

            if aspect_ratio < 0.15 or aspect_ratio > 6.0:
                continue

            proposals.append({
                "x1": int(x),
                "y1": int(y),
                "x2": int(x + bw),
                "y2": int(y + bh)
            })

        proposals.sort(
            key=lambda b: (
                b["x2"] - b["x1"]
            ) * (
                b["y2"] - b["y1"]
            ),
            reverse=True
        )

        return proposals[:5]

    except Exception as e:
        logger.exception(
            f"[DETECTOR] Salient region detection failed: {e}"
        )
        return []


def _create_full_image_candidate(
    image_bgr: np.ndarray
) -> Dict[str, Any]:

    h, w = image_bgr.shape[:2]

    return {
        "id": 1,
        "detected_label": "produce_candidate",
        "display_name": "Produce Item",
        "confidence": 0.50,
        "is_candidate": True,
        "bbox": {
            "x1": 0,
            "y1": 0,
            "x2": int(w),
            "y2": int(h)
        }
    }


def _create_fallback_candidates(
    image_bgr: np.ndarray
) -> List[Dict[str, Any]]:

    proposals = find_salient_food_regions(
        image_bgr
    )

    if not proposals:
        return [
            _create_full_image_candidate(
                image_bgr
            )
        ]

    candidates = []

    for index, proposal in enumerate(
        proposals[:5],
        start=1
    ):
        candidates.append({
            "id": index,
            "detected_label": "produce_candidate",
            "display_name": "Produce Item",
            "confidence": 0.50,
            "is_candidate": True,
            "bbox": proposal
        })

    return candidates


def detect_objects_in_image(
    image_bgr: np.ndarray
) -> Tuple[
    List[Dict[str, Any]],
    bool,
    Optional[str]
]:

    started_at = __import__(
        "time"
    ).perf_counter()

    logger.info(
        "[DETECTOR] Detection started."
    )

    if image_bgr is None:
        logger.error(
            "[DETECTOR] Received None image."
        )

        return [], True, "Invalid image."

    if not isinstance(
        image_bgr,
        np.ndarray
    ):
        logger.error(
            "[DETECTOR] Invalid image type."
        )

        return [], True, "Invalid image format."

    if image_bgr.size == 0:
        logger.error(
            "[DETECTOR] Received empty image."
        )

        return [], True, "Empty image."

    try:
        h, w = image_bgr.shape[:2]

        logger.info(
            f"[DETECTOR] Image shape: {h}x{w}"
        )

        if h < 10 or w < 10:
            logger.warning(
                "[DETECTOR] Image resolution too small."
            )

            return [], True, "Image resolution is too small."

    except Exception as e:
        logger.exception(
            f"[DETECTOR] Image validation failed: {e}"
        )

        return [], True, "Invalid image."

    detector = None

    try:
        logger.info(
            "[DETECTOR] Loading detector model..."
        )

        detector = get_detector_model()

        logger.info(
            "[DETECTOR] Detector model loaded."
        )

    except Exception as e:
        logger.exception(
            f"[DETECTOR] Detector model loading failed: {e}"
        )

    if detector is not None:

        try:
            logger.info(
                "[DETECTOR] Starting YOLO inference..."
            )

            results = detector.predict(
                source=image_bgr,
                conf=float(CONFIDENCE_THRESHOLD),
                imgsz=640,
                max_det=10,
                device="cpu",
                verbose=False
            )

            logger.info(
                "[DETECTOR] YOLO inference completed."
            )

            food_detections = []
            non_food_detections = []

            for result_index, result in enumerate(
                results
            ):

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
                        cls_id = int(
                            box.cls[0]
                        )

                        cls_name = str(
                            names.get(
                                cls_id,
                                ""
                            )
                        ).lower().strip()

                        conf = float(
                            box.conf[0]
                        )

                        coords = box.xyxy[
                            0
                        ].tolist()

                        if len(coords) != 4:
                            continue

                        x1, y1, x2, y2 = [
                            int(v)
                            for v in coords
                        ]

                        x1 = max(
                            0,
                            min(x1, w - 1)
                        )

                        y1 = max(
                            0,
                            min(y1, h - 1)
                        )

                        x2 = max(
                            x1 + 1,
                            min(x2, w)
                        )

                        y2 = max(
                            y1 + 1,
                            min(y2, h)
                        )

                        if (
                            cls_name
                            in NON_FOOD_REJECTION_CLASSES
                            and conf >= 0.45
                        ):
                            non_food_detections.append(
                                (
                                    cls_name,
                                    conf
                                )
                            )

                        elif (
                            cls_name
                            in COCO_FOOD_CLASSES
                            and conf >= float(
                                CONFIDENCE_THRESHOLD
                            )
                        ):
                            food_detections.append({
                                "detected_label": cls_name,
                                "display_name": COCO_FOOD_CLASSES[
                                    cls_name
                                ],
                                "confidence": round(
                                    conf,
                                    4
                                ),
                                "bbox": {
                                    "x1": x1,
                                    "y1": y1,
                                    "x2": x2,
                                    "y2": y2
                                }
                            })

                    except Exception as box_error:
                        logger.warning(
                            f"[DETECTOR] Invalid detection box: "
                            f"{box_error}"
                        )
                        continue

            logger.info(
                f"[DETECTOR] YOLO results | "
                f"food={len(food_detections)} | "
                f"non_food={len(non_food_detections)}"
            )

            if food_detections:

                for index, detection in enumerate(
                    food_detections,
                    start=1
                ):
                    detection["id"] = index

                elapsed = (
                    __import__(
                        "time"
                    ).perf_counter()
                    - started_at
                ) * 1000

                logger.info(
                    f"[DETECTOR] Food detection successful | "
                    f"objects={len(food_detections)} | "
                    f"time={elapsed:.2f} ms"
                )

                return (
                    food_detections,
                    False,
                    None
                )

            if non_food_detections:

                prominent_non_food = max(
                    non_food_detections,
                    key=lambda x: x[1]
                )

                label = (
                    prominent_non_food[0]
                    .title()
                )

                confidence = int(
                    prominent_non_food[1] * 100
                )

                elapsed = (
                    __import__(
                        "time"
                    ).perf_counter()
                    - started_at
                ) * 1000

                logger.info(
                    f"[DETECTOR] Non-food detected | "
                    f"label={label} | "
                    f"confidence={confidence}% | "
                    f"time={elapsed:.2f} ms"
                )

                return (
                    [],
                    True,
                    (
                        f"Detected non-food object "
                        f"({label} - {confidence}%). "
                        f"Please upload or capture a food item."
                    )
                )

            logger.info(
                "[DETECTOR] YOLO found no supported food. "
                "Using fallback localization."
            )

        except Exception as e:

            logger.exception(
                f"[DETECTOR] YOLO inference failed: {e}"
            )

    else:

        logger.warning(
            "[DETECTOR] Detector unavailable. "
            "Using fallback localization."
        )

    fallback_started = (
        __import__(
            "time"
        ).perf_counter()
    )

    fallback_candidates = _create_fallback_candidates(
        image_bgr
    )

    fallback_time = (
        __import__(
            "time"
        ).perf_counter()
        - fallback_started
    ) * 1000

    logger.info(
        f"[DETECTOR] Fallback localization completed | "
        f"objects={len(fallback_candidates)} | "
        f"time={fallback_time:.2f} ms"
    )

    total_time = (
        __import__(
            "time"
        ).perf_counter()
        - started_at
    ) * 1000

    logger.info(
        f"[DETECTOR] Detection completed | "
        f"objects={len(fallback_candidates)} | "
        f"total_time={total_time:.2f} ms"
    )

    return (
        fallback_candidates,
        False,
        None
    )