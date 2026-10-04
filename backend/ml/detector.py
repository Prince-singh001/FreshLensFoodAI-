import cv2
import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional

try:
    from utils.logger import logger
except ImportError:
    from backend.utils.logger import logger


def _box_area(box):
    return max(0, box["x2"] - box["x1"]) * max(0, box["y2"] - box["y1"])


def _clip_box(box, width, height):
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


def _iou(a, b):
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


def _intersection_over_smaller(a, b):
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


def _create_candidate(box, width, height, confidence, source):
    box = _clip_box(box, width, height)

    return {
        "detected_label": "produce_candidate",
        "display_name": "Produce Item",
        "confidence": round(float(confidence), 4),
        "is_candidate": True,
        "bbox": box,
        "source": source
    }


def _foreground_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, s, v = cv2.split(hsv)

    colorful = (
        (s > 30) &
        (v > 35)
    )

    dark = gray < 190

    mask = np.where(
        colorful | dark,
        255,
        0
    ).astype(np.uint8)

    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    kernel_medium = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
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


def _detect_round_objects(image):
    height, width = image.shape[:2]
    min_dim = min(height, width)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        1.2
    )

    min_radius = max(
        18,
        int(min_dim * 0.045)
    )

    max_radius = max(
        min_radius + 10,
        int(min_dim * 0.25)
    )

    min_distance = max(
        28,
        int(min_dim * 0.10)
    )

    candidates = []

    for param2 in (32, 28, 24, 20):

        try:
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1.15,
                minDist=min_distance,
                param1=100,
                param2=param2,
                minRadius=min_radius,
                maxRadius=max_radius
            )
        except Exception:
            circles = None

        if circles is None:
            continue

        circles = np.round(
            circles[0]
        ).astype(int)

        for x, y, radius in circles:

            if radius <= 0:
                continue

            padding = max(
                5,
                int(radius * 0.16)
            )

            box = {
                "x1": x - radius - padding,
                "y1": y - radius - padding,
                "x2": x + radius + padding,
                "y2": y + radius + padding
            }

            box = _clip_box(
                box,
                width,
                height
            )

            bw = box["x2"] - box["x1"]
            bh = box["y2"] - box["y1"]

            if bw < 35 or bh < 35:
                continue

            candidates.append(
                _create_candidate(
                    box,
                    width,
                    height,
                    0.88,
                    "round_object"
                )
            )

        if len(candidates) >= 3:
            break

    return candidates


def _detect_elongated_objects(image):
    height, width = image.shape[:2]
    total_area = height * width

    mask = _foreground_mask(image)

    candidates = []

    regions = [
        (
            0,
            int(height * 0.72),
            "top_elongated"
        ),
        (
            0,
            height,
            "elongated"
        )
    ]

    for start_y, end_y, source in regions:

        roi = mask[start_y:end_y, :]

        contours, _ = cv2.findContours(
            roi,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < total_area * 0.015:
                continue

            if area > total_area * 0.55:
                continue

            x, local_y, w, h = cv2.boundingRect(
                contour
            )

            y = local_y + start_y

            if w < 45 or h < 25:
                continue

            aspect = w / float(max(h, 1))

            if aspect < 1.35:
                continue

            rect_area = w * h

            if rect_area <= 0:
                continue

            fill_ratio = area / rect_area

            if fill_ratio < 0.18:
                continue

            box = {
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h
            }

            candidates.append(
                _create_candidate(
                    box,
                    width,
                    height,
                    0.82,
                    source
                )
            )

    return candidates


def _detect_contour_objects(image):
    height, width = image.shape[:2]
    total_area = height * width

    mask = _foreground_mask(image)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < total_area * 0.012:
            continue

        if area > total_area * 0.40:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if w < 30 or h < 30:
            continue

        box_area = w * h

        if box_area <= 0:
            continue

        fill_ratio = area / box_area

        if fill_ratio < 0.12:
            continue

        aspect = w / float(max(h, 1))

        if aspect < 0.25 or aspect > 5.5:
            continue

        candidates.append(
            _create_candidate(
                {
                    "x1": x,
                    "y1": y,
                    "x2": x + w,
                    "y2": y + h
                },
                width,
                height,
                0.58,
                "contour"
            )
        )

    return candidates


def _detect_grid_regions(image):
    height, width = image.shape[:2]

    candidates = []

    cols = 3
    rows = 2

    cell_w = width / cols
    cell_h = height / rows

    for row in range(rows):
        for col in range(cols):

            x1 = int(max(0, col * cell_w - cell_w * 0.10))
            y1 = int(max(0, row * cell_h - cell_h * 0.10))

            x2 = int(min(width, (col + 1) * cell_w + cell_w * 0.10))
            y2 = int(min(height, (row + 1) * cell_h + cell_h * 0.10))

            candidates.append(
                _create_candidate(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    },
                    width,
                    height,
                    0.35,
                    "grid"
                )
            )

    return candidates


def _deduplicate_candidates(candidates, max_candidates=8):
    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda item: item.get(
            "confidence",
            0
        ),
        reverse=True
    )

    selected = []

    for candidate in candidates:

        box = candidate["bbox"]

        duplicate = False

        for existing in selected:

            existing_box = existing["bbox"]

            iou = _iou(
                box,
                existing_box
            )

            smaller_overlap = _intersection_over_smaller(
                box,
                existing_box
            )

            if iou >= 0.55:
                duplicate = True
                break

            if smaller_overlap >= 0.82:
                duplicate = True
                break

        if duplicate:
            continue

        selected.append(candidate)

        if len(selected) >= max_candidates:
            break

    selected.sort(
        key=lambda item: (
            item["bbox"]["y1"],
            item["bbox"]["x1"]
        )
    )

    for index, candidate in enumerate(
        selected,
        start=1
    ):
        candidate["id"] = index

    return selected


def find_salient_food_regions(image_bgr):
    if image_bgr is None:
        return []

    if not isinstance(
        image_bgr,
        np.ndarray
    ):
        return []

    if image_bgr.size == 0:
        return []

    height, width = image_bgr.shape[:2]

    if height <= 0 or width <= 0:
        return []

    try:

        max_dimension = max(
            height,
            width
        )

        if max_dimension > 900:

            scale = 900.0 / max_dimension

            working = cv2.resize(
                image_bgr,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA
            )

        else:
            working = image_bgr.copy()

        wh, ww = working.shape[:2]

        candidates = []

        round_candidates = _detect_round_objects(
            working
        )

        candidates.extend(
            round_candidates
        )

        elongated_candidates = _detect_elongated_objects(
            working
        )

        candidates.extend(
            elongated_candidates
        )

        contour_candidates = _detect_contour_objects(
            working
        )

        candidates.extend(
            contour_candidates
        )

        if len(candidates) < 2:

            grid_candidates = _detect_grid_regions(
                working
            )

            candidates.extend(
                grid_candidates
            )

        scale_x = width / float(ww)
        scale_y = height / float(wh)

        converted = []

        for candidate in candidates:

            box = candidate["bbox"]

            original_box = {
                "x1": int(box["x1"] * scale_x),
                "y1": int(box["y1"] * scale_y),
                "x2": int(box["x2"] * scale_x),
                "y2": int(box["y2"] * scale_y)
            }

            candidate["bbox"] = _clip_box(
                original_box,
                width,
                height
            )

            converted.append(candidate)

        result = _deduplicate_candidates(
            converted,
            max_candidates=8
        )

        if not result:

            result = [
                _create_candidate(
                    {
                        "x1": 0,
                        "y1": 0,
                        "x2": width,
                        "y2": height
                    },
                    width,
                    height,
                    0.30,
                    "full_image"
                )
            ]

        logger.info(
            f"[DETECTOR] Multi-object fallback generated "
            f"{len(result)} candidate(s)."
        )

        for candidate in result:

            logger.info(
                f"[DETECTOR] Candidate "
                f"{candidate['id']} | "
                f"source={candidate.get('source')} | "
                f"bbox={candidate['bbox']} | "
                f"confidence={candidate['confidence']}"
            )

        return result

    except Exception as error:

        logger.exception(
            f"[DETECTOR] Localization failed: {error}"
        )

        return []


def detect_objects_in_image(
    image_bgr: np.ndarray
) -> Tuple[
    List[Dict[str, Any]],
    bool,
    Optional[str]
]:

    started_at = time.perf_counter()

    logger.info(
        "[DETECTOR] Detection started."
    )

    if image_bgr is None:
        return [], True, "Invalid image."

    if not isinstance(
        image_bgr,
        np.ndarray
    ):
        return [], True, "Invalid image format."

    if image_bgr.size == 0:
        return [], True, "Empty image."

    try:

        height, width = image_bgr.shape[:2]

        logger.info(
            f"[DETECTOR] Image shape: {height}x{width}"
        )

        if height < 10 or width < 10:
            return [], True, "Image resolution is too small."

        logger.info(
            "[DETECTOR] YOLO disabled."
        )

        logger.info(
            "[DETECTOR] Running lightweight multi-object localization."
        )

        candidates = find_salient_food_regions(
            image_bgr
        )

        elapsed = (
            time.perf_counter() - started_at
        ) * 1000

        logger.info(
            f"[DETECTOR] Detection completed | "
            f"objects={len(candidates)} | "
            f"time={elapsed:.2f} ms"
        )

        return candidates, False, None

    except Exception as error:

        logger.exception(
            f"[DETECTOR] Detection failed: {error}"
        )

        return [], True, "Unable to analyze image."