import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

try:
    from ml.loader import get_detector_model, get_classes_metadata
    from config import CONFIDENCE_THRESHOLD
    from utils.logger import logger
except ImportError:
    from backend.ml.loader import get_detector_model, get_classes_metadata
    from backend.config import CONFIDENCE_THRESHOLD
    from backend.utils.logger import logger

# COCO food class labels and non-food class labels
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
    "bowl": "Food"  # Often contains salad/soup/rice
}

# Unambiguous non-food rejection categories from COCO
# (humans, indoor furniture, electronics, rooms, apparel, outdoor vehicles, animals)
NON_FOOD_REJECTION_CLASSES = {
    # People & apparel
    "person", "backpack", "umbrella", "handbag", "tie", "suitcase",
    # Furniture & household fixtures
    "chair", "couch", "bed", "dining table", "toilet", "bench",
    # Electronics & appliances
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    # Everyday non-food items
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    # Vehicles & outdoor
    "car", "truck", "bus", "train", "motorcycle", "bicycle", "airplane", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter",
    # Animals
    "dog", "cat", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "bird",
    # Sports items
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket"
}


def find_salient_food_regions(image_bgr: np.ndarray, min_area_ratio: float = 0.04) -> List[Dict[str, int]]:
    """
    Locates distinct produce/food regions in an image when produce items are displayed
    (e.g., tomatoes, potatoes, cucumbers, bitter gourds on a board/table).
    Returns list of bounding boxes: [{'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...}]
    """
    h, w = image_bgr.shape[:2]
    total_area = h * w

    # Morphological segmentation to locate foreground objects
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(blurred, 30, 120)
    combined = cv2.bitwise_or(otsu, edges)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    proposals = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < total_area * min_area_ratio:
            continue
        if area > total_area * 0.95:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = bw / float(bh)
        if aspect_ratio < 0.15 or aspect_ratio > 6.0:
            continue

        proposals.append({
            "x1": int(x),
            "y1": int(y),
            "x2": int(x + bw),
            "y2": int(y + bh)
        })

    proposals.sort(key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]), reverse=True)
    return proposals[:5]


def detect_objects_in_image(image_bgr: np.ndarray) -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
    """
    Executes multi-object detection pipeline.
    Returns:
        (detected_objects, is_rejected_non_food, rejection_reason)
    """
    h, w = image_bgr.shape[:2]
    detected_objects: List[Dict[str, Any]] = []
    detector = get_detector_model()

    if detector is not None:
        try:
            results = detector(image_bgr, verbose=False, conf=CONFIDENCE_THRESHOLD)
            
            non_food_detections = []
            food_detections = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = r.names.get(cls_id, "").lower()
                    conf = float(box.conf[0])
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = [int(v) for v in coords]

                    if cls_name in NON_FOOD_REJECTION_CLASSES and conf >= 0.45:
                        non_food_detections.append((cls_name, conf))
                    elif cls_name in COCO_FOOD_CLASSES and conf >= CONFIDENCE_THRESHOLD:
                        food_detections.append({
                            "detected_label": cls_name,
                            "display_name": COCO_FOOD_CLASSES[cls_name],
                            "confidence": round(conf, 4),
                            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                        })

            if food_detections:
                for i, fd in enumerate(food_detections, start=1):
                    fd["id"] = i
                    detected_objects.append(fd)
                return detected_objects, False, None

            # If unambiguous non-food detected (person, chair, laptop, dog, etc.) and no food:
            if non_food_detections:
                prominent_non_food = max(non_food_detections, key=lambda x: x[1])
                return [], True, f"Detected non-food object ({prominent_non_food[0].title()} - {int(prominent_non_food[1]*100)}%). Please upload or capture a food item."

        except Exception as e:
            logger.error(f"Error during YOLO detector execution: {e}")

    # Fallback to produce salient localization proposals
    proposals = find_salient_food_regions(image_bgr)
    if len(proposals) > 1:
        for i, prop in enumerate(proposals, start=1):
            detected_objects.append({
                "id": i,
                "detected_label": "produce_candidate",
                "display_name": "Produce Item",
                "confidence": 0.50,
                "is_candidate": True,
                "bbox": prop
            })
    else:
        bbox = proposals[0] if proposals else {"x1": 0, "y1": 0, "x2": w, "y2": h}
        detected_objects.append({
            "id": 1,
            "detected_label": "produce_candidate",
            "display_name": "Produce Item",
            "confidence": 0.50,
            "is_candidate": True,
            "bbox": bbox
        })

    return detected_objects, False, None
