import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Tuple, Dict, Any, Optional, List
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

try:
    from config import (
        ALLOWED_EXTENSIONS,
        ALLOWED_MIME_TYPES,
        MAX_CONTENT_LENGTH,
        MIN_IMAGE_DIMENSION,
        MAX_IMAGE_DIMENSION,
        BLUR_THRESHOLD,
        DARKNESS_THRESHOLD,
        BRIGHTNESS_THRESHOLD,
        IMG_SIZE
    )
    from utils.logger import logger
except ImportError:
    from backend.config import (
        ALLOWED_EXTENSIONS,
        ALLOWED_MIME_TYPES,
        MAX_CONTENT_LENGTH,
        MIN_IMAGE_DIMENSION,
        MAX_IMAGE_DIMENSION,
        BLUR_THRESHOLD,
        DARKNESS_THRESHOLD,
        BRIGHTNESS_THRESHOLD,
        IMG_SIZE
    )
    from backend.utils.logger import logger

def validate_uploaded_file(file: FileStorage) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validates uploaded file against extension, MIME type, size, and image integrity.
    Returns: (is_valid, secure_file_name, error_message)
    """
    if not file or not file.filename:
        return False, None, "No file was selected for upload."

    raw_filename = file.filename.strip()
    if "." not in raw_filename:
        return False, None, "File lacks a valid extension."

    ext = raw_filename.rsplit(".", 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, None, f"Unsupported file extension '.{ext}'. Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}."

    # Validate MIME type if provided by client
    if file.mimetype and file.mimetype.lower() not in ALLOWED_MIME_TYPES:
        # Some clients send application/octet-stream; we will verify real content below
        pass

    # Read first chunk to check size and content integrity
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size == 0:
        return False, None, "The uploaded file is empty (0 bytes)."

    if file_size > MAX_CONTENT_LENGTH:
        limit_mb = MAX_CONTENT_LENGTH / (1024 * 1024)
        return False, None, f"File size exceeds the {limit_mb:.1f}MB upload limit."

    # Verify image integrity via PIL without saving to disk
    try:
        header = file.read(4096)
        file.seek(0)
        img = Image.open(file)
        img.verify()
        file.seek(0)
    except Exception as e:
        logger.warning(f"Image integrity verification failed: {e}")
        return False, None, "Corrupted or unreadable image file. Please upload a valid image."

    safe_name = secure_filename(raw_filename)
    return True, safe_name, None


def check_image_quality(image_bgr: np.ndarray) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Evaluates image resolution, blurriness, darkness, and overexposure.
    Returns: (is_acceptable, rejection_reason, quality_metrics)
    """
    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        return False, "Image could not be decoded.", {}

    h, w = image_bgr.shape[:2]
    if h < MIN_IMAGE_DIMENSION or w < MIN_IMAGE_DIMENSION:
        return False, f"Image resolution ({w}x{h}) is too low. Minimum required is {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}px.", {"width": w, "height": h}

    if h > MAX_IMAGE_DIMENSION or w > MAX_IMAGE_DIMENSION:
        # Scale down rather than rejecting overly large images
        scale = MAX_IMAGE_DIMENSION / max(h, w)
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = image_bgr.shape[:2]

    # Convert to grayscale for illumination & blur checks
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))

    # Laplacian variance for blur check
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    metrics = {
        "width": w,
        "height": h,
        "brightness": round(mean_brightness, 2),
        "blur_score": round(laplacian_var, 2)
    }

    if mean_brightness < DARKNESS_THRESHOLD:
        return False, "Image is too dark to reliably analyze. Please upload an image with adequate lighting.", metrics

    if mean_brightness > BRIGHTNESS_THRESHOLD:
        return False, "Image is excessively overexposed or completely white. Please provide a balanced photo.", metrics

    if laplacian_var < BLUR_THRESHOLD:
        return False, "Image is too blurry. Please upload a sharper, steady image of your food.", metrics

    return True, None, metrics


def crop_bounding_box(image_bgr: np.ndarray, bbox: Dict[str, int], pad_pct: float = 0.05) -> np.ndarray:
    """
    Extracts the image region inside bbox with optional margin padding.
    bbox format: {'x1': int, 'y1': int, 'x2': int, 'y2': int}
    """
    h, w = image_bgr.shape[:2]
    x1, y1 = bbox["x1"], bbox["y1"]
    x2, y2 = bbox["x2"], bbox["y2"]

    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * pad_pct)
    pad_y = int(bh * pad_pct)

    x1_pad = max(0, x1 - pad_x)
    y1_pad = max(0, y1 - pad_y)
    x2_pad = min(w, x2 + pad_x)
    y2_pad = min(h, y2 + pad_y)

    cropped = image_bgr[y1_pad:y2_pad, x1_pad:x2_pad]
    if cropped.size == 0:
        return image_bgr
    return cropped


def preprocess_for_freshness(image_bgr: np.ndarray, target_size: Tuple[int, int] = (IMG_SIZE, IMG_SIZE)) -> np.ndarray:
    """
    Resizes, transforms BGR->RGB, normalizes [0, 1], and batches image for Keras/TFLite model.
    """
    resized = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    batched = np.expand_dims(normalized, axis=0)
    return batched


def draw_annotations(image_bgr: np.ndarray, detected_objects: List[Dict[str, Any]]) -> np.ndarray:
    """
    Renders high-contrast bounding boxes, label pills, and confidence tags on image.
    Colors:
      - Fresh Produce: Vibrant Emerald Green
      - Spoiled Produce: Crimson Coral Red
      - General Food / Not Available: Sky Cyan
    """
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2] 

    color_map = {
        "Fresh": (16, 185, 129),     # Emerald Green in BGR
        "Spoiled": (68, 68, 239),     # Crimson Red in BGR
        "Not Available": (246, 130, 59), # Sky Blue in BGR
        "Unknown": (11, 158, 245)     # Amber in BGR
    }

    for obj in detected_objects:
        bbox = obj.get("bbox")
        if not bbox:
            continue

        x1 = max(0, int(bbox.get("x1", 0)))
        y1 = max(0, int(bbox.get("y1", 0)))
        x2 = min(w, int(bbox.get("x2", w)))
        y2 = min(h, int(bbox.get("y2", h)))

        condition = obj.get("freshness") or "Not Available"
        color = color_map.get(condition, (246, 130, 59))

        # Draw smooth rounded-corner rectangle or clean bounding box
        line_thickness = max(2, int(min(w, h) * 0.004))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

        # Build label text
        item_name = obj.get("item", "Item")
        det_conf = obj.get("detection_confidence", 0.0)
        
        if condition in ("Fresh", "Spoiled"):
            label_text = f"{item_name} | {condition} ({int(det_conf * 100)}%)"
        else:
            label_text = f"{item_name} ({int(det_conf * 100)}%)"

        # Calculate text sizing
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, min(w, h) * 0.0007)
        font_thickness = max(1, int(font_scale * 2))
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

        # Draw label background badge
        badge_y1 = max(0, y1 - text_h - 10)
        badge_y2 = y1
        badge_x1 = x1
        badge_x2 = min(w, x1 + text_w + 14)

        cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
        # White text inside badge
        cv2.putText(
            annotated,
            label_text,
            (badge_x1 + 7, badge_y2 - 5),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA
        )

    return annotated
