import os
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any

try:
    from utils.image_utils import check_image_quality
    from utils.logger import logger
except ImportError:
    from backend.utils.image_utils import check_image_quality
    from backend.utils.logger import logger

def validate_image_for_inference(image_path: str) -> Tuple[bool, Optional[str], Optional[str], Optional[np.ndarray], Dict[str, Any]]:
    """
    Validates image file existence, decodes into BGR matrix, and performs quality analysis.
    Returns:
        (is_valid, error_code, error_message, image_bgr, quality_metrics)
    """
    if not os.path.exists(image_path):
        return False, "FILE_NOT_FOUND", "The specified image file could not be found.", None, {}

    try:
        # Load image via OpenCV
        image_bgr = cv2.imread(image_path)
        if image_bgr is None or image_bgr.size == 0:
            return False, "CORRUPT_IMAGE", "Could not decode image. The file appears to be damaged or not a valid image format.", None, {}

        is_acceptable, rejection_reason, metrics = check_image_quality(image_bgr)
        if not is_acceptable:
            return False, "POOR_IMAGE_QUALITY", rejection_reason, image_bgr, metrics

        return True, None, None, image_bgr, metrics

    except Exception as e:
        logger.error(f"Unexpected error in validate_image_for_inference: {e}")
        return False, "IMAGE_PROCESSING_ERROR", "An error occurred while inspecting the image.", None, {}
