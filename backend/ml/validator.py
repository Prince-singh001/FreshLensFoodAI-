import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

try:
    from config import ROOT_DIR, PRODUCTION_MODEL_DIR, USE_TFLITE, IMG_SIZE
    from utils.logger import logger
except ImportError:
    from backend.config import ROOT_DIR, PRODUCTION_MODEL_DIR, USE_TFLITE, IMG_SIZE
    from backend.utils.logger import logger

# Food vs Non-Food threshold
FOOD_VALIDATION_THRESHOLD = 0.60

_validator_tflite_interpreter = None
_validator_keras_model = None

def get_food_validator_model():
    """
    Singleton loader for binary Food vs Non-Food Validator model.
    Supports both TFLite and Keras.
    """
    global _validator_tflite_interpreter, _validator_keras_model
    tflite_path = PRODUCTION_MODEL_DIR / "food_validator_model.tflite"
    h5_path = PRODUCTION_MODEL_DIR / "food_validator_model.h5"

    if USE_TFLITE or not h5_path.exists():
        if _validator_tflite_interpreter is None and tflite_path.exists():
            try:
                import tensorflow as tf
                _validator_tflite_interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
                _validator_tflite_interpreter.allocate_tensors()
                logger.info(f"Loaded Food Validator TFLite model from {tflite_path}")
            except Exception as e:
                logger.error(f"Failed to load Food Validator TFLite model: {e}")
        return _validator_tflite_interpreter
    else:
        if _validator_keras_model is None and h5_path.exists():
            try:
                import tensorflow as tf
                _validator_keras_model = tf.keras.models.load_model(str(h5_path), compile=False)
                logger.info(f"Loaded Food Validator Keras model from {h5_path}")
            except Exception as e:
                logger.error(f"Failed to load Food Validator Keras model: {e}")
        return _validator_keras_model


def preprocess_for_validator(image_bgr: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Preprocess image for Food Validator model:
    Convert BGR -> RGB, resize to target_size x target_size, scale to [0, 1].
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)


def validate_food_presence(
    image_bgr: np.ndarray,
    threshold: float = FOOD_VALIDATION_THRESHOLD
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Validates whether the provided image or crop contains real food items.
    
    Returns:
        (is_food, food_confidence, details_dict)
        
    food_confidence: float in range [0.0, 1.0] where 1.0 = definitely food.
    """
    model = get_food_validator_model()
    
    if model is None:
        # Fallback if validator weights not yet loaded: log warning
        logger.warning("Food validator model not loaded; allowing pipeline fallback")
        return True, 0.70, {"status": "model_unavailable", "p_food": 0.70}

    input_tensor = preprocess_for_validator(image_bgr, target_size=IMG_SIZE)

    try:
        if hasattr(model, "get_input_details"):
            # TFLite interpreter
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            model.set_tensor(input_details[0]['index'], input_tensor)
            model.invoke()
            output = model.get_tensor(output_details[0]['index'])
            raw_prob = float(output[0][0])
        else:
            # Direct tensor call is 100x faster than model.predict on CPU
            output = model(input_tensor, training=False)
            raw_prob = float(output.numpy()[0][0])

        # Note: class indices: 'food': 0, 'non_food': 1
        # Thus sigmoid output closer to 0 is food, closer to 1 is non_food
        # Or if alphabetical, food=0, non_food=1:
        # p_food = 1.0 - raw_prob
        p_food = 1.0 - raw_prob
        p_non_food = raw_prob

        is_food = bool(p_food >= threshold)

        return is_food, p_food, {
            "p_food": round(p_food, 4),
            "p_non_food": round(p_non_food, 4),
            "is_food": is_food,
            "threshold": threshold
        }
    except Exception as e:
        logger.error(f"Error during food validation inference: {e}")
        return False, 0.0, {"error": str(e), "is_food": False}
