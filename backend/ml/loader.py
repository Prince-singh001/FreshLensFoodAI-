import os
import json
import threading
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from config import (
        CLASSES_CONFIG_PATH,
        KERAS_MODEL_PATH,
        TFLITE_MODEL_PATH,
        USE_TFLITE,
        IMG_SIZE,
        MODELS_DIR,
        LEGACY_BACKEND_MODELS_DIR
    )
    from utils.logger import logger
except ImportError:
    from backend.config import (
        CLASSES_CONFIG_PATH,
        KERAS_MODEL_PATH,
        TFLITE_MODEL_PATH,
        USE_TFLITE,
        IMG_SIZE,
        MODELS_DIR,
        LEGACY_BACKEND_MODELS_DIR
    )
    from backend.utils.logger import logger

_model_lock = threading.Lock()

_classes_metadata: Optional[Dict[str, Any]] = None
_legacy_class_indices: Optional[Dict[str, int]] = None
_legacy_class_names: Optional[Dict[int, str]] = None

_keras_freshness_model = None
_tflite_freshness_interpreter = None
_detector_model = None


def get_classes_metadata() -> Dict[str, Any]:
    global _classes_metadata
    if _classes_metadata is not None:
        return _classes_metadata

    with _model_lock:
        if _classes_metadata is not None:
            return _classes_metadata

        if not os.path.exists(CLASSES_CONFIG_PATH):
            raise FileNotFoundError(f"Classes configuration file not found at: {CLASSES_CONFIG_PATH}")

        with open(CLASSES_CONFIG_PATH, "r", encoding="utf-8") as f:
            _classes_metadata = json.load(f)

        logger.info(f"Loaded {len(_classes_metadata.get('classes', {}))} classes from {CLASSES_CONFIG_PATH}")
        return _classes_metadata


def get_legacy_class_indices() -> Dict[str, int]:
    global _legacy_class_indices, _legacy_class_names
    if _legacy_class_indices is not None:
        return _legacy_class_indices

    with _model_lock:
        if _legacy_class_indices is not None:
            return _legacy_class_indices

        candidates = [
            MODELS_DIR / "production" / "class_indices.json",
            MODELS_DIR / "class_indices.json",
            LEGACY_BACKEND_MODELS_DIR / "class_indices.json",
        ]
        chosen_path = None
        for p in candidates:
            if p.exists():
                chosen_path = p
                break

        if not chosen_path:
            raise FileNotFoundError(f"class_indices.json not found in any standard path ({candidates})")

        with open(chosen_path, "r", encoding="utf-8") as f:
            _legacy_class_indices = json.load(f)

        _legacy_class_names = {int(v): k for k, v in _legacy_class_indices.items()}
        logger.info(f"Loaded {len(_legacy_class_indices)} legacy class indices from {chosen_path}")
        return _legacy_class_indices


def get_legacy_class_names() -> Dict[int, str]:
    if _legacy_class_names is None:
        get_legacy_class_indices()
    return _legacy_class_names


def get_freshness_model():
    """
    Thread-safely loads and warms up the freshness classifier (TFLite or Keras) once.
    """
    global _keras_freshness_model, _tflite_freshness_interpreter

    with _model_lock:
        if USE_TFLITE:
            if _tflite_freshness_interpreter is not None:
                return _tflite_freshness_interpreter

            logger.info(f"Loading FreshLens AI TFLite model from {TFLITE_MODEL_PATH}...")
            if not os.path.exists(TFLITE_MODEL_PATH):
                raise FileNotFoundError(f"TFLite model not found at {TFLITE_MODEL_PATH}")

            try:
                import tensorflow as tf
                interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
            except ImportError:
                import tflite_runtime.interpreter as tflite
                interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)

            interpreter.allocate_tensors()
            _tflite_freshness_interpreter = interpreter
            logger.info("TFLite freshness model loaded and allocated successfully.")
            return _tflite_freshness_interpreter

        else:
            if _keras_freshness_model is not None:
                return _keras_freshness_model

            logger.info(f"Loading FreshLens AI Keras model from {KERAS_MODEL_PATH}...")
            if not os.path.exists(KERAS_MODEL_PATH):
                # Fallback to TFLite if Keras .h5 is missing
                if os.path.exists(TFLITE_MODEL_PATH):
                    logger.warning(f"Keras model not found at {KERAS_MODEL_PATH}, falling back to TFLite model.")
                    return get_freshness_model()
                raise FileNotFoundError(f"Keras model not found at {KERAS_MODEL_PATH}")

            import tensorflow as tf
            model = tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False)

            # Warm up model to ensure instant response on first real request
            logger.info("Warming up Keras model with dummy tensor...")
            dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            model.predict(dummy, verbose=0)
            _keras_freshness_model = model
            logger.info("Keras freshness model loaded and warmed up successfully.")
            return _keras_freshness_model


def get_detector_model():
    """
    Loads YOLO detector model once. Prefers ultralytics if available;
    otherwise falls back to OpenCV DNN or local detector.
    """
    global _detector_model
    if _detector_model is not None:
        return _detector_model

    with _model_lock:
        if _detector_model is not None:
            return _detector_model

        try:
            from ultralytics import YOLO
            # Check for local production yolov8n weights
            local_weights = [
                MODELS_DIR / "production" / "yolov8n.pt",
                MODELS_DIR / "yolov8n.pt",
                Path("yolov8n.pt")
            ]
            chosen_weight = "yolov8n.pt"
            for w in local_weights:
                if w.exists():
                    chosen_weight = str(w)
                    break

            logger.info(f"Loading Ultralytics YOLO detector with weights: {chosen_weight}")
            _detector_model = YOLO(chosen_weight)
            logger.info("Ultralytics YOLO detector loaded successfully.")
            return _detector_model
        except Exception as e:
            logger.warning(f"Ultralytics YOLO unavailable or failed to load: {e}. Using fallback detector.")
            _detector_model = None
            return None
