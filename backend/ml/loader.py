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

ENABLE_YOLO_DETECTOR = (
    os.getenv("FRESH_LENS_ENABLE_YOLO", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)


def get_classes_metadata() -> Dict[str, Any]:
    global _classes_metadata

    if _classes_metadata is not None:
        return _classes_metadata

    with _model_lock:
        if _classes_metadata is not None:
            return _classes_metadata

        if not os.path.exists(CLASSES_CONFIG_PATH):
            raise FileNotFoundError(
                f"Classes configuration file not found at: {CLASSES_CONFIG_PATH}"
            )

        with open(CLASSES_CONFIG_PATH, "r", encoding="utf-8") as f:
            _classes_metadata = json.load(f)

        logger.info(
            f"Loaded {len(_classes_metadata.get('classes', {}))} classes from "
            f"{CLASSES_CONFIG_PATH}"
        )

        return _classes_metadata


def get_legacy_class_indices() -> Dict[str, int]:
    global _legacy_class_indices
    global _legacy_class_names

    if _legacy_class_indices is not None:
        return _legacy_class_indices

    with _model_lock:
        if _legacy_class_indices is not None:
            return _legacy_class_indices

        candidates = [
            MODELS_DIR / "production" / "class_indices.json",
            MODELS_DIR / "class_indices.json",
            LEGACY_BACKEND_MODELS_DIR / "class_indices.json"
        ]

        chosen_path = None

        for path in candidates:
            if path.exists():
                chosen_path = path
                break

        if chosen_path is None:
            raise FileNotFoundError(
                f"class_indices.json not found in any standard path: {candidates}"
            )

        with open(chosen_path, "r", encoding="utf-8") as f:
            _legacy_class_indices = json.load(f)

        _legacy_class_names = {
            int(value): key
            for key, value in _legacy_class_indices.items()
        }

        logger.info(
            f"Loaded {len(_legacy_class_indices)} legacy class indices "
            f"from {chosen_path}"
        )

        return _legacy_class_indices


def get_legacy_class_names() -> Dict[int, str]:
    global _legacy_class_names

    if _legacy_class_names is None:
        get_legacy_class_indices()

    return _legacy_class_names


def _load_tflite_model():
    global _tflite_freshness_interpreter

    if _tflite_freshness_interpreter is not None:
        return _tflite_freshness_interpreter

    logger.info(
        f"Loading FreshLens AI TFLite model from {TFLITE_MODEL_PATH}..."
    )

    if not os.path.exists(TFLITE_MODEL_PATH):
        raise FileNotFoundError(
            f"TFLite model not found at {TFLITE_MODEL_PATH}"
        )

    try:
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(
            model_path=str(TFLITE_MODEL_PATH)
        )
    except ImportError:
        import tflite_runtime.interpreter as tflite

        interpreter = tflite.Interpreter(
            model_path=str(TFLITE_MODEL_PATH)
        )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()

    if input_details:
        input_shape = input_details[0]["shape"]

        if len(input_shape) == 4:
            dummy = np.zeros(
                tuple(input_shape),
                dtype=np.float32
            )

            try:
                interpreter.set_tensor(
                    input_details[0]["index"],
                    dummy
                )
                interpreter.invoke()
                logger.info(
                    "TFLite freshness model warmup completed successfully."
                )
            except Exception as warmup_error:
                logger.warning(
                    f"TFLite warmup skipped: {warmup_error}"
                )

    _tflite_freshness_interpreter = interpreter

    logger.info(
        "TFLite freshness model loaded and allocated successfully."
    )

    return _tflite_freshness_interpreter


def _load_keras_model():
    global _keras_freshness_model

    if _keras_freshness_model is not None:
        return _keras_freshness_model

    logger.info(
        f"Loading FreshLens AI Keras model from {KERAS_MODEL_PATH}..."
    )

    if not os.path.exists(KERAS_MODEL_PATH):
        raise FileNotFoundError(
            f"Keras model not found at {KERAS_MODEL_PATH}"
        )

    import tensorflow as tf

    model = tf.keras.models.load_model(
        KERAS_MODEL_PATH,
        compile=False
    )

    dummy = np.zeros(
        (1, IMG_SIZE, IMG_SIZE, 3),
        dtype=np.float32
    )

    model.predict(
        dummy,
        verbose=0
    )

    _keras_freshness_model = model

    logger.info(
        "Keras freshness model loaded and warmed up successfully."
    )

    return _keras_freshness_model


def get_freshness_model():
    global _keras_freshness_model
    global _tflite_freshness_interpreter

    with _model_lock:
        if USE_TFLITE:
            if _tflite_freshness_interpreter is not None:
                return _tflite_freshness_interpreter

            return _load_tflite_model()

        if _keras_freshness_model is not None:
            return _keras_freshness_model

        if os.path.exists(KERAS_MODEL_PATH):
            return _load_keras_model()

        if os.path.exists(TFLITE_MODEL_PATH):
            logger.warning(
                f"Keras model not found at {KERAS_MODEL_PATH}. "
                "Using TFLite model instead."
            )
            return _load_tflite_model()

        raise FileNotFoundError(
            "Neither Keras nor TFLite freshness model was found."
        )


def get_detector_model():
    global _detector_model

    if _detector_model is not None:
        return _detector_model

    if not ENABLE_YOLO_DETECTOR:
        logger.info(
            "YOLO detector disabled. Using lightweight image fallback."
        )
        return None

    with _model_lock:
        if _detector_model is not None:
            return _detector_model

        try:
            from ultralytics import YOLO

            local_weights = [
                MODELS_DIR / "production" / "yolov8n.pt",
                MODELS_DIR / "yolov8n.pt",
                Path("yolov8n.pt")
            ]

            chosen_weight = None

            for weight_path in local_weights:
                if weight_path.exists():
                    chosen_weight = str(weight_path)
                    break

            if chosen_weight is None:
                logger.warning(
                    "YOLO weights not found. Using lightweight image fallback."
                )
                return None

            logger.info(
                f"Loading Ultralytics YOLO detector with weights: "
                f"{chosen_weight}"
            )

            _detector_model = YOLO(chosen_weight)

            logger.info(
                "Ultralytics YOLO detector loaded successfully."
            )

            return _detector_model

        except Exception as error:
            logger.warning(
                f"YOLO detector unavailable: {error}. "
                "Using lightweight image fallback."
            )
            _detector_model = None
            return None