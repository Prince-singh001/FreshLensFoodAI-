import threading
import time
import cv2
import numpy as np

try:
    from config import PRODUCTION_MODEL_DIR, USE_TFLITE
    from utils.logger import logger
except ImportError:
    from backend.config import PRODUCTION_MODEL_DIR, USE_TFLITE
    from backend.utils.logger import logger


FOOD_VALIDATION_THRESHOLD = 0.60

_validator_tflite_interpreter = None
_validator_keras_model = None

_validator_lock = threading.RLock()


def get_food_validator_model():

    global _validator_tflite_interpreter
    global _validator_keras_model

    with _validator_lock:

        tflite_path = (
            PRODUCTION_MODEL_DIR
            / "food_validator_model.tflite"
        )

        h5_path = (
            PRODUCTION_MODEL_DIR
            / "food_validator_model.h5"
        )

        if USE_TFLITE or not h5_path.exists():

            if (
                _validator_tflite_interpreter is None
                and tflite_path.exists()
            ):

                try:

                    import tensorflow as tf

                    logger.info(
                        "[VALIDATOR] Loading Food Validator TFLite model..."
                    )

                    _validator_tflite_interpreter = (
                        tf.lite.Interpreter(
                            model_path=str(tflite_path),
                            num_threads=1
                        )
                    )

                    _validator_tflite_interpreter.allocate_tensors()

                    logger.info(
                        "[VALIDATOR] Food Validator TFLite loaded successfully."
                    )

                except Exception as e:

                    logger.exception(
                        f"[VALIDATOR] TFLite load failed: {e}"
                    )

                    _validator_tflite_interpreter = None

            return _validator_tflite_interpreter

        if (
            _validator_keras_model is None
            and h5_path.exists()
        ):

            try:

                import tensorflow as tf

                logger.info(
                    "[VALIDATOR] Loading Food Validator Keras model..."
                )

                _validator_keras_model = (
                    tf.keras.models.load_model(
                        str(h5_path),
                        compile=False
                    )
                )

                logger.info(
                    "[VALIDATOR] Food Validator Keras loaded successfully."
                )

            except Exception as e:

                logger.exception(
                    f"[VALIDATOR] Keras load failed: {e}"
                )

                _validator_keras_model = None

        return _validator_keras_model


def preprocess_for_validator(
    image_bgr: np.ndarray,
    target_size: int = 224
) -> np.ndarray:

    if image_bgr is None:
        raise ValueError(
            "Invalid image."
        )

    if image_bgr.size == 0:
        raise ValueError(
            "Empty image."
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    resized = cv2.resize(
        image_rgb,
        (
            target_size,
            target_size
        ),
        interpolation=cv2.INTER_AREA
    )

    normalized = (
        resized.astype(
            np.float32
        )
        / 255.0
    )

    return np.expand_dims(
        normalized,
        axis=0
    )


def validate_food_presence(
    image_bgr,
    threshold=FOOD_VALIDATION_THRESHOLD
):

    started = time.perf_counter()

    try:

        model = get_food_validator_model()

        if model is None:

            logger.warning(
                "[VALIDATOR] Food validator unavailable. "
                "Using safe fallback."
            )

            return (
                True,
                0.70,
                0.30
            )

        input_tensor = preprocess_for_validator(
            image_bgr
        )

        with _validator_lock:

            if hasattr(
                model,
                "get_input_details"
            ):

                input_details = (
                    model.get_input_details()
                )

                output_details = (
                    model.get_output_details()
                )

                if not input_details:
                    raise RuntimeError(
                        "Validator has no input tensor."
                    )

                if not output_details:
                    raise RuntimeError(
                        "Validator has no output tensor."
                    )

                input_info = input_details[0]

                target_dtype = input_info.get(
                    "dtype",
                    np.float32
                )

                if target_dtype != np.float32:

                    scale, zero_point = (
                        input_info.get(
                            "quantization",
                            (0.0, 0)
                        )
                    )

                    if scale and scale > 0:

                        input_tensor = (
                            input_tensor
                            / float(scale)
                        ) + float(zero_point)

                    input_tensor = input_tensor.astype(
                        target_dtype
                    )

                else:

                    input_tensor = input_tensor.astype(
                        np.float32
                    )

                model.set_tensor(
                    input_info["index"],
                    input_tensor
                )

                model.invoke()

                output_info = output_details[0]

                output = model.get_tensor(
                    output_info["index"]
                )

                raw_prob = float(
                    np.asarray(output).reshape(-1)[0]
                )

            else:

                output = model(
                    input_tensor,
                    training=False
                )

                raw_prob = float(
                    output.numpy()
                    .reshape(-1)[0]
                )

        raw_prob = float(
            np.clip(
                raw_prob,
                0.0,
                1.0
            )
        )

        p_non_food = raw_prob
        p_food = 1.0 - raw_prob

        is_food = bool(
            p_food >= threshold
        )

        elapsed = (
            time.perf_counter()
            - started
        ) * 1000

        logger.info(
            "[VALIDATOR] Completed | "
            f"food={is_food} | "
            f"p_food={p_food:.4f} | "
            f"p_non_food={p_non_food:.4f} | "
            f"time={elapsed:.2f}ms"
        )

        return (
            is_food,
            p_food,
            p_non_food
        )

    except Exception as e:

        elapsed = (
            time.perf_counter()
            - started
        ) * 1000

        logger.exception(
            "[VALIDATOR] Inference failed | "
            f"time={elapsed:.2f}ms | error={e}"
        )

        return (
            False,
            0.0,
            1.0
        )