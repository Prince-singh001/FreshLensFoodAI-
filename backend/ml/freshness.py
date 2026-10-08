import cv2
import numpy as np
import time
from typing import Dict, Any, Tuple, Optional

try:
    from ml.loader import (
        get_freshness_model,
        get_legacy_class_indices,
        get_legacy_class_names,
        get_classes_metadata,
        _model_lock
    )

    from config import (
        USE_TFLITE,
        FRESHNESS_CONFIDENCE_THRESHOLD,
        FOOD_CONFIDENCE_THRESHOLD
    )

    from utils.image_utils import preprocess_for_freshness
    from utils.logger import logger

except ImportError:

    from backend.ml.loader import (
        get_freshness_model,
        get_legacy_class_indices,
        get_legacy_class_names,
        get_classes_metadata,
        _model_lock
    )

    from backend.config import (
        USE_TFLITE,
        FRESHNESS_CONFIDENCE_THRESHOLD,
        FOOD_CONFIDENCE_THRESHOLD
    )

    from backend.utils.image_utils import preprocess_for_freshness
    from backend.utils.logger import logger


def parse_class_label(
    label: str
) -> Tuple[str, str, str]:

    clean = str(
        label
    ).lower().strip()

    if clean.startswith("fresh"):

        condition = "Fresh"
        raw_item = clean[5:]

    elif clean.startswith("spoiled"):

        condition = "Spoiled"
        raw_item = clean[7:]

    elif clean.startswith("spoile"):

        condition = "Spoiled"
        raw_item = clean[6:]

    else:

        condition = "Unknown"
        raw_item = clean

    raw_item = (
        raw_item
        .replace("_", " ")
        .replace("-", " ")
    )

    raw_item = (
        raw_item
        .replace(
            "bittergroud",
            "bitter gourd"
        )
        .replace(
            "bittergourd",
            "bitter gourd"
        )
        .strip()
    )

    key_mapping = {
        "apples": "apple",
        "apple": "apple",
        "banana": "banana",
        "bananas": "banana",
        "oranges": "orange",
        "orange": "orange",
        "tomato": "tomato",
        "tomatoes": "tomato",
        "potato": "potato",
        "potatoes": "potato",
        "cucumber": "cucumber",
        "cucumbers": "cucumber",
        "bitter gourd": "bitter_gourd",
        "onion": "onion",
        "onions": "onion",
        "carrot": "carrot",
        "carrots": "carrot"
    }

    canonical_key = key_mapping.get(
        raw_item,
        raw_item.replace(
            " ",
            "_"
        )
    )

    display_names = {
        "apple": "Apple",
        "banana": "Banana",
        "orange": "Orange",
        "tomato": "Tomato",
        "potato": "Potato",
        "cucumber": "Cucumber",
        "bitter_gourd": "Bitter Gourd",
        "onion": "Onion",
        "carrot": "Carrot"
    }

    display_name = display_names.get(
        canonical_key,
        canonical_key.replace(
            "_",
            " "
        ).title()
    )

    return (
        canonical_key,
        display_name,
        condition
    )


def _prepare_tflite_input(
    image: np.ndarray,
    input_details: Dict[str, Any]
) -> np.ndarray:

    tensor = preprocess_for_freshness(
        image
    )

    target_dtype = input_details.get(
        "dtype",
        np.float32
    )

    if target_dtype == np.float32:

        return tensor.astype(
            np.float32
        )

    scale, zero_point = (
        input_details.get(
            "quantization",
            (0.0, 0)
        )
    )

    if scale and scale > 0:

        tensor = (
            tensor / float(scale)
        ) + float(zero_point)

    return tensor.astype(
        target_dtype
    )


def _normalize_probabilities(
    values: np.ndarray
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=np.float32
    ).reshape(-1)

    if values.size == 0:

        return values

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    if np.any(values < 0):

        shifted = (
            values
            - np.max(values)
        )

        exponent = np.exp(
            shifted
        )

        denominator = float(
            exponent.sum()
        )

        if denominator > 0:

            values = (
                exponent
                / denominator
            )

    total = float(
        values.sum()
    )

    if total > 0:

        values = (
            values
            / total
        )

    return values.astype(
        np.float32
    )


def run_freshness_inference(
    cropped_image_bgr: np.ndarray
) -> np.ndarray:

    if cropped_image_bgr is None:

        raise ValueError(
            "Invalid crop."
        )

    if cropped_image_bgr.size == 0:

        raise ValueError(
            "Empty crop."
        )

    started = time.perf_counter()

    model = get_freshness_model()

    if model is None:

        raise RuntimeError(
            "Freshness model is unavailable."
        )

    if USE_TFLITE:

        with _model_lock:

            input_details = (
                model.get_input_details()
            )

            output_details = (
                model.get_output_details()
            )

            if not input_details:

                raise RuntimeError(
                    "TFLite model has no input tensor."
                )

            if not output_details:

                raise RuntimeError(
                    "TFLite model has no output tensor."
                )

            input_info = input_details[0]

            input_tensor = (
                _prepare_tflite_input(
                    cropped_image_bgr,
                    input_info
                )
            )

            model.set_tensor(
                input_info["index"],
                input_tensor
            )

            model.invoke()

            output_info = output_details[0]

            raw_output = model.get_tensor(
                output_info["index"]
            )

            probs = np.asarray(
                raw_output,
                dtype=np.float32
            )

            scale, zero_point = (
                output_info.get(
                    "quantization",
                    (0.0, 0)
                )
            )

            if (
                scale
                and scale > 0
                and raw_output.dtype != np.float32
            ):

                probs = (
                    probs
                    - float(zero_point)
                ) * float(scale)

    else:

        input_tensor = (
            preprocess_for_freshness(
                cropped_image_bgr
            )
        )

        with _model_lock:

            output = model(
                input_tensor,
                training=False
            )

            probs = output.numpy()[0]

    result = _normalize_probabilities(
        probs
    )

    elapsed = (
        time.perf_counter()
        - started
    ) * 1000

    logger.info(
        "[FRESHNESS] Model inference completed | "
        f"time={elapsed:.2f}ms"
    )

    return result


def _get_food_scores(
    probs: np.ndarray,
    class_names
) -> Dict[str, float]:

    food_scores = {}

    for idx, prob in enumerate(
        probs[:len(class_names)]
    ):

        label = (
            class_names[idx]
            .lower()
            .strip()
        )

        canonical_key, _, _ = (
            parse_class_label(
                label
            )
        )

        food_scores[
            canonical_key
        ] = (
            food_scores.get(
                canonical_key,
                0.0
            )
            + float(prob)
        )

    return food_scores


def _find_pair_indices(
    item_key: str,
    class_indices: Dict[str, int]
):

    compact_key = (
        item_key
        .replace("_", "")
        .replace(" ", "")
    )

    fresh_candidates = [
        f"fresh{item_key}",
        f"fresh{item_key}s",
        f"fresh{compact_key}",
        f"fresh{compact_key}s"
    ]

    spoiled_candidates = [
        f"spoile{item_key}",
        f"spoile{item_key}s",
        f"spoile{compact_key}",
        f"spoile{compact_key}s",
        f"spoiled{item_key}",
        f"spoiled{item_key}s",
        f"spoiled{compact_key}",
        f"spoiled{compact_key}s"
    ]

    fresh_idx = None
    spoiled_idx = None

    for candidate in fresh_candidates:

        if candidate in class_indices:

            fresh_idx = class_indices[
                candidate
            ]

            break

    for candidate in spoiled_candidates:

        if candidate in class_indices:

            spoiled_idx = class_indices[
                candidate
            ]

            break

    return (
        fresh_idx,
        spoiled_idx
    )


def classify_food_crop(
    cropped_image_bgr: np.ndarray
) -> Tuple[str, str, str, float]:

    if (
        cropped_image_bgr is None
        or cropped_image_bgr.size == 0
    ):

        return (
            "unknown",
            "Unknown Food",
            "Food",
            0.0
        )

    probs = run_freshness_inference(
        cropped_image_bgr
    )

    class_names = (
        get_legacy_class_names()
    )

    classes_meta = (
        get_classes_metadata()
        .get(
            "classes",
            {}
        )
    )

    if (
        probs.size == 0
        or len(class_names) == 0
    ):

        return (
            "unknown",
            "Unknown Food",
            "Food",
            0.0
        )

    food_scores = _get_food_scores(
        probs,
        class_names
    )

    if not food_scores:

        return (
            "unknown",
            "Unknown Food",
            "Food",
            0.0
        )

    best_key, best_score = max(
        food_scores.items(),
        key=lambda x: x[1]
    )

    meta = classes_meta.get(
        best_key,
        {}
    )

    display_name = meta.get(
        "display_name",
        best_key.replace(
            "_",
            " "
        ).title()
    )

    category = meta.get(
        "category",
        "Produce"
    )

    logger.info(
        "[CLASSIFIER] Crop classified -> "
        f"{display_name} ({best_score:.4f})"
    )

    return (
        best_key,
        display_name,
        category,
        float(best_score)
    )


def evaluate_freshness_for_item(
    cropped_image_bgr: np.ndarray,
    item_key: Optional[str] = None
) -> Dict[str, Any]:

    if (
        cropped_image_bgr is None
        or cropped_image_bgr.size == 0
    ):

        return {
            "item_key": "unknown",
            "item": "Unknown",
            "category": "Produce",
            "freshness": None,
            "freshness_status": "Not Available",
            "freshness_confidence": 0.0,
            "is_confident": False
        }

    probs = run_freshness_inference(
        cropped_image_bgr
    )

    class_indices = (
        get_legacy_class_indices()
    )

    class_names = (
        get_legacy_class_names()
    )

    classes_meta = (
        get_classes_metadata()
        .get(
            "classes",
            {}
        )
    )

    if probs.size == 0:

        return {
            "item_key": "unknown",
            "item": "Unknown",
            "category": "Produce",
            "freshness": None,
            "freshness_status": "Not Available",
            "freshness_confidence": 0.0,
            "is_confident": False
        }

    if (
        not item_key
        or item_key == "unknown"
    ):

        food_scores = _get_food_scores(
            probs,
            class_names
        )

        if food_scores:

            item_key = max(
                food_scores.items(),
                key=lambda x: x[1]
            )[0]

        else:

            item_key = "unknown"

    if item_key in classes_meta:

        meta = classes_meta[
            item_key
        ]

        if not meta.get(
            "freshness_supported",
            False
        ):

            return {
                "item_key": item_key,
                "item": meta.get(
                    "display_name",
                    item_key
                ),
                "category": meta.get(
                    "category",
                    "Produce"
                ),
                "freshness": None,
                "freshness_status": "Not Available",
                "freshness_confidence": 0.0,
                "is_confident": True
            }

        fresh_idx, spoiled_idx = (
            _find_pair_indices(
                item_key,
                class_indices
            )
        )

        if (
            fresh_idx is not None
            and spoiled_idx is not None
            and fresh_idx < len(probs)
            and spoiled_idx < len(probs)
        ):

            p_fresh = float(
                probs[fresh_idx]
            )

            p_spoiled = float(
                probs[spoiled_idx]
            )

            total = (
                p_fresh
                + p_spoiled
            )

            if total > 1e-6:

                norm_fresh = (
                    p_fresh
                    / total
                )

                norm_spoiled = (
                    p_spoiled
                    / total
                )

            else:

                norm_fresh = 0.5
                norm_spoiled = 0.5

            condition = (
                "Fresh"
                if p_fresh >= p_spoiled
                else "Spoiled"
            )

            confidence = max(
                norm_fresh,
                norm_spoiled
            )

            stability_warning = ""

            if abs(
                norm_fresh
                - norm_spoiled
            ) < 0.12:

                stability_warning = (
                    "Borderline freshness prediction. "
                    "Inspect closely before use."
                )

            logger.info(
                "[FRESHNESS] "
                f"{meta.get('display_name', item_key)} -> "
                f"{condition} ({confidence:.4f})"
            )

            return {
                "item_key": item_key,
                "item": meta.get(
                    "display_name",
                    item_key
                ),
                "category": meta.get(
                    "category",
                    "Produce"
                ),
                "freshness": condition,
                "freshness_status": condition,
                "freshness_confidence": round(
                    confidence,
                    4
                ),
                "stability_warning": stability_warning,
                "is_confident": bool(
                    confidence
                    >= FRESHNESS_CONFIDENCE_THRESHOLD
                )
            }

    valid_count = min(
        len(probs),
        len(class_names)
    )

    if valid_count == 0:

        return {
            "item_key": "unknown",
            "item": "Unknown",
            "category": "Produce",
            "freshness": None,
            "freshness_status": "Not Available",
            "freshness_confidence": 0.0,
            "is_confident": False
        }

    ranked_indices = np.argsort(
        probs[:valid_count]
    )[::-1]

    best_idx = int(
        ranked_indices[0]
    )

    best_prob = float(
        probs[best_idx]
    )

    best_label = class_names[
        best_idx
    ]

    canonical_key, display_name, condition = (
        parse_class_label(
            best_label
        )
    )

    category = classes_meta.get(
        canonical_key,
        {}
    ).get(
        "category",
        "Produce"
    )

    return {
        "item_key": canonical_key,
        "item": display_name,
        "category": category,
        "freshness": condition,
        "freshness_status": condition,
        "freshness_confidence": round(
            best_prob,
            4
        ),
        "stability_warning": "",
        "is_confident": bool(
            best_prob
            >= FOOD_CONFIDENCE_THRESHOLD
        )
    }