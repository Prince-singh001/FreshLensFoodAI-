import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

try:
    from ml.loader import get_freshness_model, get_legacy_class_indices, get_legacy_class_names, get_classes_metadata
    from config import USE_TFLITE, FRESHNESS_CONFIDENCE_THRESHOLD, FOOD_CONFIDENCE_THRESHOLD, IMG_SIZE
    from utils.image_utils import preprocess_for_freshness
    from utils.logger import logger
except ImportError:
    from backend.ml.loader import get_freshness_model, get_legacy_class_indices, get_legacy_class_names, get_classes_metadata
    from backend.config import USE_TFLITE, FRESHNESS_CONFIDENCE_THRESHOLD, FOOD_CONFIDENCE_THRESHOLD, IMG_SIZE
    from backend.utils.image_utils import preprocess_for_freshness
    from backend.utils.logger import logger

def parse_class_label(label: str) -> Tuple[str, str, str]:
    """
    Parses legacy class labels like 'freshapples', 'spoilebittergroud'
    into: (canonical_item_key, item_display_name, freshness_condition)
    """
    clean = label.lower().strip()
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

    raw_item = raw_item.replace("_", " ").replace("-", " ")
    raw_item = raw_item.replace("bittergroud", "bitter gourd").replace("bittergourd", "bitter gourd")
    raw_item = raw_item.strip()

    # Normalization to canonical class key in classes.json
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

    canonical_key = key_mapping.get(raw_item, raw_item.replace(" ", "_"))
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
    display_name = display_names.get(canonical_key, canonical_key.replace("_", " ").title())
    return canonical_key, display_name, condition


def run_freshness_inference(cropped_image_bgr: np.ndarray) -> np.ndarray:
    """
    Executes forward pass through MobileNetV2 model (TFLite or Keras)
    and returns 1D softmax probability array.
    """
    input_tensor = preprocess_for_freshness(cropped_image_bgr)
    model = get_freshness_model()

    if USE_TFLITE:
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(input_details[0]['index'], input_tensor)
        model.invoke()
        output = model.get_tensor(output_details[0]['index'])
        probs = output[0]
    else:
        output = model(input_tensor, training=False)
        probs = output.numpy()[0]

    return np.array(probs, dtype=np.float32)


def evaluate_freshness_for_item(
    cropped_image_bgr: np.ndarray,
    item_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates freshness for a produce crop using honest model outputs.
    If item_key is given (e.g. from detector), evaluates the specific produce pair.
    Otherwise determines most probable produce class and freshness condition.
    """
    probs = run_freshness_inference(cropped_image_bgr)
    class_indices = get_legacy_class_indices()
    class_names = get_legacy_class_names()
    classes_meta = get_classes_metadata().get("classes", {})

    # If item_key is specified and in classes metadata
    if item_key and item_key in classes_meta:
        meta = classes_meta[item_key]
        if not meta.get("freshness_supported", False):
            return {
                "item_key": item_key,
                "item": meta["display_name"],
                "category": meta["category"],
                "freshness": None,
                "freshness_status": "Not Available",
                "freshness_confidence": None,
                "raw_probabilities": {},
                "is_confident": True
            }

        # Check corresponding fresh and spoiled classes
        fresh_label_candidates = [f"fresh{item_key}", f"fresh{item_key}s", f"fresh{item_key.replace('_', '')}"]
        spoiled_label_candidates = [f"spoile{item_key}", f"spoile{item_key}s", f"spoiled{item_key}", f"spoile{item_key.replace('_', '')}"]

        fresh_idx = None
        for cand in fresh_label_candidates:
            if cand in class_indices:
                fresh_idx = class_indices[cand]
                break

        spoiled_idx = None
        for cand in spoiled_label_candidates:
            if cand in class_indices:
                spoiled_idx = class_indices[cand]
                break

        if fresh_idx is not None and spoiled_idx is not None:
            p_fresh = float(probs[fresh_idx])
            p_spoiled = float(probs[spoiled_idx])
            denom = p_fresh + p_spoiled
            if denom > 1e-6:
                norm_fresh = p_fresh / denom
                norm_spoiled = p_spoiled / denom
            else:
                norm_fresh = p_fresh
                norm_spoiled = p_spoiled

            condition = "Fresh" if p_fresh >= p_spoiled else "Spoiled"
            freshness_conf = float(max(norm_fresh, norm_spoiled))

            return {
                "item_key": item_key,
                "item": meta["display_name"],
                "category": meta["category"],
                "freshness": condition,
                "freshness_status": condition,
                "freshness_confidence": round(freshness_conf, 4),
                "is_confident": bool(freshness_conf >= FRESHNESS_CONFIDENCE_THRESHOLD)
            }

    # Unspecified item: find argmax from all 18 classes
    top_indices = probs.argsort()[::-1]
    best_idx = int(top_indices[0])
    best_prob = float(probs[best_idx])
    best_label = class_names[best_idx]

    canonical_key, display_name, condition = parse_class_label(best_label)
    category = classes_meta.get(canonical_key, {}).get("category", "Produce")

    # Borderline stability check
    second_idx = int(top_indices[1])
    second_prob = float(probs[second_idx])
    stability_warning = ""
    if abs(best_prob - second_prob) < 0.12:
        stability_warning = "Borderline freshness result. Consider verifying with better lighting."

    # Top predictions breakdown
    breakdown = []
    for idx in top_indices[:3]:
        lbl = class_names[int(idx)]
        k, name, cond = parse_class_label(lbl)
        breakdown.append({
            "label": lbl,
            "item": name,
            "condition": cond,
            "confidence": round(float(probs[idx]) * 100, 2)
        })

    # Require validated threshold (0.60) to consider identification confident
    is_confident = bool(best_prob >= FOOD_CONFIDENCE_THRESHOLD)

    return {
        "item_key": canonical_key,
        "item": display_name,
        "category": category,
        "freshness": condition,
        "freshness_status": condition,
        "freshness_confidence": round(best_prob, 4),
        "stability_warning": stability_warning,
        "breakdown": breakdown,
        "is_confident": is_confident
    }
