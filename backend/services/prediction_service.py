import os
import cv2
import time
from typing import Dict, Any, List, Optional
from werkzeug.datastructures import FileStorage

try:
    from config import UPLOAD_FOLDER, CONFIDENCE_THRESHOLD, FOOD_CONFIDENCE_THRESHOLD, FOOD_VALIDATION_THRESHOLD
    from ml.quality import validate_image_for_inference
    from ml.detector import detect_objects_in_image
    from ml.freshness import evaluate_freshness_for_item
    from ml.loader import get_classes_metadata
    from utils.image_utils import (
        validate_uploaded_file,
        crop_bounding_box,
        draw_annotations
    )
    from utils.logger import logger
    from services.history_service import history_service
except ImportError:
    from backend.config import UPLOAD_FOLDER, CONFIDENCE_THRESHOLD, FOOD_CONFIDENCE_THRESHOLD, FOOD_VALIDATION_THRESHOLD
    from backend.ml.quality import validate_image_for_inference
    from backend.ml.detector import detect_objects_in_image
    from backend.ml.freshness import evaluate_freshness_for_item
    from backend.ml.loader import get_classes_metadata
    from backend.utils.image_utils import (
        validate_uploaded_file,
        crop_bounding_box,
        draw_annotations
    )
    from backend.utils.logger import logger
    from backend.services.history_service import history_service

class PredictionService:
    def process_image_upload(
        self,
        file: FileStorage,
        selected_category: str = "All"
    ) -> Dict[str, Any]:
        """
        Main pipeline orchestrating validation, detection, freshness evaluation,
        annotation, and structured response generation.
        """
        start_time = time.perf_counter()

        # Step 1: File format, size, and integrity validation
        is_valid_file, safe_filename, file_error = validate_uploaded_file(file)
        if not is_valid_file:
            return {
                "success": False,
                "status": "invalid_file",
                "error": {
                    "code": "INVALID_IMAGE",
                    "message": file_error or "Invalid image file uploaded."
                }
            }

        # Save uploaded file
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        timestamp_prefix = time.strftime("%Y%m%d%H%M%S_")
        stored_filename = f"{timestamp_prefix}{safe_filename}"
        stored_path = os.path.join(UPLOAD_FOLDER, stored_filename)
        file.save(stored_path)

        # Step 2: Image Quality Analysis (Blur, Darkness, Resolution)
        is_quality_ok, quality_code, quality_msg, image_bgr, metrics = validate_image_for_inference(stored_path)
        if not is_quality_ok:
            return {
                "success": False,
                "status": "poor_image_quality",
                "image_url": f"/static/uploads/{stored_filename}",
                "error": {
                    "code": quality_code or "POOR_IMAGE_QUALITY",
                    "message": quality_msg or "Please upload a clearer and better-lit image."
                },
                "quality_metrics": metrics
            }

        # Step 3: Multi-object Detection & Early Non-Food Rejection
        raw_detections, is_rejected_non_food, non_food_reason = detect_objects_in_image(image_bgr)
        if is_rejected_non_food:
            return {
                "success": False,
                "status": "no_food",
                "food_detected": False,
                "image_url": f"/static/uploads/{stored_filename}",
                "message": non_food_reason or "No food detected. Please capture a fruit, vegetable, or supported food item.",
                "error": {
                    "code": "NON_FOOD_DETECTED",
                    "message": non_food_reason or "No food detected."
                }
            }

        # Step 4: Binary Food vs Non-Food Validation
        # If YOLO did NOT detect explicit food classes, validate whole frame with Food Validator
        has_explicit_coco_food = any(
            d.get("detected_label") in ("banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake")
            for d in raw_detections
        )

        from ml.validator import validate_food_presence
        is_food_frame, food_frame_prob, frame_val_details = validate_food_presence(
            image_bgr,
            threshold=FOOD_VALIDATION_THRESHOLD
        )

        if not has_explicit_coco_food and not is_food_frame:
            return {
                "success": False,
                "status": "no_food",
                "food_detected": False,
                "image_url": f"/static/uploads/{stored_filename}",
                "message": "No food detected. Please point the camera at a fruit, vegetable, or supported food item.",
                "error": {
                    "code": "NO_FOOD_DETECTED",
                    "message": "No food detected in the image. Please point the camera at a fruit, vegetable, or supported food item."
                },
                "validation": frame_val_details
            }

        classes_meta = get_classes_metadata().get("classes", {})
        processed_objects: List[Dict[str, Any]] = []

        total_fruits = 0
        total_vegetables = 0
        total_foods = 0

        # Step 5: Per-object crop, validation & freshness analysis
        for det in raw_detections:
            det_label = det.get("detected_label", "").lower()
            bbox = det.get("bbox", {"x1": 0, "y1": 0, "x2": image_bgr.shape[1], "y2": image_bgr.shape[0]})
            det_conf = det.get("confidence", 0.50)

            cropped = crop_bounding_box(image_bgr, bbox)

            # Map detector label to canonical classes.json key
            matched_key = None
            for key, meta in classes_meta.items():
                aliases = meta.get("detector_aliases", []) + [key]
                if any(alias in det_label for alias in aliases):
                    matched_key = key
                    break

            # If this was an unverified candidate proposal, validate the cropped region
            if matched_key is None:
                is_crop_food, crop_food_prob, _ = validate_food_presence(cropped, threshold=FOOD_VALIDATION_THRESHOLD)
                if not is_crop_food:
                    continue

            # Evaluate freshness on cropped region ONLY if it passed food validation
            freshness_result = evaluate_freshness_for_item(cropped, item_key=matched_key)

            item_name = freshness_result.get("item", det.get("display_name", "Produce Item"))
            category = freshness_result.get("category", "Produce")
            freshness = freshness_result.get("freshness")
            freshness_status = freshness_result.get("freshness_status", "Not Available")
            freshness_conf = freshness_result.get("freshness_confidence", 0.0)
            stability_warn = freshness_result.get("stability_warning", "")
            is_confident = freshness_result.get("is_confident", True)

            # Enforce food confidence threshold (0.60)
            if matched_key is None and (not is_confident or freshness_conf < FOOD_CONFIDENCE_THRESHOLD):
                continue

            # Update category counts
            cat_lower = category.lower()
            if "fruit" in cat_lower:
                total_fruits += 1
            elif "veg" in cat_lower:
                total_vegetables += 1
            else:
                total_foods += 1

            obj_record = {
                "id": det.get("id", len(processed_objects) + 1),
                "item": item_name,
                "item_key": freshness_result.get("item_key", matched_key or "unknown"),
                "category": category,
                "detection_confidence": round(float(det_conf), 4),
                "freshness": freshness,
                "freshness_status": freshness_status,
                "freshness_confidence": freshness_conf,
                "bbox": bbox,
                "stability_warning": stability_warn,
                "storage_tip": classes_meta.get(matched_key or "", {}).get("storage_tip", ""),
                "safety_guideline": classes_meta.get(matched_key or "", {}).get("safety_guideline", ""),
                "nutrition": classes_meta.get(matched_key or "", {}).get("nutrition", {})
            }
            processed_objects.append(obj_record)

        # Check if no valid objects survived confidence filters
        if not processed_objects:
            if is_food_frame:
                return {
                    "success": False,
                    "status": "low_confidence",
                    "food_detected": False,
                    "image_url": f"/static/uploads/{stored_filename}",
                    "message": "The food could not be identified confidently. Please try a clearer image.",
                    "error": {
                        "code": "LOW_CONFIDENCE",
                        "message": "The food could not be identified confidently. Please try a closer image with better lighting."
                    }
                }
            return {
                "success": False,
                "status": "no_food",
                "food_detected": False,
                "image_url": f"/static/uploads/{stored_filename}",
                "message": "No food detected. Please capture a fruit, vegetable, or supported food item.",
                "error": {
                    "code": "NO_FOOD_DETECTED",
                    "message": "No food detected. Please capture a fruit, vegetable, or supported food item."
                }
            }

        # Step 6: Draw annotations on image
        annotated_bgr = draw_annotations(image_bgr, processed_objects)
        annotated_filename = f"annotated_{stored_filename}"
        annotated_path = os.path.join(UPLOAD_FOLDER, annotated_filename)
        cv2.imwrite(annotated_path, annotated_bgr)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # Derive primary object for compatibility with legacy single-item view
        primary_obj = processed_objects[0]
        primary_item = primary_obj["item"]
        primary_condition = primary_obj["freshness_status"]
        primary_conf = primary_obj["freshness_confidence"] or primary_obj["detection_confidence"]
        primary_conf_pct = round(primary_conf * 100, 2)

        # Category mismatch warning if user selected specific category filter
        category_warning = ""
        user_cat = selected_category.capitalize()
        if user_cat not in ("All", "") and user_cat != primary_obj["category"]:
            category_warning = (
                f"The image was recognized as {primary_obj['category']}, "
                f"but you had '{selected_category}' selected."
            )

        summary = {
            "total_objects": len(processed_objects),
            "fruits": total_fruits,
            "vegetables": total_vegetables,
            "food": total_foods
        }

        response_data = {
            "success": True,
            "status": "success",
            "food_detected": True,
            "food_name": primary_item,
            "category": primary_obj["category"],
            "image_url": f"/static/uploads/{stored_filename}",
            "annotated_image_url": f"/static/uploads/{annotated_filename}",
            "inference_time_ms": elapsed_ms,
            "objects": processed_objects,
            "summary": summary,

            # Legacy compatibility fields for existing frontend
            "food_name": primary_item,
            "condition": primary_condition,
            "confidence": primary_conf_pct,
            "label": f"{primary_condition.lower()}{primary_item.lower().replace(' ', '')}",
            "message": (
                "Visually appears fresh based on trained model assessment. Always verify before consumption."
                if primary_condition == "Fresh"
                else (
                    "Shows signs of spoilage or discoloration. Do not consume if spoiled."
                    if primary_condition == "Spoiled"
                    else "Recognized food item. Freshness analysis not available for this category."
                )
            ),
            "selected_category": selected_category.capitalize(),
            "detected_category": primary_obj["category"],
            "warning": category_warning,
            "stability_warning": primary_obj.get("stability_warning", ""),
            "top_predictions": [
                {
                    "item": obj["item"],
                    "condition": obj["freshness_status"],
                    "confidence": round((obj["freshness_confidence"] or obj["detection_confidence"]) * 100, 2)
                }
                for obj in processed_objects
            ]
        }

        # Record scan in history
        history_record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "food_name": primary_item,
            "condition": primary_condition,
            "confidence": primary_conf_pct,
            "category": primary_obj["category"],
            "selected_category": selected_category.capitalize(),
            "detected_category": primary_obj["category"],
            "image_url": f"/static/uploads/{stored_filename}",
            "annotated_image_url": f"/static/uploads/{annotated_filename}",
            "total_objects": len(processed_objects),
            "objects": processed_objects,
            "warning": category_warning,
            "stability_warning": primary_obj.get("stability_warning", "")
        }
        history_service.add_record(history_record)

        return response_data

prediction_service = PredictionService()
