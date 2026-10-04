import os
import cv2
import time
from typing import Dict, Any, List
from werkzeug.datastructures import FileStorage

try:
    from config import (
        UPLOAD_FOLDER,
        CONFIDENCE_THRESHOLD,
        FOOD_CONFIDENCE_THRESHOLD,
        FOOD_VALIDATION_THRESHOLD
    )
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
    from backend.config import (
        UPLOAD_FOLDER,
        CONFIDENCE_THRESHOLD,
        FOOD_CONFIDENCE_THRESHOLD,
        FOOD_VALIDATION_THRESHOLD
    )
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

    def _log_stage(self, stage: str, started_at: float):
        elapsed = (time.perf_counter() - started_at) * 1000
        logger.info(
            f"[PREDICT] {stage} completed in {elapsed:.2f} ms"
        )

    def process_image_upload(
        self,
        file: FileStorage,
        selected_category: str = "All"
    ) -> Dict[str, Any]:

        request_started = time.perf_counter()

        logger.info(
            f"[PREDICT] Request started | "
            f"filename={getattr(file, 'filename', 'unknown')} | "
            f"category={selected_category}"
        )

        try:
            stage_started = time.perf_counter()

            is_valid_file, safe_filename, file_error = validate_uploaded_file(file)

            self._log_stage(
                "File validation",
                stage_started
            )

            if not is_valid_file:
                logger.warning(
                    f"[PREDICT] Invalid file: {file_error}"
                )

                return {
                    "success": False,
                    "status": "invalid_file",
                    "error": {
                        "code": "INVALID_IMAGE",
                        "message": file_error or "Invalid image file uploaded."
                    }
                }

            os.makedirs(UPLOAD_FOLDER, exist_ok=True)

            timestamp_prefix = time.strftime("%Y%m%d%H%M%S_")
            stored_filename = f"{timestamp_prefix}{safe_filename}"
            stored_path = os.path.join(
                UPLOAD_FOLDER,
                stored_filename
            )

            stage_started = time.perf_counter()

            file.save(stored_path)

            self._log_stage(
                "File save",
                stage_started
            )

            if not os.path.exists(stored_path):
                logger.error(
                    f"[PREDICT] Saved file does not exist: {stored_path}"
                )

                return {
                    "success": False,
                    "status": "invalid_file",
                    "error": {
                        "code": "FILE_SAVE_FAILED",
                        "message": "Uploaded image could not be saved."
                    }
                }

            file_size = os.path.getsize(stored_path)

            logger.info(
                f"[PREDICT] Saved image | "
                f"path={stored_path} | "
                f"size={file_size} bytes"
            )

            stage_started = time.perf_counter()

            is_quality_ok, quality_code, quality_msg, image_bgr, metrics = (
                validate_image_for_inference(stored_path)
            )

            self._log_stage(
                "Image quality validation",
                stage_started
            )

            if not is_quality_ok:
                logger.warning(
                    f"[PREDICT] Image quality rejected | "
                    f"code={quality_code} | "
                    f"message={quality_msg}"
                )

                return {
                    "success": False,
                    "status": "poor_image_quality",
                    "image_url": f"/static/uploads/{stored_filename}",
                    "error": {
                        "code": quality_code or "POOR_IMAGE_QUALITY",
                        "message": (
                            quality_msg
                            or "Please upload a clearer and better-lit image."
                        )
                    },
                    "quality_metrics": metrics
                }

            if image_bgr is None:
                logger.error(
                    "[PREDICT] Image quality service returned no image."
                )

                return {
                    "success": False,
                    "status": "invalid_file",
                    "image_url": f"/static/uploads/{stored_filename}",
                    "error": {
                        "code": "IMAGE_DECODE_FAILED",
                        "message": "The uploaded image could not be decoded."
                    }
                }

            if len(image_bgr.shape) < 2:
                logger.error(
                    "[PREDICT] Invalid decoded image shape."
                )

                return {
                    "success": False,
                    "status": "invalid_file",
                    "image_url": f"/static/uploads/{stored_filename}",
                    "error": {
                        "code": "INVALID_IMAGE_SHAPE",
                        "message": "The uploaded image has an invalid format."
                    }
                }

            logger.info(
                f"[PREDICT] Image decoded | "
                f"shape={image_bgr.shape}"
            )

            stage_started = time.perf_counter()

            logger.info(
                "[PREDICT] Starting object detection..."
            )

            raw_detections, is_rejected_non_food, non_food_reason = (
                detect_objects_in_image(image_bgr)
            )

            self._log_stage(
                "Object detection",
                stage_started
            )

            logger.info(
                f"[PREDICT] Detection result | "
                f"detections={len(raw_detections)} | "
                f"rejected_non_food={is_rejected_non_food}"
            )

            if is_rejected_non_food:
                logger.info(
                    "[PREDICT] Non-food image rejected by detector."
                )

                return {
                    "success": False,
                    "status": "no_food",
                    "food_detected": False,
                    "image_url": f"/static/uploads/{stored_filename}",
                    "message": (
                        non_food_reason
                        or "No food detected. Please capture a fruit, "
                        "vegetable, or supported food item."
                    ),
                    "error": {
                        "code": "NON_FOOD_DETECTED",
                        "message": (
                            non_food_reason
                            or "No food detected."
                        )
                    }
                }

            explicit_food_labels = {
                "banana",
                "apple",
                "sandwich",
                "orange",
                "broccoli",
                "carrot",
                "hot dog",
                "pizza",
                "donut",
                "cake"
            }

            has_explicit_coco_food = any(
                str(d.get("detected_label", "")).lower()
                in explicit_food_labels
                for d in raw_detections
            )

            logger.info(
                f"[PREDICT] Explicit food detected: "
                f"{has_explicit_coco_food}"
            )

            stage_started = time.perf_counter()

            logger.info(
                "[PREDICT] Starting food validation..."
            )

            from ml.validator import validate_food_presence

            is_food_frame, food_frame_prob, frame_val_details = (
                validate_food_presence(
                    image_bgr,
                    threshold=FOOD_VALIDATION_THRESHOLD
                )
            )

            self._log_stage(
                "Whole-frame food validation",
                stage_started
            )

            logger.info(
                f"[PREDICT] Food validation result | "
                f"is_food={is_food_frame} | "
                f"probability={food_frame_prob}"
            )

            if not has_explicit_coco_food and not is_food_frame:
                logger.info(
                    "[PREDICT] Whole frame rejected as non-food."
                )

                return {
                    "success": False,
                    "status": "no_food",
                    "food_detected": False,
                    "image_url": f"/static/uploads/{stored_filename}",
                    "message": (
                        "No food detected. Please point the camera at "
                        "a fruit, vegetable, or supported food item."
                    ),
                    "error": {
                        "code": "NO_FOOD_DETECTED",
                        "message": (
                            "No food detected in the image. Please point "
                            "the camera at a fruit, vegetable, or supported "
                            "food item."
                        )
                    },
                    "validation": frame_val_details
                }

            stage_started = time.perf_counter()

            classes_meta = get_classes_metadata().get(
                "classes",
                {}
            )

            self._log_stage(
                "Metadata loading",
                stage_started
            )

            processed_objects: List[Dict[str, Any]] = []

            total_fruits = 0
            total_vegetables = 0
            total_foods = 0

            logger.info(
                f"[PREDICT] Processing {len(raw_detections)} detections..."
            )

            for index, det in enumerate(raw_detections):

                object_started = time.perf_counter()

                det_label = str(
                    det.get("detected_label", "")
                ).lower()

                bbox = det.get(
                    "bbox",
                    {
                        "x1": 0,
                        "y1": 0,
                        "x2": image_bgr.shape[1],
                        "y2": image_bgr.shape[0]
                    }
                )

                det_conf = det.get(
                    "confidence",
                    0.50
                )

                logger.info(
                    f"[PREDICT] Object {index + 1} | "
                    f"label={det_label} | "
                    f"confidence={det_conf}"
                )

                stage_started = time.perf_counter()

                cropped = crop_bounding_box(
                    image_bgr,
                    bbox
                )

                self._log_stage(
                    f"Object {index + 1} crop",
                    stage_started
                )

                if cropped is None:
                    logger.warning(
                        f"[PREDICT] Object {index + 1} crop failed."
                    )
                    continue

                matched_key = None

                for key, meta in classes_meta.items():
                    aliases = meta.get(
                        "detector_aliases",
                        []
                    ) + [key]

                    if any(
                        str(alias).lower() in det_label
                        for alias in aliases
                        if alias
                    ):
                        matched_key = key
                        break

                logger.info(
                    f"[PREDICT] Object {index + 1} matched key: "
                    f"{matched_key}"
                )

                if matched_key is None:

                    stage_started = time.perf_counter()

                    is_crop_food, crop_food_prob, crop_details = (
                        validate_food_presence(
                            cropped,
                            threshold=FOOD_VALIDATION_THRESHOLD
                        )
                    )

                    self._log_stage(
                        f"Object {index + 1} crop food validation",
                        stage_started
                    )

                    logger.info(
                        f"[PREDICT] Object {index + 1} crop validation | "
                        f"is_food={is_crop_food} | "
                        f"probability={crop_food_prob}"
                    )

                    if not is_crop_food:
                        logger.info(
                            f"[PREDICT] Object {index + 1} rejected."
                        )
                        continue

                stage_started = time.perf_counter()

                logger.info(
                    f"[PREDICT] Starting freshness analysis | "
                    f"object={index + 1} | "
                    f"item_key={matched_key}"
                )

                freshness_result = evaluate_freshness_for_item(
                    cropped,
                    item_key=matched_key
                )

                self._log_stage(
                    f"Object {index + 1} freshness analysis",
                    stage_started
                )

                if not isinstance(freshness_result, dict):
                    logger.error(
                        f"[PREDICT] Invalid freshness result for "
                        f"object {index + 1}"
                    )
                    continue

                item_name = freshness_result.get(
                    "item",
                    det.get(
                        "display_name",
                        "Produce Item"
                    )
                )

                category = freshness_result.get(
                    "category",
                    "Produce"
                )

                freshness = freshness_result.get(
                    "freshness"
                )

                freshness_status = freshness_result.get(
                    "freshness_status",
                    "Not Available"
                )

                freshness_conf = float(
                    freshness_result.get(
                        "freshness_confidence",
                        0.0
                    ) or 0.0
                )

                stability_warn = freshness_result.get(
                    "stability_warning",
                    ""
                )

                is_confident = freshness_result.get(
                    "is_confident",
                    True
                )

                if (
                    matched_key is None
                    and (
                        not is_confident
                        or freshness_conf < FOOD_CONFIDENCE_THRESHOLD
                    )
                ):
                    logger.info(
                        f"[PREDICT] Object {index + 1} "
                        f"rejected due to low confidence."
                    )
                    continue

                cat_lower = str(
                    category
                ).lower()

                if "fruit" in cat_lower:
                    total_fruits += 1
                elif "veg" in cat_lower:
                    total_vegetables += 1
                else:
                    total_foods += 1

                obj_record = {
                    "id": det.get(
                        "id",
                        len(processed_objects) + 1
                    ),
                    "item": item_name,
                    "item_key": freshness_result.get(
                        "item_key",
                        matched_key or "unknown"
                    ),
                    "category": category,
                    "detection_confidence": round(
                        float(det_conf),
                        4
                    ),
                    "freshness": freshness,
                    "freshness_status": freshness_status,
                    "freshness_confidence": round(
                        freshness_conf,
                        4
                    ),
                    "bbox": bbox,
                    "stability_warning": stability_warn,
                    "storage_tip": classes_meta.get(
                        matched_key or "",
                        {}
                    ).get(
                        "storage_tip",
                        ""
                    ),
                    "safety_guideline": classes_meta.get(
                        matched_key or "",
                        {}
                    ).get(
                        "safety_guideline",
                        ""
                    ),
                    "nutrition": classes_meta.get(
                        matched_key or "",
                        {}
                    ).get(
                        "nutrition",
                        {}
                    )
                }

                processed_objects.append(
                    obj_record
                )

                logger.info(
                    f"[PREDICT] Object {index + 1} completed | "
                    f"item={item_name} | "
                    f"status={freshness_status} | "
                    f"total_time={(time.perf_counter() - object_started) * 1000:.2f} ms"
                )

            logger.info(
                f"[PREDICT] Valid objects after processing: "
                f"{len(processed_objects)}"
            )

            if not processed_objects:

                if is_food_frame:
                    logger.info(
                        "[PREDICT] Food detected but confidence too low."
                    )

                    return {
                        "success": False,
                        "status": "low_confidence",
                        "food_detected": False,
                        "image_url": f"/static/uploads/{stored_filename}",
                        "message": (
                            "The food could not be identified confidently. "
                            "Please try a clearer image."
                        ),
                        "error": {
                            "code": "LOW_CONFIDENCE",
                            "message": (
                                "The food could not be identified confidently. "
                                "Please try a closer image with better lighting."
                            )
                        }
                    }

                logger.info(
                    "[PREDICT] No valid food objects survived validation."
                )

                return {
                    "success": False,
                    "status": "no_food",
                    "food_detected": False,
                    "image_url": f"/static/uploads/{stored_filename}",
                    "message": (
                        "No food detected. Please capture a fruit, "
                        "vegetable, or supported food item."
                    ),
                    "error": {
                        "code": "NO_FOOD_DETECTED",
                        "message": (
                            "No food detected. Please capture a fruit, "
                            "vegetable, or supported food item."
                        )
                    }
                }

            stage_started = time.perf_counter()

            annotated_bgr = draw_annotations(
                image_bgr,
                processed_objects
            )

            self._log_stage(
                "Drawing annotations",
                stage_started
            )

            annotated_filename = (
                f"annotated_{stored_filename}"
            )

            annotated_path = os.path.join(
                UPLOAD_FOLDER,
                annotated_filename
            )

            stage_started = time.perf_counter()

            write_success = cv2.imwrite(
                annotated_path,
                annotated_bgr
            )

            self._log_stage(
                "Saving annotated image",
                stage_started
            )

            if not write_success:
                logger.warning(
                    f"[PREDICT] Could not save annotated image: "
                    f"{annotated_path}"
                )

                annotated_filename = stored_filename

            elapsed_ms = round(
                (
                    time.perf_counter()
                    - request_started
                ) * 1000.0,
                2
            )

            primary_obj = processed_objects[0]

            primary_item = primary_obj["item"]

            primary_condition = primary_obj[
                "freshness_status"
            ]

            primary_conf = (
                primary_obj["freshness_confidence"]
                or primary_obj["detection_confidence"]
            )

            primary_conf_pct = round(
                float(primary_conf) * 100,
                2
            )

            category_warning = ""

            user_cat = selected_category.capitalize()

            if (
                user_cat not in ("All", "")
                and user_cat != primary_obj["category"]
            ):
                category_warning = (
                    f"The image was recognized as "
                    f"{primary_obj['category']}, "
                    f"but you had '{selected_category}' selected."
                )

            summary = {
                "total_objects": len(
                    processed_objects
                ),
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
                "image_url": (
                    f"/static/uploads/{stored_filename}"
                ),
                "annotated_image_url": (
                    f"/static/uploads/{annotated_filename}"
                ),
                "inference_time_ms": elapsed_ms,
                "objects": processed_objects,
                "summary": summary,
                "condition": primary_condition,
                "confidence": primary_conf_pct,
                "label": (
                    f"{primary_condition.lower()}"
                    f"{primary_item.lower().replace(' ', '')}"
                ),
                "message": (
                    "Visually appears fresh based on trained "
                    "model assessment. Always verify before consumption."
                    if primary_condition == "Fresh"
                    else (
                        "Shows signs of spoilage or discoloration. "
                        "Do not consume if spoiled."
                        if primary_condition == "Spoiled"
                        else (
                            "Recognized food item. Freshness analysis "
                            "not available for this category."
                        )
                    )
                ),
                "selected_category": selected_category.capitalize(),
                "detected_category": primary_obj["category"],
                "warning": category_warning,
                "stability_warning": primary_obj.get(
                    "stability_warning",
                    ""
                ),
                "top_predictions": [
                    {
                        "item": obj["item"],
                        "condition": obj["freshness_status"],
                        "confidence": round(
                            (
                                obj["freshness_confidence"]
                                or obj["detection_confidence"]
                            ) * 100,
                            2
                        )
                    }
                    for obj in processed_objects
                ]
            }

            stage_started = time.perf_counter()

            history_record = {
                "timestamp": time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "food_name": primary_item,
                "condition": primary_condition,
                "confidence": primary_conf_pct,
                "category": primary_obj["category"],
                "selected_category": selected_category.capitalize(),
                "detected_category": primary_obj["category"],
                "image_url": (
                    f"/static/uploads/{stored_filename}"
                ),
                "annotated_image_url": (
                    f"/static/uploads/{annotated_filename}"
                ),
                "total_objects": len(
                    processed_objects
                ),
                "objects": processed_objects,
                "warning": category_warning,
                "stability_warning": primary_obj.get(
                    "stability_warning",
                    ""
                )
            }

            try:
                history_service.add_record(
                    history_record
                )

                self._log_stage(
                    "History save",
                    stage_started
                )

            except Exception as history_error:
                logger.exception(
                    f"[PREDICT] History save failed: "
                    f"{history_error}"
                )

            logger.info(
                f"[PREDICT] REQUEST COMPLETED | "
                f"food={primary_item} | "
                f"condition={primary_condition} | "
                f"objects={len(processed_objects)} | "
                f"total_time={elapsed_ms:.2f} ms"
            )

            return response_data

        except Exception as e:
            elapsed_ms = round(
                (
                    time.perf_counter()
                    - request_started
                ) * 1000.0,
                2
            )

            logger.exception(
                f"[PREDICT] REQUEST FAILED | "
                f"time={elapsed_ms:.2f} ms | "
                f"error={e}"
            )

            return {
                "success": False,
                "status": "prediction_error",
                "error": {
                    "code": "PREDICTION_PIPELINE_ERROR",
                    "message": (
                        "An error occurred while analyzing the image."
                    )
                },
                "inference_time_ms": elapsed_ms
            }


prediction_service = PredictionService()