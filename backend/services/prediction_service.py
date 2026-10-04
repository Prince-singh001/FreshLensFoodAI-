import os
import cv2
import time
from typing import Dict, Any, List
from werkzeug.datastructures import FileStorage

try:
    from config import (
        UPLOAD_FOLDER,
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

    def _log_stage(
        self,
        stage: str,
        started_at: float
    ):
        elapsed = (
            time.perf_counter()
            - started_at
        ) * 1000

        logger.info(
            f"[PREDICT] {stage} completed in {elapsed:.2f} ms"
        )

    def _calculate_iou(
        self,
        box_a: Dict[str, int],
        box_b: Dict[str, int]
    ) -> float:

        x1 = max(
            box_a["x1"],
            box_b["x1"]
        )

        y1 = max(
            box_a["y1"],
            box_b["y1"]
        )

        x2 = min(
            box_a["x2"],
            box_b["x2"]
        )

        y2 = min(
            box_a["y2"],
            box_b["y2"]
        )

        width = max(
            0,
            x2 - x1
        )

        height = max(
            0,
            y2 - y1
        )

        intersection = width * height

        if intersection <= 0:
            return 0.0

        area_a = max(
            0,
            box_a["x2"] - box_a["x1"]
        ) * max(
            0,
            box_a["y2"] - box_a["y1"]
        )

        area_b = max(
            0,
            box_b["x2"] - box_b["x1"]
        ) * max(
            0,
            box_b["y2"] - box_b["y1"]
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union

    def _is_duplicate_object(
        self,
        bbox: Dict[str, int],
        processed_objects: List[Dict[str, Any]]
    ) -> bool:

        for existing in processed_objects:

            existing_bbox = existing.get(
                "bbox"
            )

            if not existing_bbox:
                continue

            overlap = self._calculate_iou(
                bbox,
                existing_bbox
            )

            if overlap >= 0.68:
                return True

        return False

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

            is_valid_file, safe_filename, file_error = (
                validate_uploaded_file(file)
            )

            self._log_stage(
                "File validation",
                stage_started
            )

            if not is_valid_file:

                return {
                    "success": False,
                    "status": "invalid_file",
                    "error": {
                        "code": "INVALID_IMAGE",
                        "message": (
                            file_error
                            or "Invalid image file uploaded."
                        )
                    }
                }

            os.makedirs(
                UPLOAD_FOLDER,
                exist_ok=True
            )

            timestamp_prefix = time.strftime(
                "%Y%m%d%H%M%S_"
            )

            stored_filename = (
                f"{timestamp_prefix}{safe_filename}"
            )

            stored_path = os.path.join(
                UPLOAD_FOLDER,
                stored_filename
            )

            stage_started = time.perf_counter()

            file.save(
                stored_path
            )

            self._log_stage(
                "File save",
                stage_started
            )

            if not os.path.exists(
                stored_path
            ):

                return {
                    "success": False,
                    "status": "invalid_file",
                    "error": {
                        "code": "FILE_SAVE_FAILED",
                        "message": (
                            "Uploaded image could not be saved."
                        )
                    }
                }

            file_size = os.path.getsize(
                stored_path
            )

            logger.info(
                f"[PREDICT] Saved image | "
                f"path={stored_path} | "
                f"size={file_size} bytes"
            )

            stage_started = time.perf_counter()

            (
                is_quality_ok,
                quality_code,
                quality_msg,
                image_bgr,
                metrics
            ) = validate_image_for_inference(
                stored_path
            )

            self._log_stage(
                "Image quality validation",
                stage_started
            )

            if not is_quality_ok:

                return {
                    "success": False,
                    "status": "poor_image_quality",
                    "image_url": (
                        f"/static/uploads/{stored_filename}"
                    ),
                    "error": {
                        "code": (
                            quality_code
                            or "POOR_IMAGE_QUALITY"
                        ),
                        "message": (
                            quality_msg
                            or "Please upload a clearer image."
                        )
                    },
                    "quality_metrics": metrics
                }

            if image_bgr is None:

                return {
                    "success": False,
                    "status": "invalid_file",
                    "image_url": (
                        f"/static/uploads/{stored_filename}"
                    ),
                    "error": {
                        "code": "IMAGE_DECODE_FAILED",
                        "message": (
                            "The uploaded image could not be decoded."
                        )
                    }
                }

            height, width = image_bgr.shape[:2]

            logger.info(
                f"[PREDICT] Image decoded | "
                f"shape={image_bgr.shape}"
            )

            stage_started = time.perf_counter()

            logger.info(
                "[PREDICT] Starting object detection..."
            )

            (
                raw_detections,
                is_rejected_non_food,
                non_food_reason
            ) = detect_objects_in_image(
                image_bgr
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

                return {
                    "success": False,
                    "status": "no_food",
                    "food_detected": False,
                    "image_url": (
                        f"/static/uploads/{stored_filename}"
                    ),
                    "message": (
                        non_food_reason
                        or "No food detected."
                    ),
                    "error": {
                        "code": "NON_FOOD_DETECTED",
                        "message": (
                            non_food_reason
                            or "No food detected."
                        )
                    }
                }

            stage_started = time.perf_counter()

            from ml.validator import validate_food_presence

            (
                is_food_frame,
                food_frame_prob,
                frame_val_details
            ) = validate_food_presence(
                image_bgr,
                threshold=FOOD_VALIDATION_THRESHOLD
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

            if not is_food_frame:

                return {
                    "success": False,
                    "status": "no_food",
                    "food_detected": False,
                    "image_url": (
                        f"/static/uploads/{stored_filename}"
                    ),
                    "message": (
                        "No food detected. Please capture "
                        "a fruit, vegetable, or supported food item."
                    ),
                    "error": {
                        "code": "NO_FOOD_DETECTED",
                        "message": (
                            "No food detected in the image."
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

            processed_objects = []

            total_fruits = 0
            total_vegetables = 0
            total_foods = 0

            logger.info(
                f"[PREDICT] Processing "
                f"{len(raw_detections)} candidate objects..."
            )

            for index, detection in enumerate(
                raw_detections
            ):

                object_started = time.perf_counter()

                bbox = detection.get(
                    "bbox"
                )

                if not bbox:
                    continue

                bbox = {
                    "x1": max(
                        0,
                        min(
                            int(bbox["x1"]),
                            width - 1
                        )
                    ),
                    "y1": max(
                        0,
                        min(
                            int(bbox["y1"]),
                            height - 1
                        )
                    ),
                    "x2": max(
                        1,
                        min(
                            int(bbox["x2"]),
                            width
                        )
                    ),
                    "y2": max(
                        1,
                        min(
                            int(bbox["y2"]),
                            height
                        )
                    )
                }

                if bbox["x2"] <= bbox["x1"]:
                    continue

                if bbox["y2"] <= bbox["y1"]:
                    continue

                if self._is_duplicate_object(
                    bbox,
                    processed_objects
                ):
                    logger.info(
                        f"[PREDICT] Object {index + 1} "
                        f"skipped as duplicate."
                    )
                    continue

                det_label = str(
                    detection.get(
                        "detected_label",
                        ""
                    )
                ).lower()

                det_conf = float(
                    detection.get(
                        "confidence",
                        0.50
                    )
                    or 0.50
                )

                logger.info(
                    f"[PREDICT] Candidate {index + 1} | "
                    f"label={det_label} | "
                    f"confidence={det_conf} | "
                    f"bbox={bbox}"
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
                    continue

                crop_height, crop_width = (
                    cropped.shape[:2]
                )

                if crop_width < 40 or crop_height < 40:

                    logger.info(
                        f"[PREDICT] Object {index + 1} "
                        f"crop too small."
                    )

                    continue

                stage_started = time.perf_counter()

                logger.info(
                    f"[PREDICT] Starting freshness analysis | "
                    f"object={index + 1}"
                )

                freshness_result = (
                    evaluate_freshness_for_item(
                        cropped,
                        item_key=None
                    )
                )

                self._log_stage(
                    f"Object {index + 1} freshness analysis",
                    stage_started
                )

                if not isinstance(
                    freshness_result,
                    dict
                ):
                    continue

                item_key = freshness_result.get(
                    "item_key"
                )

                item_name = freshness_result.get(
                    "item",
                    "Produce Item"
                )

                category = freshness_result.get(
                    "category",
                    "Produce"
                )

                freshness_status = freshness_result.get(
                    "freshness_status",
                    "Not Available"
                )

                freshness_conf = float(
                    freshness_result.get(
                        "freshness_confidence",
                        0.0
                    )
                    or 0.0
                )

                is_confident = bool(
                    freshness_result.get(
                        "is_confident",
                        False
                    )
                )

                logger.info(
                    f"[PREDICT] Object {index + 1} "
                    f"classification | "
                    f"item={item_name} | "
                    f"condition={freshness_status} | "
                    f"confidence={freshness_conf}"
                )

                if not is_confident:

                    logger.info(
                        f"[PREDICT] Object {index + 1} "
                        f"rejected due to low confidence."
                    )

                    continue

                category_lower = str(
                    category
                ).lower()

                if "fruit" in category_lower:

                    total_fruits += 1

                elif "veg" in category_lower:

                    total_vegetables += 1

                else:

                    total_foods += 1

                obj_record = {
                    "id": len(processed_objects) + 1,
                    "item": item_name,
                    "item_key": (
                        item_key
                        or "unknown"
                    ),
                    "category": category,
                    "detection_confidence": round(
                        det_conf,
                        4
                    ),
                    "freshness": freshness_result.get(
                        "freshness"
                    ),
                    "freshness_status": freshness_status,
                    "freshness_confidence": round(
                        freshness_conf,
                        4
                    ),
                    "bbox": bbox,
                    "stability_warning": freshness_result.get(
                        "stability_warning",
                        ""
                    ),
                    "storage_tip": classes_meta.get(
                        item_key or "",
                        {}
                    ).get(
                        "storage_tip",
                        ""
                    ),
                    "safety_guideline": classes_meta.get(
                        item_key or "",
                        {}
                    ).get(
                        "safety_guideline",
                        ""
                    ),
                    "nutrition": classes_meta.get(
                        item_key or "",
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
                    f"time={(time.perf_counter() - object_started) * 1000:.2f} ms"
                )

            logger.info(
                f"[PREDICT] Valid objects after processing: "
                f"{len(processed_objects)}"
            )

            if not processed_objects:

                return {
                    "success": False,
                    "status": "low_confidence",
                    "food_detected": bool(
                        is_food_frame
                    ),
                    "image_url": (
                        f"/static/uploads/{stored_filename}"
                    ),
                    "message": (
                        "Food was detected, but individual "
                        "items could not be identified confidently."
                    ),
                    "error": {
                        "code": "LOW_CONFIDENCE",
                        "message": (
                            "Please use a clearer image with "
                            "better lighting and less overlap."
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
                annotated_filename = stored_filename

            primary_obj = processed_objects[0]

            primary_item = primary_obj[
                "item"
            ]

            primary_condition = primary_obj[
                "freshness_status"
            ]

            primary_conf = (
                primary_obj[
                    "freshness_confidence"
                ]
                or primary_obj[
                    "detection_confidence"
                ]
            )

            primary_conf_pct = round(
                float(primary_conf) * 100,
                2
            )

            category_warning = ""

            user_category = (
                selected_category.capitalize()
            )

            if (
                user_category not in (
                    "All",
                    ""
                )
                and user_category != primary_obj[
                    "category"
                ]
            ):

                category_warning = (
                    f"The image contains "
                    f"{primary_obj['category']} items, "
                    f"but '{selected_category}' was selected."
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
                "category": primary_obj[
                    "category"
                ],
                "image_url": (
                    f"/static/uploads/{stored_filename}"
                ),
                "annotated_image_url": (
                    f"/static/uploads/{annotated_filename}"
                ),
                "inference_time_ms": round(
                    (
                        time.perf_counter()
                        - request_started
                    ) * 1000,
                    2
                ),
                "objects": processed_objects,
                "summary": summary,
                "condition": primary_condition,
                "confidence": primary_conf_pct,
                "label": (
                    f"{primary_condition.lower()}"
                    f"{primary_item.lower().replace(' ', '')}"
                ),
                "message": (
                    "Food items analyzed individually. "
                    "Always verify freshness before consumption."
                ),
                "selected_category": (
                    selected_category.capitalize()
                ),
                "detected_category": primary_obj[
                    "category"
                ],
                "warning": category_warning,
                "stability_warning": primary_obj.get(
                    "stability_warning",
                    ""
                ),
                "top_predictions": [
                    {
                        "item": obj["item"],
                        "condition": obj[
                            "freshness_status"
                        ],
                        "confidence": round(
                            (
                                obj[
                                    "freshness_confidence"
                                ]
                                or obj[
                                    "detection_confidence"
                                ]
                            ) * 100,
                            2
                        )
                    }
                    for obj in processed_objects
                ]
            }

            history_record = {
                "timestamp": time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "food_name": primary_item,
                "condition": primary_condition,
                "confidence": primary_conf_pct,
                "category": primary_obj[
                    "category"
                ],
                "selected_category": (
                    selected_category.capitalize()
                ),
                "detected_category": primary_obj[
                    "category"
                ],
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

            except Exception as history_error:

                logger.exception(
                    f"[PREDICT] History save failed: "
                    f"{history_error}"
                )

            logger.info(
                f"[PREDICT] REQUEST COMPLETED | "
                f"primary={primary_item} | "
                f"condition={primary_condition} | "
                f"objects={len(processed_objects)}"
            )

            return response_data

        except Exception as error:

            elapsed_ms = round(
                (
                    time.perf_counter()
                    - request_started
                ) * 1000,
                2
            )

            logger.exception(
                f"[PREDICT] REQUEST FAILED | "
                f"time={elapsed_ms:.2f} ms | "
                f"error={error}"
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