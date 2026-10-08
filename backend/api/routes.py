from datetime import datetime
import time

from flask import request, jsonify

from . import api_bp

try:
    from services.prediction_service import prediction_service
    from services.feedback_service import feedback_service
    from services.history_service import history_service
    from services.chatbot_service import process_agent_chat
    from ml.loader import get_classes_metadata
    from config import ENV
    from utils.logger import logger

except ImportError:
    from backend.services.prediction_service import prediction_service
    from backend.services.feedback_service import feedback_service
    from backend.services.history_service import history_service
    from backend.services.chatbot_service import process_agent_chat
    from backend.ml.loader import get_classes_metadata
    from backend.config import ENV
    from backend.utils.logger import logger


@api_bp.route("/health", methods=["GET"])
def health_check():

    return jsonify({
        "success": True,
        "status": "healthy",
        "service": "FreshLens AI API",
        "version": "2.0.0",
        "environment": ENV,
        "timestamp": datetime.now().isoformat()
    }), 200


@api_bp.route("/classes", methods=["GET"])
def list_classes():

    try:

        meta = get_classes_metadata()

        return jsonify({
            "success": True,
            "total_classes": len(
                meta.get(
                    "classes",
                    {}
                )
            ),
            "data": meta.get(
                "classes",
                {}
            )
        }), 200

    except Exception as e:

        logger.exception(
            f"Classes API error: {e}"
        )

        return jsonify({
            "success": False,
            "error": {
                "code": "CLASSES_ERROR",
                "message": "Failed to load food classes."
            }
        }), 500


@api_bp.route("/predict", methods=["POST"])
def predict():

    request_started = time.perf_counter()

    try:

        logger.info(
            "Prediction request received."
        )

        if "file" not in request.files:

            logger.warning(
                "Prediction request rejected: "
                "file field missing."
            )

            return jsonify({
                "success": False,
                "error": {
                    "code": "MISSING_FILE",
                    "message": (
                        "No file part found in the request. "
                        "Please provide an image file."
                    )
                }
            }), 400

        file = request.files["file"]

        if file is None:

            logger.warning(
                "Prediction request rejected: "
                "empty file object."
            )

            return jsonify({
                "success": False,
                "error": {
                    "code": "EMPTY_FILE",
                    "message": "The uploaded file is empty."
                }
            }), 400

        if not file.filename:

            logger.warning(
                "Prediction request rejected: "
                "filename missing."
            )

            return jsonify({
                "success": False,
                "error": {
                    "code": "EMPTY_FILENAME",
                    "message": (
                        "Please provide a valid image file."
                    )
                }
            }), 400

        selected_category = request.form.get(
            "selected_category",
            "All"
        ).strip()

        logger.info(
            f"Prediction started | "
            f"filename={file.filename} | "
            f"category={selected_category}"
        )

        result = prediction_service.process_image_upload(
            file,
            selected_category=selected_category
        )

        elapsed = (
            time.perf_counter()
            - request_started
        )

        if result is None:

            logger.error(
                "Prediction service returned None | "
                f"elapsed={elapsed:.2f}s"
            )

            return jsonify({
                "success": False,
                "error": {
                    "code": "EMPTY_PREDICTION_RESULT",
                    "message": (
                        "The prediction service "
                        "returned no result."
                    )
                },
                "elapsed_time_ms": round(
                    elapsed * 1000,
                    2
                )
            }), 500

        if not isinstance(result, dict):

            logger.error(
                "Invalid prediction result type: "
                f"{type(result).__name__}"
            )

            return jsonify({
                "success": False,
                "error": {
                    "code": "INVALID_PREDICTION_RESULT",
                    "message": (
                        "The prediction service "
                        "returned an invalid response."
                    )
                },
                "elapsed_time_ms": round(
                    elapsed * 1000,
                    2
                )
            }), 500

        result.setdefault(
            "inference_time_ms",
            round(
                elapsed * 1000,
                2
            )
        )

        logger.info(
            f"Prediction completed | "
            f"elapsed={elapsed:.2f}s | "
            f"success={result.get('success')} | "
            f"status={result.get('status')}"
        )

        is_valid_inference = (
            result.get("success") is True
            or result.get("status") in (
                "no_food",
                "low_confidence",
                "poor_image_quality"
            )
        )

        status_code = (
            200
            if is_valid_inference
            else 422
        )

        return jsonify(
            result
        ), status_code

    except Exception as e:

        elapsed = (
            time.perf_counter()
            - request_started
        )

        logger.exception(
            f"Prediction API error after "
            f"{elapsed:.2f}s: {e}"
        )

        return jsonify({
            "success": False,
            "error": {
                "code": "PREDICTION_ERROR",
                "message": (
                    "Failed to analyze the uploaded image."
                )
            },
            "elapsed_time_ms": round(
                elapsed * 1000,
                2
            )
        }), 500


@api_bp.route("/feedback", methods=["POST"])
def submit_feedback():

    try:

        data = (
            request.get_json(
                silent=True
            )
            or request.form.to_dict()
        )

        if not data:

            return jsonify({
                "success": False,
                "error": {
                    "code": "INVALID_BODY",
                    "message": (
                        "Request body must be valid "
                        "JSON or form data."
                    )
                }
            }), 400

        image_url = data.get(
            "image_url",
            ""
        ).strip()

        label = data.get(
            "label",
            ""
        ).strip()

        predicted_class = data.get(
            "predicted_class",
            label
        ).strip()

        correct_class = data.get(
            "correct_class",
            label
        ).strip()

        notes = data.get(
            "notes",
            ""
        ).strip()

        if not image_url:

            return jsonify({
                "success": False,
                "error": {
                    "code": "MISSING_IMAGE_URL",
                    "message": (
                        "Field 'image_url' is required."
                    )
                }
            }), 400

        if not correct_class:

            return jsonify({
                "success": False,
                "error": {
                    "code": "MISSING_CORRECT_CLASS",
                    "message": (
                        "Field 'correct_class' "
                        "or 'label' is required."
                    )
                }
            }), 400

        res = feedback_service.submit_feedback(
            image_url=image_url,
            predicted_class=predicted_class,
            correct_class=correct_class,
            user_notes=notes
        )

        return jsonify(
            res
        ), 200

    except Exception as e:

        logger.exception(
            f"Feedback API error: {e}"
        )

        return jsonify({
            "success": False,
            "error": {
                "code": "FEEDBACK_ERROR",
                "message": (
                    "Failed to submit feedback."
                )
            }
        }), 500


@api_bp.route("/history", methods=["GET"])
def get_history():

    try:

        limit = min(
            int(
                request.args.get(
                    "limit",
                    50
                )
            ),
            100
        )

        offset = max(
            int(
                request.args.get(
                    "offset",
                    0
                )
            ),
            0
        )

    except ValueError:

        limit = 50
        offset = 0

    try:

        records = history_service.get_all(
            limit=limit,
            offset=offset
        )

        total = history_service.count()

        return jsonify({
            "success": True,
            "total": total,
            "limit": limit,
            "offset": offset,
            "data": records
        }), 200

    except Exception as e:

        logger.exception(
            f"History API error: {e}"
        )

        return jsonify({
            "success": False,
            "error": {
                "code": "HISTORY_ERROR",
                "message": (
                    "Failed to load scan history."
                )
            }
        }), 500


@api_bp.route("/history", methods=["DELETE"])
def delete_history():

    try:

        record_id = request.args.get(
            "id"
        )

        if not record_id:

            body = (
                request.get_json(
                    silent=True
                )
                or {}
            )

            record_id = body.get(
                "id"
            )

        if record_id:

            deleted = (
                history_service.delete_by_id(
                    record_id
                )
            )

            if deleted:

                return jsonify({
                    "success": True,
                    "message": (
                        f"Scan record {record_id} "
                        "deleted successfully."
                    )
                }), 200

            return jsonify({
                "success": False,
                "error": {
                    "code": "NOT_FOUND",
                    "message": (
                        f"Record with ID "
                        f"'{record_id}' not found."
                    )
                }
            }), 404

        history_service.clear()

        return jsonify({
            "success": True,
            "message": (
                "All scan history "
                "cleared successfully."
            )
        }), 200

    except Exception as e:

        logger.exception(
            f"Delete history API error: {e}"
        )

        return jsonify({
            "success": False,
            "error": {
                "code": "HISTORY_DELETE_ERROR",
                "message": (
                    "Failed to delete scan history."
                )
            }
        }), 500


@api_bp.route("/chat", methods=["POST"])
def chat():

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    message = data.get(
        "message",
        ""
    ).strip()

    conversation_id = data.get(
        "conversation_id"
    )

    language = data.get(
        "language",
        "en"
    )

    scan_context = data.get(
        "scan_context"
    )

    if not message:

        return jsonify({
            "success": False,
            "error": {
                "code": "MISSING_MESSAGE",
                "message": (
                    "Field 'message' is required."
                )
            }
        }), 400

    try:

        agent_res = process_agent_chat(
            message=message,
            conversation_id=conversation_id,
            language=language,
            scan_context=scan_context
        )

        return jsonify({
            "success": True,
            "conversation_id": agent_res.get(
                "conversation_id"
            ),
            "language": agent_res.get(
                "language"
            ),
            "response": agent_res.get(
                "answer"
            ),
            "answer": agent_res.get(
                "answer"
            ),
            "tool_used": agent_res.get(
                "tool_used"
            ),
            "sources": agent_res.get(
                "sources",
                []
            )
        }), 200

    except Exception as e:

        logger.exception(
            f"Chatbot execution error: {e}"
        )

        return jsonify({
            "success": False,
            "error": {
                "code": "CHAT_ERROR",
                "message": (
                    "Failed to process chat query."
                )
            }
        }), 500


@api_bp.route(
    "/food/<food_name>",
    methods=["GET"]
)
def get_food_info(food_name):

    try:

        metadata = (
            get_classes_metadata()
            .get(
                "classes",
                {}
            )
        )

        query = (
            food_name
            .lower()
            .strip()
        )

        for key, data in metadata.items():

            aliases = (
                data.get(
                    "detector_aliases",
                    []
                )
                + [
                    key,
                    data.get(
                        "display_name",
                        ""
                    ).lower()
                ]
            )

            if any(
                alias == query
                or alias in query
                or query in alias
                for alias in aliases
                if alias
            ):

                return jsonify({
                    "success": True,
                    "data": {
                        "key": key,
                        "name": data.get(
                            "display_name"
                        ),
                        "display_name": data.get(
                            "display_name"
                        ),
                        "category": data.get(
                            "category"
                        ),
                        "freshness_supported": data.get(
                            "freshness_supported"
                        ),
                        "storage_tip": data.get(
                            "storage_tip"
                        ),
                        "safety_guideline": data.get(
                            "safety_guideline"
                        ),
                        "nutrition": data.get(
                            "nutrition"
                        )
                    }
                }), 200

        return jsonify({
            "success": False,
            "error": {
                "code": "NOT_FOUND",
                "message": (
                    f"Food item '{food_name}' "
                    "is not currently in the catalog."
                )
            }
        }), 404

    except Exception as e:

        logger.exception(
            f"Food information API error: {e}"
        )

        return jsonify({
            "success": False,
            "error": {
                "code": "FOOD_INFO_ERROR",
                "message": (
                    "Failed to load food information."
                )
            }
        }), 500