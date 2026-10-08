import os
import sys
from datetime import datetime
from flask import Flask, request, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import (
    SECRET_KEY,
    DEBUG,
    PORT,
    HOST,
    MAX_CONTENT_LENGTH,
    UPLOAD_FOLDER,
    TEMPLATE_FOLDER,
    STATIC_FOLDER,
    ENV
)

from api import api_bp

from ml.loader import (
    get_freshness_model,
    get_classes_metadata,
    get_legacy_class_indices,
    get_detector_model
)

from services.history_service import history_service
from services.prediction_service import prediction_service
from services.feedback_service import feedback_service
from services.chatbot_service import get_chatbot_response
from utils.logger import logger


app = Flask(
    __name__,
    template_folder=TEMPLATE_FOLDER,
    static_folder=STATIC_FOLDER
)

app.secret_key = SECRET_KEY

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["TEMPLATES_AUTO_RELOAD"] = True

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def initialize_models():

    logger.info("Initializing FreshLens AI models...")

    try:
        get_classes_metadata()

        get_legacy_class_indices()

        get_detector_model()

        get_freshness_model()

        logger.info(
            "FreshLens AI model initialization completed successfully."
        )

        return True

    except Exception as e:

        logger.exception(
            f"FreshLens AI model initialization failed: {e}"
        )

        return False


MODEL_READY = initialize_models()


app.register_blueprint(api_bp)


@app.context_processor
def inject_global_template_vars():

    return {
        "has_history": history_service.count() > 0,
        "current_year": datetime.now().year,
        "env": ENV
    }


@app.route("/")
def home():

    recent_scans = history_service.get_all(
        limit=5
    )

    return render_template(
        "index.html",
        recent_scans=recent_scans
    )


@app.route("/scan")
def scan():

    return render_template(
        "scan.html"
    )


@app.route("/analysis")
def analysis():

    scan_id = request.args.get(
        "id",
        ""
    )

    return render_template(
        "analysis.html",
        scan_id=scan_id
    )


@app.route("/features")
def features():

    return render_template(
        "features.html"
    )


@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


@app.route("/contact")
def contact():

    return render_template(
        "contact.html"
    )


@app.route("/chatbot")
def chatbot_page():

    return render_template(
        "chatbot.html"
    )


@app.route("/history")
def history():

    records = history_service.get_all(
        limit=100
    )

    return render_template(
        "history.html",
        history=records
    )


@app.route("/health")
def health():

    return jsonify({
        "success": True,
        "status": "healthy" if MODEL_READY else "degraded",
        "service": "FreshLens AI",
        "version": "2.0.0",
        "environment": ENV,
        "models_ready": MODEL_READY,
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route("/predict", methods=["POST"])
def legacy_predict():

    if "file" not in request.files:

        return jsonify({
            "success": False,
            "error": "No file uploaded. Please select an image."
        }), 400

    if not MODEL_READY:

        return jsonify({
            "success": False,
            "error": {
                "code": "MODEL_NOT_READY",
                "message": "AI models are not ready. Please try again shortly."
            }
        }), 503

    file = request.files["file"]

    selected_category = request.form.get(
        "selected_category",
        "All"
    ).strip()

    result = prediction_service.process_image_upload(
        file,
        selected_category=selected_category
    )

    status_code = (
        200
        if result.get("success")
        else 422
    )

    return jsonify(
        result
    ), status_code


@app.route("/feedback", methods=["POST"])
def legacy_feedback():

    data = (
        request.get_json(silent=True)
        or request.form.to_dict()
    )

    if not data:

        return jsonify({
            "error": "Request body must be JSON or form data."
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

    if not image_url or not correct_class:

        return jsonify({
            "error": "Missing image_url or label"
        }), 400

    res = feedback_service.submit_feedback(
        image_url=image_url,
        predicted_class=predicted_class,
        correct_class=correct_class,
        user_notes=notes
    )

    return jsonify({
        "success": True,
        "label_saved": correct_class,
        "is_core_class": True,
        "trained_real": False,
        "message": res["message"]
    }), 200


@app.route("/clear-history", methods=["POST"])
def legacy_clear_history():

    success = history_service.clear()

    if success:

        return jsonify({
            "success": True,
            "message": "History cleared successfully"
        })

    return jsonify({
        "error": "Failed to clear history"
    }), 500


@app.route("/chat", methods=["POST"])
def legacy_chat():

    data = request.get_json(
        silent=True
    ) or {}

    message = data.get(
        "message",
        ""
    ).strip()

    if not message:

        return jsonify({
            "error": "Message is required"
        }), 400

    try:

        reply = get_chatbot_response(
            message
        )

        return jsonify({
            "response": reply
        })

    except Exception as e:

        logger.exception(
            f"Error in legacy chat endpoint: {e}"
        )

        return jsonify({
            "error": "Unable to process chat request."
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):

    return jsonify({
        "success": False,
        "error": {
            "code": "FILE_TOO_LARGE",
            "message": "The uploaded image exceeds the 16MB file size limit."
        }
    }), 413


@app.errorhandler(404)
def not_found(error):

    if request.path.startswith("/api/"):

        return jsonify({
            "success": False,
            "error": {
                "code": "NOT_FOUND",
                "message": f"Endpoint '{request.path}' not found."
            }
        }), 404

    return render_template(
        "index.html"
    ), 404


@app.errorhandler(500)
def server_error(error):

    logger.exception(
        f"Internal server error: {error}"
    )

    if request.path.startswith("/api/"):

        return jsonify({
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An internal error occurred. Please try again shortly."
            }
        }), 500

    return render_template(
        "index.html"
    ), 500


if __name__ == "__main__":

    logger.info(
        f"Starting FreshLens AI server on "
        f"http://{HOST}:{PORT} "
        f"(debug={DEBUG})"
    )

    app.run(
        debug=DEBUG,
        host=HOST,
        port=PORT
    )