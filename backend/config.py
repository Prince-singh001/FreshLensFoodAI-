import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

# Environment and Server configuration
ENV = os.environ.get("FLASK_ENV", "production" if os.environ.get("RENDER") else "development")
DEBUG = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true") and ENV == "development"
PORT = int(os.environ.get("PORT", "5000"))
HOST = os.environ.get("HOST", "0.0.0.0")
SECRET_KEY = os.environ.get("SECRET_KEY", "foodlens_secure_secret_key_prod_2026")

# File Upload configuration
UPLOAD_FOLDER = os.environ.get(
    "UPLOAD_FOLDER",
    str(ROOT_DIR / "frontend" / "static" / "uploads")
)
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))  # 16MB limit
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Image Validation thresholds
MIN_IMAGE_DIMENSION = 64
MAX_IMAGE_DIMENSION = 4096
BLUR_THRESHOLD = 30.0  # Variance of Laplacian below this is flagged as blurry
DARKNESS_THRESHOLD = 20.0  # Average pixel brightness below this is too dark
BRIGHTNESS_THRESHOLD = 245.0  # Average pixel brightness above this is overexposed

# ML Model & Detection Thresholds
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.45"))
FOOD_CONFIDENCE_THRESHOLD = float(os.environ.get("FOOD_CONFIDENCE_THRESHOLD", "0.60"))
FRESHNESS_CONFIDENCE_THRESHOLD = float(os.environ.get("FRESHNESS_CONFIDENCE_THRESHOLD", "0.60"))
FOOD_VALIDATION_THRESHOLD = float(os.environ.get("FOOD_VALIDATION_THRESHOLD", "0.60"))
IMG_SIZE = 224

# Paths for Models
MODELS_DIR = ROOT_DIR / "models"
PRODUCTION_MODEL_DIR = MODELS_DIR / "production"
LEGACY_BACKEND_MODELS_DIR = BASE_DIR / "models"

# Prefer FoodLens / FreshLens model, fallback to legacy naming for full backwards compatibility
if (PRODUCTION_MODEL_DIR / "foodlens_mobilenetv2_model.h5").exists():
    DEFAULT_KERAS_PATH = str(PRODUCTION_MODEL_DIR / "foodlens_mobilenetv2_model.h5")
elif (PRODUCTION_MODEL_DIR / "freshlens_mobilenetv2_model.h5").exists():
    DEFAULT_KERAS_PATH = str(PRODUCTION_MODEL_DIR / "freshlens_mobilenetv2_model.h5")
elif (PRODUCTION_MODEL_DIR / "safebite_mobilenetv2_model.h5").exists():
    DEFAULT_KERAS_PATH = str(PRODUCTION_MODEL_DIR / "safebite_mobilenetv2_model.h5")
else:
    DEFAULT_KERAS_PATH = str(LEGACY_BACKEND_MODELS_DIR / "safebite_mobilenetv2_model.h5")

if (PRODUCTION_MODEL_DIR / "foodlens_mobilenetv2_model.tflite").exists():
    DEFAULT_TFLITE_PATH = str(PRODUCTION_MODEL_DIR / "foodlens_mobilenetv2_model.tflite")
elif (PRODUCTION_MODEL_DIR / "freshlens_mobilenetv2_model.tflite").exists():
    DEFAULT_TFLITE_PATH = str(PRODUCTION_MODEL_DIR / "freshlens_mobilenetv2_model.tflite")
elif (PRODUCTION_MODEL_DIR / "safebite_mobilenetv2_model.tflite").exists():
    DEFAULT_TFLITE_PATH = str(PRODUCTION_MODEL_DIR / "safebite_mobilenetv2_model.tflite")
else:
    DEFAULT_TFLITE_PATH = str(LEGACY_BACKEND_MODELS_DIR / "safebite_mobilenetv2_model.tflite")

KERAS_MODEL_PATH = os.environ.get("KERAS_MODEL_PATH", DEFAULT_KERAS_PATH)
TFLITE_MODEL_PATH = os.environ.get("TFLITE_MODEL_PATH", DEFAULT_TFLITE_PATH)

# Prefer TFLite on Render or low-memory Linux, full TF Keras on dev Windows if preferred
USE_TFLITE = os.environ.get("USE_TFLITE", "").lower() in ("1", "true") or (os.environ.get("RENDER") is not None)

# Central Data storage
DATA_DIR = BASE_DIR / "data"
CLASSES_CONFIG_PATH = str(DATA_DIR / "classes.json")
HISTORY_FILE = str(DATA_DIR / "history.json")
FEEDBACK_FILE = str(DATA_DIR / "feedback.json")
FEEDBACK_UPLOAD_DIR = str(DATA_DIR / "feedback_uploads")

# Frontend Template and Static folders
TEMPLATE_FOLDER = str(ROOT_DIR / "frontend" / "templates")
STATIC_FOLDER = str(ROOT_DIR / "frontend" / "static")

# Dataset Paths
DATASET_DIR = ROOT_DIR / "dataset"
BACKEND_DATASET_DIR = BASE_DIR / "dataset"

# Reports directory
REPORTS_DIR = ROOT_DIR / "reports"
GRAPHS_DIR = ROOT_DIR / "graphs"

class Config:
    """Namespace class for settings."""
    ENV = ENV
    DEBUG = DEBUG
    PORT = PORT
    HOST = HOST
    SECRET_KEY = SECRET_KEY
    UPLOAD_FOLDER = UPLOAD_FOLDER
    MAX_CONTENT_LENGTH = MAX_CONTENT_LENGTH
    CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD
    KERAS_MODEL_PATH = KERAS_MODEL_PATH
    TFLITE_MODEL_PATH = TFLITE_MODEL_PATH
    USE_TFLITE = USE_TFLITE
    TEMPLATE_FOLDER = TEMPLATE_FOLDER
    STATIC_FOLDER = STATIC_FOLDER

