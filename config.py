"""Central settings. Override any of them with environment variables."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-before-deploying")
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "")  # empty = open dashboard (development only)

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(BASE_DIR, "uploads"))
MAX_UPLOAD_MB = 16
ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "model", "maize_model.keras"))
CLASSES_PATH = os.environ.get("CLASSES_PATH", os.path.join(BASE_DIR, "model", "class_names.json"))
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.70"))

# The project is dataset-driven, so the local case log is stored as JSON rather than SQLite.
CASE_LOG_PATH = os.environ.get("CASE_LOG_PATH") or os.environ.get("SQLITE_PATH", os.path.join(BASE_DIR, "maize_cases.json"))
SQLITE_PATH = CASE_LOG_PATH
