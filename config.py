import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Project paths
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
DATASET_DIR = PROJECT_ROOT / "dataset"

# ----------------------------------------------------------------------------
# Dataset configuration (populated via Roboflow download)
# ----------------------------------------------------------------------------
DATASET_YAML = DATASET_DIR / "data.yaml"
DATASET_TRAIN = DATASET_DIR / "train/images"
DATASET_VAL = DATASET_DIR / "val/images"
DATASET_TEST = DATASET_DIR / "test/images"

# ----------------------------------------------------------------------------
# Roboflow settings (update ROBOFLOW_API_KEY before running download script)
# ----------------------------------------------------------------------------
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "your_roboflow_api_key_here")
ROBOFLOW_WORKSPACE = "lsgi"
ROBOFLOW_PROJECT = "oil-spill-yolov8-complete-dataset"
ROBOFLOW_VERSION = 1
ROBOFLOW_FORMAT = "yolov8"

# ----------------------------------------------------------------------------
# Model configuration
# ----------------------------------------------------------------------------
YOLO_MODEL_PATH = MODEL_DIR / "best.pt"
YOLO_BASE_MODEL = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640
BATCH_SIZE = 16
LEARNING_RATE = 0.001
PATIENCE = 10

# ----------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------
CLASSES = ["oil_spill", "ship", "wake"]

# ----------------------------------------------------------------------------
# Flask app configuration
# ----------------------------------------------------------------------------
DEBUG = False
SECRET_KEY = os.getenv("SECRET_KEY", "change-this-secret-key")
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff"}
UPLOAD_FOLDER = PROJECT_ROOT / "static" / "uploads"
DETECTIONS_FOLDER = PROJECT_ROOT / "static" / "detections"
REPORTS_FOLDER = PROJECT_ROOT / "reports"

# ----------------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "app.log"

# ----------------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------------
def ensure_directories():
    """Ensure required directories exist."""
    for directory in [MODEL_DIR, LOGS_DIR, DATASET_DIR, UPLOAD_FOLDER, DETECTIONS_FOLDER, REPORTS_FOLDER]:
        directory.mkdir(parents=True, exist_ok=True)


def is_allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS