# Configuration settings for oil spill detection application

# Path configurations
DATA_DIR = "data"
MODEL_DIR = "models"
OUTPUT_DIR = "output"

# Model parameters
MODEL_NAME = "oil_spill_detector"
CONFIDENCE_THRESHOLD = 0.75

# Processing parameters
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 32

# Debug mode
DEBUG = True

# Static file configurations
STATIC_DIR = "static"
IMAGE_BACKGROUND = "static/image.jpg"