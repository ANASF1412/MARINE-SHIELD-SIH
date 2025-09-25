import cv2
import numpy as np
from config import IMAGE_SIZE

def preprocess_image(image_path):
    """Preprocess image for YOLOv8 inference"""
    try:
        # Read and resize image
        img = cv2.imread(image_path)
        img = cv2.resize(img, IMAGE_SIZE)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        return img
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")

def validate_image(file):
    """Validate uploaded image file"""
    from config import ALLOWED_EXTENSIONS
    
    if file and '.' in file.filename:
        ext = file.filename.rsplit('.', 1)[1].lower()
        if ext in ALLOWED_EXTENSIONS:
            return True
    return False