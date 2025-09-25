"""
YOLOv8 training script for oil spill detection.
Run from project root using: python -m src.train
"""

from ultralytics import YOLO
import shutil
import sys
from pathlib import Path
from config import MODEL_DIR, DATASET_BASE

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def train_model():
    """Train YOLOv8 model on oil spill dataset"""
    # Create models directory if not exists
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Train model
    results = model.train(
        data=DATASET_BASE / 'data.yaml',
        epochs=50,
        imgsz=640,
        save=True,
        name='oil_spill_detection'
    )
    
    # Copy best weights to models directory
    best_weights = Path('runs/detect/oil_spill_detection/weights/best.pt')
    if best_weights.exists():
        shutil.copy(best_weights, MODEL_DIR / 'best.pt')
        print(f"[SUCCESS] Model weights saved to {MODEL_DIR / 'best.pt'}")
    else:
        raise FileNotFoundError("Training completed but weights file not found")

if __name__ == "__main__":
    train_model()
