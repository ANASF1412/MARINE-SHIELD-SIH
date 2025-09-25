import os
import shutil
import yaml
from config import *

def setup_dataset_structure():
    """Create dataset directory structure"""
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

def create_data_yaml():
    """Create data.yaml for YOLOv8 training"""
    data = {
        'train': TRAIN_DIR,
        'val': VAL_DIR,
        'test': TEST_DIR,
        'nc': len(CLASSES),
        'names': CLASSES
    }
    
    with open(f"{DATASET_DIR}/data.yaml", 'w') as f:
        yaml.dump(data, f)
