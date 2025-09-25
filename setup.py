import os
from config import *

def create_project_structure():
    """Initialize project directory structure"""
    directories = [
        DATA_DIR,
        MODEL_DIR,
        OUTPUT_DIR,
        DATASET_DIR,
        'logs',
        'src/uploads',
        'static/images'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    create_project_structure()
    print("Project structure initialized successfully")
