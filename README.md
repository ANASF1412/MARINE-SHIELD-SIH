# Oil Spill Detection App

## Overview
The Oil Spill Detection App is a web-based application designed to detect oil spills using the YOLOv8 model. This application leverages Flask for the backend and provides a user-friendly interface for users to upload images and receive detection results.

## Features
- Upload images for oil spill detection
- Real-time detection using YOLOv8
- Display results with highlighted areas of detection
- Responsive design for various devices

## Project Structure
```
oil_spill_detection_app
├── src
│   ├── static
│   │   ├── css
│   │   │   └── style.css
│   │   └── js
│   │       └── main.js
│   ├── templates
│   │   ├── base.html
│   │   ├── index.html
│   │   └── results.html
│   ├── models
│   │   └── yolo_model.py
│   ├── utils
│   │   ├── detector.py
│   │   └── preprocessing.py
│   ├── routes
│   │   └── main_routes.py
│   └── app.py
├── tests
│   └── test_detector.py
├── requirements.txt
├── config.py
└── README.md
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd oil_spill_detection_app
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Run the Flask application:
   ```
   python src/app.py
   ```
2. Open your web browser and go to `http://127.0.0.1:5000`.

## Validation and Execution

### 1. Validate Dataset
First, validate the dataset structure and files:
```bash
python validate_setup.py
```
This will check:
- Dataset paths existence
- Image and label directories
- File counts and formats
- Output validation results to console and log file

### 2. Run Flask Application
After successful validation, start the Flask app:
```bash
python app.py
```

### Troubleshooting
- If validation fails, check dataset_validation.log for details
- Ensure all paths in config.py are correct
- Verify image/label pairs exist in all splits

## Model Setup

### Training YOLOv8 Model

1. Ensure dataset is properly organized at `F:/SIH/DATASET`
2. Install requirements:
   ```bash
   pip install ultralytics
   ```
3. Train model:
   ```bash
   python src/train.py
   ```
4. Verify weights:
   ```bash
   python validate_setup.py
   ```

The trained model weights will be saved to `models/best.pt`.

### Using Pre-trained Weights

If you have existing weights:
1. Copy your `best.pt` to `models/best.pt`
2. Run validation to verify the weights file

## Training the Model

### Prerequisites
- Python 3.8+
- ultralytics package
- Dataset at F:/SIH/DATASET

### Running Training Script
From the project root directory:
```bash
# Install requirements
pip install -r requirements.txt

# Run training script as module
python -m src.train
```

### Common Issues
- If you get import errors, ensure you're running from project root
- Verify config.py exists in project root
- Check all paths in config.py match your setup

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License.