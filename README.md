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

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License.