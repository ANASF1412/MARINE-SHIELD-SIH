# Oil Spill Detection Web Application Overview

## Project Structure
- **app.py / run.py**: Entry points for launching the Flask-based web interface.
- **config.py**: Central configuration including dataset path, API keys, and model settings.
- **download_dataset.py**: Script to fetch the Oil Spill YOLOv8 dataset from Roboflow.
- **validate_setup.py**: Validation utility ensuring dataset splits and YAML configuration are correct.
- **models/**: Stores trained YOLO weights (`best.pt`) and supporting metadata (`dataset.yaml`).
- **src/**: Primary application package.
  - **src/app.py**: Creates the Flask application instance and registers routes.
  - **src/train.py**: YOLOv8 training pipeline using the Roboflow dataset and defined hyperparameters.
  - **src/models/yolo_model.py**: Model loader handling YOLOv8 weights, device selection, and inference preparation.
  - **src/routes/main_routes.py**: Flask routes for uploading images, running detections, and rendering results.
  - **src/utils/**: Helper modules for dataset management, detection utilities, preprocessing, and validation.
    - **dataset_validator.py**: Confirms dataset integrity, YAML content, and model availability.
    - **detector.py**: Orchestrates YOLO inference, post-processing, and detection formatting for the backend.
    - **preprocessing.py**: Image preprocessing helpers used before inference or training.
  - **src/templates/**: Jinja2 templates (`base.html`, `index.html`, `results.html`) for the web interface.
  - **src/static/**: CSS and JavaScript assets powering the UI experience.
- **templates/** & **static/**: Legacy or alternative Flask UI assets used outside the `src/` package.
- **tests/**: Automated unit tests (e.g., `test_detector.py`) covering inference workflows.
- **runs/**: YOLO training outputs and experiment artifacts captured during development.
- **dataset/ (expected)**: Target directory for the downloaded Roboflow dataset with `train/`, `val/`, and `test/` splits.

## Key Workflows
1. **Dataset Acquisition**
   - Configure `ROBOFLOW_API_KEY` in `config.py`.
   - Execute `python download_dataset.py` to populate `dataset/` with the YOLOv8-formatted data.
   - Run `python validate_setup.py` to verify data integrity, YAML configuration, and required splits.

2. **Model Training**
   - Train YOLOv8 via `python src/train.py`, leveraging the dataset paths defined in `config.py`.
   - Store best-performing weights in `models/best.pt` for subsequent inference.

3. **Backend Inference Pipeline**
   - `src/models/yolo_model.py` loads the trained weights and prepares inference.
   - `src/utils/detector.py` exposes functions to process uploaded images and generate detection metadata.
   - Flask routes in `src/routes/main_routes.py` invoke detector utilities and return JSON or rendered templates.

4. **Frontend Experience**
   - Users upload images through the Flask UI (`src/templates/index.html`).
   - Detection results (bounding boxes, masks, metrics) are displayed on `results.html` with static assets from `src/static/`.
   - A PDF report can be generated summarizing detections, images, and statistics for download.

5. **Quality Assurance**
   - `tests/` includes unit tests exercising detector behavior to ensure regression coverage.
   - Logging across modules captures dataset validation, training progress, and inference events for debugging.

This summary should guide further enhancements and troubleshooting for the oil spill detection Flask application.