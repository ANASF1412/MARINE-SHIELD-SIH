import os
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import json
from PIL import Image # Import Pillow library to get image dimensions
# --- Insert: Import YOLO from ultralytics ---
# Note: You need to install the ultralytics library: pip install ultralytics
# from ultralytics import YOLO # --- MODIFIED: Commented out for frontend-only demo

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# Configuration
# Changed: Simplified upload folder path to be relative to the src directory
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- YOLOv8 Model Integration (Updated) ---
# --- MODIFIED: All model loading and detection logic is commented out or replaced with mock data ---
# def load_yolo_model():
#     """
#     Loads the pretrained YOLOv8 model.
#     Make sure you have a 'best.pt' model file in a 'model' directory
#     at the root of your project (d:/SIH/oil_spill_detection_app/model/best.pt).
#     """
#     model_path = os.path.join(app.root_path, '..', 'model', 'best.pt')
#     if not os.path.exists(model_path):
#         print(f"Warning: Model file not found at {model_path}. Using mock model.")
#         return None
#     print("YOLOv8 model loaded.")
#     model = YOLO(model_path)
#     return model

# def run_detection(image_path, model):
#     """
#     Runs YOLOv8 detection on the provided image.
#     If no model is loaded, returns a mock response.
#     """
#     if model is None:
#         # This is a mock response for when the model file is not found
#         return {
#             "spill_detected": True, "shape": "irregular (mock)", "area_sq_km": 15.7,
#             "density": "high (mock)", "confidence": 0.88, "bounding_box": [120, 50, 400, 300]
#         }

#     # Run prediction with the model
#     results = model(image_path)
    
#     # Process detection results
#     # This assumes your model detects a single 'oil_spill' class
#     for result in results:
#         if len(result.boxes) > 0:
#             # Get the first detected box
#             box = result.boxes[0]
#             coords = box.xyxy[0].tolist() # [x_min, y_min, x_max, y_max]
#             conf = box.conf[0].item()

#             # Placeholder logic for other metrics
#             area = (coords[2] - coords[0]) * (coords[3] - coords[1]) / 100 # Example calculation
            
#             detection_results = {
#                 "spill_detected": True,
#                 "shape": "irregular", # Placeholder
#                 "area_sq_km": round(area, 2), # Placeholder
#                 "density": "high", # Placeholder
#                 "confidence": conf,
#                 "bounding_box": [int(c) for c in coords]
#             }
#             return detection_results

#     # Return if no spill was detected
#     return {"spill_detected": False}


# model = load_yolo_model()

def get_mock_detection_results():
    """
    Returns a mock detection result for frontend demonstration.
    """
    return {
        "spill_detected": True,
        "shape": "irregular (demo)",
        "area_sq_km": 25.4,
        "density": "medium (demo)",
        "confidence": 0.91,
        "bounding_box": [100, 80, 450, 320] # [x_min, y_min, x_max, y_max]
    }
# --- End of Model Integration ---

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

# This route is removed as it's merged into /detect
# @app.route('/upload', methods=['POST'])

# The /detect route now handles both upload and detection logic
@app.route('/detect', methods=['POST'])
def detect():
    """Handle image upload and trigger model inference."""
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get image dimensions for rendering the bounding box correctly
        with Image.open(filepath) as img:
            img_width, img_height = img.size
        
        image_dimensions = {"width": img_width, "height": img_height}

        # --- MODIFIED: Run YOLOv8 detection is replaced with mock results ---
        results = get_mock_detection_results()

        # Prepare data for the results template
        template_data = {
            "image_name": filename,
            "detection_summary": results,
            "image_dimensions": image_dimensions
        }
        
        # Render the results page with detection data
        return render_template('results.html', results=template_data)

    return "File type not allowed", 400


@app.route('/report/<filename>')
def generate_report(filename):
    """Generate and return an oil spill detection summary report."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
        
    # --- MODIFIED: In a real app, you'd re-run detection or fetch stored results ---
    detection_results = get_mock_detection_results()
    
    report_data = {
        "image_name": filename,
        "detection_summary": detection_results
    }
    
    # For simplicity, returning JSON. This could be a PDF.
    return jsonify(report_data)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    # Correct the directory to serve files from
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)