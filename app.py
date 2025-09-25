from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from src.models.yolo_model import OilSpillDetector
from src.utils.dataset_validator import validate_dataset
from config import *

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
detector = OilSpillDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return render_template('result.html', success=False, message="No file uploaded")
    
    file = request.files['image']
    if file and file.filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        detections, result_image = detector.detect(filepath)
        
        return render_template('result.html',
                             success=True,
                             original_image=f"uploads/{filename}",
                             result_image=result_image,
                             detections=detections)

if __name__ == '__main__':
    try:
        validate_dataset()
        print("Dataset validation successful - starting server...")
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs('static/detections', exist_ok=True)
        app.run(debug=DEBUG)
    except Exception as e:
        print(f"Error during startup: {e}")
        sys.exit(1)
