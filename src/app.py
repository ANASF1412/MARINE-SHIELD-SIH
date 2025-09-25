from flask import Flask, request, jsonify, render_template
import os
import logging
from werkzeug.utils import secure_filename
from config import *
from utils.detector import DetectionManager
from utils.preprocessing import validate_image

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
detector = DetectionManager()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        if not validate_image(file):
            return jsonify({'error': 'Invalid file type'}), 400

        # Save and process image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get detection results
        results = detector.process_image(filepath)

        return render_template('result.html',
                             success=True,
                             detections=results['detections'],
                             report=results['report'],
                             image_path=filename)

    except Exception as e:
        logging.error(f"Error processing upload: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=DEBUG)