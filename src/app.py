from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
import os
import logging
from werkzeug.utils import secure_filename
from config import *
from src.utils.detector import DetectionManager

# Configure logging
logging.basicConfig(
	filename=LOG_FILE,
	level=LOG_LEVEL,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
detector = DetectionManager()

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload():
	try:
		if 'image' not in request.files:
			return jsonify({'error': 'No image uploaded'}), 400

		file = request.files['image']
		if file.filename == '':
			return jsonify({'error': 'Empty filename'}), 400

		# Save and process image
		filename = secure_filename(file.filename)
		os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
		filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		file.save(filepath)

		# Get segmentation results
		results = detector.process_image(filepath)

		# Build optional PDF now or via separate route
		return render_template(
			'results.html',
			results={
				'image_name': results.image_name,
				'detection_summary': results.summary,
				'overlay_path': url_for('static', filename=f"detections/overlay_{results.image_name}"),
			}
		)

	except Exception as e:
		logging.error(f"Error processing upload: {e}")
		return jsonify({'error': str(e)}), 500

@app.route('/report/<path:filename>')
def generate_report(filename):
	try:
		# Build report from last results
		from pathlib import Path
		image_name = filename
		# Recreate DetectionResult minimal to generate report using existing overlay
		class _DR:
			pass
		dr = _DR()
		dr.image_name = image_name
		dr.annotated_image_path = Path(DETECTIONS_FOLDER) / f"overlay_{image_name}"
		dr.summary = {}
		pdf_path = detector.build_pdf_report(dr)
		return send_from_directory(pdf_path.parent, pdf_path.name, as_attachment=True)
	except Exception as e:
		logging.error(f"Error generating report: {e}")
		return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
	os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
	os.makedirs(DETECTIONS_FOLDER, exist_ok=True)
	app.run(debug=DEBUG)