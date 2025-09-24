from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from config import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return render_template('result.html', success=False, message="No file uploaded")
    
    file = request.files['image']
    if file.filename == '':
        return render_template('result.html', success=False, message="No file selected")
    
    if file:
        filename = secure_filename(file.filename)
        # Save file logic would go here
        success_message = "Image uploaded successfully! Ready for analysis."
        return render_template('result.html', 
                             success=True,
                             message=success_message,
                             filename=filename)

if __name__ == '__main__':
    app.run(debug=DEBUG, host='0.0.0.0', port=5000)
if __name__ == '__main__':
    app.run(debug=DEBUG, host='0.0.0.0', port=5000)
