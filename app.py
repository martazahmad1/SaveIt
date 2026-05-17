from flask import Flask, request, jsonify, send_file, render_template
import os
from werkzeug.utils import secure_filename
from image_compressor import (compress_image, decompress_image,
                              lossless_compress, lossless_decompress,
                              smart_decompress, detect_saveit_version)

app = Flask(__name__)
# Use /tmp for uploads to support Vercel's read-only serverless environment
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

# ---- LOSSLESS (Pixel-perfect) ----

@app.route('/lossless-compress', methods=['POST'])
def lossless_compress_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'compressed.saveit')

    try:
        lossless_compress(filepath, output_saveit_path=output_path)
        orig_size = os.path.getsize(filepath)
        comp_size = os.path.getsize(output_path)
        return send_file(output_path, as_attachment=True,
                         download_name='compressed.saveit')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---- AUTOENCODER (Neural, lossy) ----

@app.route('/compress', methods=['POST'])
def compress():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'compressed.saveit')
    
    try:
        # Run compression (inference only using pre-trained PyTorch CNN model)
        compress_image(filepath, output_saveit_path=output_path)
        return send_file(output_path, as_attachment=True, download_name='compressed.saveit')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---- DECOMPRESS (auto-detects v1/v2) ----

@app.route('/decompress', methods=['POST'])
def decompress():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        version = detect_saveit_version(filepath)
        if version == 2:
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'restored.png')
            smart_decompress(filepath, output_path=output_path)
            return send_file(output_path, as_attachment=True, download_name='restored.png')
        else:
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'restored.jpg')
            smart_decompress(filepath, output_path=output_path)
            return send_file(output_path, as_attachment=True, download_name='restored.jpg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
