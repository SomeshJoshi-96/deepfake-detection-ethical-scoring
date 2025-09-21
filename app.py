# app.py
import os
import logging
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import time
import torch
from dotenv import load_dotenv
import random

from model_module import VITClassifier, model_name_or_path 
import utils 

# --- Configuration ---
load_dotenv() # Load .env file first
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "a_default_secret_key_for_dev")

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi'}

# --- Global Variables & Model Loading (ONCE at startup) ---
DEVICE = utils.get_device()
JSONBIN_API_KEY = os.environ.get('JSONBIN_API_KEY')
JSONBIN_BIN_ID = os.environ.get('JSONBIN_BIN_ID')

if not JSONBIN_API_KEY or not JSONBIN_BIN_ID:
    logging.warning("JSONBin API Key or Bin ID not found in environment. Feedback saving will fail.")

# Load Binary Detection Model (ViT) - Outside request context
MODEL_VIT = None
try:
    vit_model_path = "static/saved_models/vit_deep_fake_model_v5.pth" 
    if os.path.exists(vit_model_path):
        MODEL_VIT = VITClassifier(model_name_or_path, 2) # Ensure VITClassifier is correctly defined/imported
        MODEL_VIT.load_state_dict(torch.load(vit_model_path, map_location=DEVICE))
        MODEL_VIT.to(DEVICE)
        MODEL_VIT.eval()
        logging.info(f"ViT model loaded successfully from {vit_model_path} on {DEVICE}.")
    else:
        logging.error(f"ViT model file not found at {vit_model_path}. Binary detection will fail.")
except Exception as e:
    logging.error(f"Error loading ViT model: {e}", exc_info=True)
    MODEL_VIT = None # Ensure it's None if loading fails

# Load Ethical Category Models using the util function - Outside request context
utils.init_ethical_models() # This populates globals and ETHICAL_MODELS_DICT in utils
# Access loaded models via utils.ETHICAL_MODELS_DICT if needed directly in app

# --- Helper Functions ---
def allowed_file(filename, file_type):
    allowed_extensions = ALLOWED_IMAGE_EXTENSIONS if file_type == 'image' else ALLOWED_VIDEO_EXTENSIONS
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

# Combined detection logic helper (Minimal change from previous fix)
def run_detection(file):
    is_image = allowed_file(file.filename, 'image')
    is_video = allowed_file(file.filename, 'video')
    if not is_image and not is_video: return jsonify({'error': 'Invalid file type.'}), 400
    file_type = 'image' if is_image else 'video'

    filename = secure_filename(file.filename)
    # Use a temporary unique name for saving to avoid conflicts during processing
    temp_filename = f"{int(time.time())}_{random.randint(1000,9999)}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)

    try:
        file.save(file_path)
        logging.info(f"File saved temporarily to {file_path}")

        # 1. Binary Detection (Real/Fake) using global model
        binary_result = {"prediction": "unknown", "confidence": 0.0}
        if MODEL_VIT:
            # Use the appropriate utils function (detect_deepfake_image or video)
            if is_image:
                 binary_result = utils.detect_deepfake_image(MODEL_VIT, file_path, utils.transform_deepfake_infer, DEVICE)
            else:
                 binary_result = utils.detect_deepfake_video(MODEL_VIT, file_path, utils.transform_deepfake_infer, DEVICE)
        else:
            logging.warning("ViT model not loaded, skipping binary detection.")

        # 2. Ethical Score Calculation using global models (defined in utils)
        ethical_score = utils.DEFAULT_ETHICAL_SCORE # Default value
        try:
            # Call the main ethical scoring function from utils
            ethical_score = utils.get_ethical_score(file_path)
        except Exception as e:
            logging.error(f"Error calculating ethical score for {filename}: {e}", exc_info=True)
            # Keep default score on error


        # 3. Prepare Response
        detection_id = str(int(time.time() * 1000)) # Use timestamp as ID
        response = {
            'result': binary_result['prediction'],
            'confidence': binary_result['confidence'],
            'ethical_score': ethical_score, 
            'detection_id': detection_id,
            'file_name': filename, 
            'file_type': file_type
        }
        return jsonify(response), 200

    except Exception as e:
        logging.error(f"Error during detection process for {filename}: {e}", exc_info=True)
        return jsonify({'error': 'An error occurred during detection.'}), 500
    finally:
        # Cleanup - remove the temporary file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"Removed temporary file: {file_path}")
            except Exception as e:
                logging.error(f"Error removing file {file_path}: {e}")


@app.route('/detect/image', methods=['POST'])
def detect_image_route():
    if 'file' not in request.files: return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename, 'image'): return jsonify({'error': 'File type not allowed'}), 400
    return run_detection(file)


@app.route('/detect/video', methods=['POST'])
def detect_video_route():
    if 'file' not in request.files: return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename, 'video'): return jsonify({'error': 'File type not allowed'}), 400
    return run_detection(file)


@app.route('/api/deepfake-reasons', methods=['GET'])
def get_deepfake_reasons():
    return jsonify(utils.ALL_CATEGORIES_REASONS)


@app.route('/api/submit-feedback', methods=['POST'])
def submit_deepfake_feedback():
    """API endpoint to submit user feedback"""
    data = request.json
    if not data: return jsonify({'error': 'No JSON data received'}), 400

    required_fields = ['detection_id', 'file_name', 'file_type', 'confidence_score', 'is_fake', 'categories']
    missing = [f for f in required_fields if f not in data]
    if missing: return jsonify({'error': f'Missing fields: {missing}'}), 400
    if not isinstance(data['categories'], dict) or not data['categories']: return jsonify({'error': '"categories" must be a non-empty dictionary'}), 400

    # Basic validation of categories structure
    for cat, values in data['categories'].items():
         if not isinstance(values, dict) or 'reason_id' not in values or 'ethical_score' not in values:
              return jsonify({'error': f'Invalid structure for category "{cat}".'}), 400

    feedback_obj = utils.DeepfakeFeedback(
        is_fake=data.get('is_fake', True),
        detection_id=data['detection_id'],
        file_name=data['file_name'],
        confidence_score=data['confidence_score'],
        file_type=data['file_type'],
        categories=data['categories']
    )

    # Use the original save feedback logic from utils
    # It will use the API Key loaded from environment within utils.py
    utils.save_feedback(feedback_obj.get_feedback())

    return jsonify({'message': 'Feedback submitted successfully'}), 200

if __name__ == "__main__":
    load_dotenv()

    if not os.environ.get('JSONBIN_API_KEY') or not os.environ.get('JSONBIN_BIN_ID'):
         logging.warning("### JSONBIN API Key or Bin ID not found in environment variables! Feedback saving/loading WILL FAIL. ###")

    app.run(host="0.0.0.0", port=5000, debug=True)