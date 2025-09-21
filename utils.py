# utils.py
import torch
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
import json, os, time, requests
from tqdm import tqdm
from facenet_pytorch import MTCNN
import logging # Use logging for warnings/errors
import random

# Assuming model_module.py contains create_resnet18_classifier
from model_module import create_resnet18_classifier

# --- Constants ---
# Define a default score used when calculation fails or no data exists
# Ensure this is NOT 0 unless 0 is a meaningful neutral score on your scale
DEFAULT_ETHICAL_SCORE = 0 # Example: Midpoint of a 1-5 scale.

# --- Reason Definitions (Keep as they are) ---
DEEPFAKE_REASONS_GENERAL = [{"id": 1,"text": "Provocative"},{"id": 2,"text": "Non Provocative"},{"id": 3,"text": "Not Sure"}]
DEEPFAKE_REASONS_PERSONALITY = [{"id": 1,"text": "Influential"},{"id": 2,"text": "Non Influential"}]
DEEPFAKE_REASONS_EMOTIONS = [{"id": 1,"text": "Weird Face"},{"id": 2,"text": "Angry Face"},{"id": 3,"text": "Happy Face"},{"id": 4,"text": "Sad Face"},{"id": 5,"text": "Surprised Face"},{"id": 6,"text": "Disgusted Face"},{"id": 7,"text": "Neutral Face"},{"id": 8,"text": "Other (unspecified)"}]
DEEPFAKE_REASONS_BROAD = [{"id": 1,"text": "Political manipulation or disinformation"},{"id": 2,"text": "Celebrity impersonation without consent"},{"id": 3,"text": "Fake news or misleading content"},{"id": 4,"text": "Harassment or bullying"},{"id": 5,"text": "Non-consensual intimate content"},{"id": 6,"text": "Identity theft or fraud"},{"id": 7,"text": "Parody or satire"},{"id": 8,"text": "Artistic or creative expression"},{"id": 9,"text": "Educational or demonstration purposes"},{"id": 10,"text": "Other (unspecified)"}]

ALL_CATEGORIES_REASONS = {
    "general": DEEPFAKE_REASONS_GENERAL,
    "personality": DEEPFAKE_REASONS_PERSONALITY,
    "emotions": DEEPFAKE_REASONS_EMOTIONS,
    "broad": DEEPFAKE_REASONS_BROAD
}

# --- Device Setup ---
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Inference Transforms (Keep your original or use standard) ---
transform_deepfake_infer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Face Extraction (Keep original functions) ---
def extract_faces_from_image(image_path, min_face_size=50, device='cpu'):
    # Add try-except block for robustness if desired
    try:
        detector = MTCNN(keep_all=True, device=device, min_face_size=min_face_size)
        image = cv2.imread(image_path)
        if image is None: return []
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces_boxes, _ = detector.detect(rgb_image)
        face_crops = []
        if faces_boxes is not None:
            for box in faces_boxes:
                x1, y1, x2, y2 = map(int, box)
                if x1 < x2 and y1 < y2:
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                    face = image[y1:y2, x1:x2]
                    if face.size > 0: face_crops.append(face)
        return face_crops
    except Exception as e:
        logging.error(f"Error extracting faces from image {image_path}: {e}", exc_info=True)
        return []


def extract_faces_from_video(video_path, min_face_size=50, device='cpu', num_frames=10):
     # Add try-except block for robustness if desired
     face_crops = []
     cap = None
     try:
        detector = MTCNN(keep_all=True, device=device, min_face_size=min_face_size)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1: return []
        frame_indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret: continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces_boxes, _ = detector.detect(rgb_frame)
            if faces_boxes is None: continue
            for box in faces_boxes:
                 x1, y1, x2, y2 = map(int, box)
                 if x1 < x2 and y1 < y2:
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    face = frame[y1:y2, x1:x2]
                    if face.size > 0: face_crops.append(face)
     except Exception as e:
          logging.error(f"Error extracting faces from video {video_path}: {e}", exc_info=True)
          return [] # Return empty list on error
     finally:
          if cap and cap.isOpened():
               cap.release()
     return face_crops

# --- Original Binary Detection Functions (Modified to remove random ethical score) ---
def detect_deepfake_video(model, video_path, transform, device):
    model.eval()
    model.to(device) # Ensure model is on correct device
    faces = extract_faces_from_video(video_path, device=device)
    if not faces: return {"prediction": "unknown", "confidence": 0.0} # Handle no faces

    real_count = 0
    manipulated_count = 0
    confidences = []

    with torch.no_grad():
        for face in faces:
            try: # Add try-except for robustness per face
                image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                image = transform(image).unsqueeze(0).to(device)
                logits = model(image)
                probs = torch.softmax(logits, dim=1)
                predicted = torch.argmax(probs, dim=1).item()
                confidence = probs[0][predicted].item() * 100
                confidences.append(confidence)
                if predicted == 0:
                    real_count += 1
                else:
                    manipulated_count += 1
            except Exception as e:
                 logging.warning(f"Skipping one face during binary video detection due to error: {e}")
                 continue # Skip problematic face

    if real_count == 0 and manipulated_count == 0: # If all faces failed inference
         return {"prediction": "unknown", "confidence": 0.0}

    # Aggregate results (e.g., majority vote, average confidence of majority)
    prediction = 'real' if real_count >= manipulated_count else 'fake'
    # A simple confidence could be the average confidence, or avg conf of the winning class
    avg_confidence = np.mean(confidences) if confidences else 0.0

    # REMOVED random ethical score
    result = {"prediction": prediction, "confidence": avg_confidence}
    return result

def detect_deepfake_image(model, image_path, transform, device):
    model.eval()
    model.to(device) # Ensure model is on correct device
    faces = extract_faces_from_image(image_path, device=device)
    if not faces:
        logging.warning(f"No faces found in image {image_path} for binary detection.")
        return {"prediction": "unknown", "confidence": 0.0}

    # Use first face
    face = faces[0]
    try:
        image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(image)
            probs = torch.softmax(logits, dim=1)
            predicted = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted].item() * 100

        prediction = 'real' if predicted == 0 else 'fake'
        # REMOVED random ethical score
        result = {"prediction": prediction, "confidence": confidence}
    except Exception as e:
         logging.error(f"Error during binary image detection for {image_path}: {e}", exc_info=True)
         result = {"prediction": "error", "confidence": 0.0}

    return result

# --- Feedback Handling (Modified for Env Vars, Kept Original JSONBin Logic) ---

# FIX: Get API Key from Environment Variable
JSONBIN_API_KEY = os.environ.get('JSONBIN_API_KEY')
JSONBIN_BIN_ID = os.environ.get('JSONBIN_BIN_ID')

def get_latest_feedback_json():
    if not JSONBIN_API_KEY or not JSONBIN_BIN_ID:
         logging.error("JSONBin API Key or Bin ID not configured in environment.")
         return None # Return None or {} based on how you handle it later
    url = f'https://api.jsonbin.io/v3/b/{JSONBIN_BIN_ID}/latest'
    headers = {'X-Master-Key': JSONBIN_API_KEY, 'X-Bin-Meta': 'false'}
    try:
        req = requests.get(url, headers=headers, timeout=10)
        req.raise_for_status()
        # It seems jsonbin might return the record directly with /latest and X-Bin-Meta: false
        return req.json()
    except Exception as e:
        logging.error(f"Error fetching feedback from JSONBin: {e}")
        return None

def update_feedback_json(updated_json):
    if not JSONBIN_API_KEY or not JSONBIN_BIN_ID:
         logging.error("JSONBin API Key or Bin ID not configured.")
         return False
    url = f'https://api.jsonbin.io/v3/b/{JSONBIN_BIN_ID}'
    headers = {
        'Content-Type': 'application/json',
        'X-Master-Key': JSONBIN_API_KEY
    }
    try:
        # Wrap the dict in 'record' if your bin structure requires it
        # If /latest returned the record directly, you PUT the record directly too.
        # If unsure, test JSONBin behavior. Assuming direct record PUT here.
        req = requests.put(url, json=updated_json, headers=headers, timeout=15)
        req.raise_for_status()
        logging.info(f"Feedback update response: {req.status_code}")
        return True
    except Exception as e:
        logging.error(f"Error updating feedback on JSONBin: {e}")
        return False

# Keep original DeepfakeFeedback class
class DeepfakeFeedback():
    def __init__(self, detection_id, file_name, is_fake, confidence_score, categories, file_type):
       self.detection_id = detection_id
       self.file_name = file_name
       self.is_fake = is_fake
       self.confidence_score = confidence_score
       self.file_type = file_type
       self.categories = categories

    def get_feedback(self):
       # Prepare data structure expected by original save_feedback
        return {
           "detection_id": self.detection_id,
           "file_name": self.file_name,
           "is_fake": self.is_fake,
           "confidence_score": self.confidence_score,
           "file_type": self.file_type,
           "categories": self.categories
       }

# Keep original generate_unique_filename (as requested)
def generate_unique_filename(feedback_category, file_name):
    if file_name not in feedback_category:
        return file_name
    base, ext = os.path.splitext(file_name)
    # Generate unique suffix - using timestamp + random element for higher uniqueness chance
    unique_suffix = time.strftime('%Y%m%d%H%M%S') + "_" + str(random.randint(100, 999))
    unique_file_name = f"{base}_{unique_suffix}{ext}"
    # Ensure it's truly unique, although collision chance is very low now
    while unique_file_name in feedback_category:
        unique_suffix = time.strftime('%Y%m%d%H%M%S') + "_" + str(random.randint(100, 999))
        unique_file_name = f"{base}_{unique_suffix}{ext}"
    return unique_file_name

# Keep original save_feedback structure (using generate_unique_filename)
def save_feedback(feedback_data):
    """Saves feedback, keeping generate_unique_filename logic."""
    try:
        feedback_json_latest = get_latest_feedback_json() # Fetch latest data
        if feedback_json_latest is None:
             # Handle fetch failure - maybe initialize empty or log error and return
             logging.error("Could not fetch latest feedback, saving aborted.")
             # feedback_dict = {} # Option: Start fresh if fetch fails? Risky.
             return # Fail saving if fetch fails

        # Check if the response has a 'record' key or is the record itself
        if isinstance(feedback_json_latest, dict) and 'record' in feedback_json_latest and isinstance(feedback_json_latest['record'], dict):
             feedback_dict = feedback_json_latest['record']
        elif isinstance(feedback_json_latest, dict): # Assume it's the record directly
             feedback_dict = feedback_json_latest
        else:
             logging.error(f"Unexpected feedback structure received from JSONBin: {type(feedback_json_latest)}. Saving aborted.")
             feedback_dict = {} # Or handle error appropriately
             return


        base_data = {
            "detection_id": feedback_data['detection_id'],
            "is_fake": feedback_data['is_fake'],
            "confidence_score": feedback_data['confidence_score'],
            "file_type": feedback_data['file_type']
        }

        file_name = feedback_data['file_name']
        categories = feedback_data.get('categories', {})

        # Flag to track if any changes were made
        updated = False
        for category, values in categories.items():
             # Ensure category exists in the main dict
            if category not in feedback_dict:
                feedback_dict[category] = {}

            # Prepare data for this category entry
            data = base_data.copy()
            data['reason_id'] = values.get('reason_id')
            data['reason_text'] = values.get('reason_text')
            data['ethical_score'] = values.get('ethical_score')

            # Check if data is valid before saving
            if data['reason_id'] is not None and data['ethical_score'] is not None:
                # Use generate_unique_filename (as requested)
                unique_file_name = generate_unique_filename(feedback_dict[category], file_name)
                feedback_dict[category][unique_file_name] = data
                updated = True # Mark that we made a change
            else:
                 logging.warning(f"Skipping feedback save for category {category} due to missing reason/score.")


        # Update JSONBin only if changes were made
        if updated:
             if not update_feedback_json(feedback_dict): # Pass the updated dict
                  logging.error("Failed to update feedback JSON on remote server.")
        else:
             logging.info("No valid feedback categories provided to save.")

    except Exception as e:
        logging.error(f"Error in save_feedback: {e}", exc_info=True)


# --- Ethical Score Calculation (Minimal Changes) ---

# Load feedback at startup (accepting staleness for minimal change)
# Wrap in try-except for robustness at startup
try:
    FEEDBACK_DICT = get_latest_feedback_json()
    # Handle structure if nested under 'record'
    if isinstance(FEEDBACK_DICT, dict) and 'record' in FEEDBACK_DICT:
         FEEDBACK_DICT = FEEDBACK_DICT['record']
    if not isinstance(FEEDBACK_DICT, dict): # Ensure it's a dict after potential unwrapping
         logging.error("Failed to load feedback dictionary correctly at startup. It's not a dictionary.")
         FEEDBACK_DICT = {} # Default to empty
except Exception as e:
    logging.error(f"Exception loading initial FEEDBACK_DICT: {e}", exc_info=True)
    FEEDBACK_DICT = {} # Default to empty on any error


# FIX: Corrected predict_reason_id (essential change)
def predict_reason_id(model, file_path, transform, device):
    """Predicts 0-indexed reason ID. Includes necessary preprocessing."""
    logging.debug(f"Predicting reason for {os.path.basename(file_path)} using {model.__class__.__name__}")
    model.eval()
    model.to(device)
    is_video = file_path.lower().endswith(('.mp4', '.mov', '.avi'))
    # Use fewer frames for faster ethical scoring if desired
    faces = extract_faces_from_video(file_path, device=device, num_frames=5) if is_video \
            else extract_faces_from_image(file_path, device=device)

    if not faces:
        logging.warning(f"No faces found in {file_path} for reason prediction.")
        return None

    predictions = []
    with torch.no_grad():
        faces_to_process = faces if is_video else [faces[0]]
        for i, face in enumerate(faces_to_process):
            try:
                # --- Preprocessing Steps ---
                image_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                image_tensor = transform(image_pil).unsqueeze(0).to(device)
                # --- End Preprocessing ---

                logging.debug(f"  Face {i}: Input tensor shape: {image_tensor.shape}")
                logits = model(image_tensor)
                predicted = torch.argmax(logits, dim=1).item() # 0-indexed label
                predictions.append(predicted)
                logging.debug(f"  Face {i}: Predicted class index: {predicted}")
            except Exception as e:
                logging.error(f"Error during reason prediction inference on face {i}: {e}", exc_info=True)
                continue

    if not predictions:
        logging.warning(f"Inference failed for all faces in {file_path}.")
        return None

    # Return mode for video, single prediction for image
    final_pred = max(set(predictions), key=predictions.count) if is_video else predictions[0]
    logging.debug(f"Final predicted reason index (0-based): {final_pred}")
    return final_pred


# FIX: Corrected get_avg_score (essential change)
def get_avg_score(category, reason_id_0_based):
    """Gets average score using the global FEEDBACK_DICT (stale data warning)."""
    # FIX: Convert 0-based predicted ID to 1-based ID for matching JSON keys
    reason_id_1_based = reason_id_0_based + 1
    logging.debug(f"Looking up average score for category '{category}', reason_id (1-based) {reason_id_1_based}")

    score_sum = 0
    total_scores = 0

    # Access the global dictionary (accepting staleness)
    category_feedback = FEEDBACK_DICT.get(category, {}) # Get category dict safely
    if not category_feedback:
         logging.warning(f"No feedback data found in global dict for category '{category}'")
         return DEFAULT_ETHICAL_SCORE

    # Iterate through potentially unique filenames due to generate_unique_filename
    for unique_filename, values in category_feedback.items():
         # Check if the reason_id stored (which is 1-based) matches the target 1-based ID
        if isinstance(values, dict) and values.get('reason_id') == reason_id_1_based:
            score = values.get('ethical_score')
            if isinstance(score, (int, float)): # Check if score is valid number
                score_sum += score
                total_scores += 1
            else:
                 logging.warning(f"Invalid ethical score '{score}' found for {unique_filename} in category {category}")


    # FIX: Handle division by zero
    if total_scores > 0:
        avg = score_sum / total_scores
        logging.debug(f"Average score calculated: {avg} (from {total_scores} entries)")
        # Original code rounded, let's keep it for consistency, though float might be better
        return round(avg) # Or return float: avg
    else:
        logging.warning(f"No scores found for category '{category}', reason_id {reason_id_1_based}. Returning default.")
        return DEFAULT_ETHICAL_SCORE


# Global variables to hold loaded models (will be populated by app.py)
ethical_classifier_general = None
ethical_classifier_emotions = None
ethical_classifier_personality = None # Keep placeholder
ethical_classifier_broad = None
# Store models in a dictionary for easier access in get_ethical_score
ETHICAL_MODELS_DICT = {}

# Modified init function called by app.py
def init_ethical_models(model_dir="static/saved_models"):
    """Loads ethical models into global variables and a dictionary."""
    global ethical_classifier_general, ethical_classifier_emotions, ethical_classifier_personality, ethical_classifier_broad, ETHICAL_MODELS_DICT
    logging.info("Initializing ethical models...")

    models_to_load = {
        "general": (len(DEEPFAKE_REASONS_GENERAL), f"{model_dir}/resnet18_general_model.pth"),
        "emotions": (len(DEEPFAKE_REASONS_EMOTIONS), f"{model_dir}/resnet18_emotions_model.pth"),
        "broad": (len(DEEPFAKE_REASONS_BROAD), f"{model_dir}/resnet18_broad_model.pth"),
        "personality": (len(DEEPFAKE_REASONS_PERSONALITY), None)
    }
    device = get_device()

    for category, (num_classes, path) in models_to_load.items():
        model_instance = None # Default to None
        if path and os.path.exists(path):
            try:
                model_instance = create_resnet18_classifier(num_classes)
                model_instance.load_state_dict(torch.load(path, map_location=device))
                model_instance.to(device)
                model_instance.eval()
                logging.info(f"Loaded ethical model for '{category}' from {path}")
            except Exception as e:
                logging.error(f"Failed to load model for '{category}' from {path}: {e}", exc_info=True)
                model_instance = None # Ensure it's None on failure
        else:
             logging.warning(f"Model path not found or not specified for category '{category}': {path}")

        # Assign to global variable (optional, dict is better) and dictionary
        if category == "general": ethical_classifier_general = model_instance
        elif category == "emotions": ethical_classifier_emotions = model_instance
        elif category == "personality": ethical_classifier_personality = model_instance
        elif category == "broad": ethical_classifier_broad = model_instance
        ETHICAL_MODELS_DICT[category] = model_instance # Store in dictionary


# FIX: Corrected get_ethical_score (essential change)
def get_ethical_score(file_path):
    """Calculates the combined ethical score using globally loaded models and FEEDBACK_DICT."""
    global ETHICAL_MODELS_DICT
    logging.info(f"--- Calculating ethical score for: {os.path.basename(file_path)} ---")
    device = get_device()
    transform = transform_deepfake_infer
    weights = { "general": 0.25, "emotions": 0.25, "personality": 0.0, "broad": 0.25 }

    loaded_categories = [cat for cat, model in ETHICAL_MODELS_DICT.items() if model is not None]
    if not loaded_categories:
         logging.error("No ethical models are loaded. Cannot calculate ethical score.")
         return DEFAULT_ETHICAL_SCORE

    # Normalize weights only over loaded categories
    total_weight = sum(weights.get(cat, 0) for cat in loaded_categories)
    if total_weight <= 0: 
         logging.warning("Sum of weights for loaded categories is zero. Using equal weights.")
         num_loaded = len(loaded_categories)
         weights = {cat: 1.0 / num_loaded for cat in loaded_categories}
    else:
         weights = {cat: w / total_weight for cat, w in weights.items() if cat in loaded_categories}


    final_score = 0.0
    for category in loaded_categories:
        model = ETHICAL_MODELS_DICT[category]
        logging.debug(f"Processing category: {category}")

        predicted_label = predict_reason_id(model, file_path, transform, device)

        if predicted_label is not None:
            avg_score = get_avg_score(category, predicted_label)
            logging.debug(f"  Category '{category}': Predicted index={predicted_label}, Avg Score={avg_score}")
            score_contribution = weights.get(category, 0) * avg_score
            final_score += score_contribution
            logging.debug(f"  Score contribution: {score_contribution:.3f} (Weight: {weights.get(category, 0):.2f})")
        else:
            logging.warning(f"  Category '{category}': Prediction failed. Using default score {DEFAULT_ETHICAL_SCORE} with weight.")
            score_contribution = weights.get(category, 0) * DEFAULT_ETHICAL_SCORE
            final_score += score_contribution
            logging.debug(f"  Score contribution (default): {score_contribution:.3f} (Weight: {weights.get(category, 0):.2f})")


    logging.info(f"Final calculated score for {os.path.basename(file_path)}: {final_score}")
    return 10 - round(final_score, 3)