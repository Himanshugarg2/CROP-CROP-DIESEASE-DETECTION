import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Configuration
UPLOAD_FOLDER = 'static/user_uploaded'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load ML model
try:
    model = load_model("model/v3_pred_cott_dis.h5")
    print('@@ Model loaded successfully')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    """Check if uploaded file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def pred_cot_dieas(cott_plant):
    """Predict cotton plant disease"""
    try:
        # Load and preprocess image
        test_image = load_img(cott_plant, target_size=(150, 150))
        test_image = img_to_array(test_image) / 255
        test_image = np.expand_dims(test_image, axis=0)

        # Predict
        result = model.predict(test_image).round(3)
        pred = np.argmax(result)

        # Define a clear mapping of predictions
        class_labels = {
            0: "Healthy Cotton Plant",
            1: "Diseased Cotton Plant",
            2: "Healthy Cotton Plant"
        }

        prediction = class_labels.get(pred, "Unknown Plant Condition")
        confidence = result[0][pred] * 100

        return prediction, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error in prediction", 0

@app.route("/", methods=['GET'])
def home():
    """Render home page"""
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    # Check if model is loaded
    if model is None:
        return "Machine learning model not loaded", 500

    # Check if file was uploaded
    if 'image' not in request.files:
        return "No file part", 400
    
    file = request.files['image']
    
    # Check if filename is empty
    if file.filename == '':
        return "No selected file", 400
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        return "Invalid file type", 400
    
    try:
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure upload directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the file
        file.save(file_path)
        
        # Predict
        prediction, confidence = pred_cot_dieas(file_path)
        
        # Render result
        return render_template('result.html', 
                               pred_output=prediction, 
                               user_image=filename,
                               confidence=f"{confidence:.2f}")
    
    except Exception as e:
        # Log the error for debugging
        print(f"Prediction error: {e}")
        return "Error during prediction", 500

@app.route('/servalliance', methods=['GET'])
def servalliance():
    """Render servalliance page with camera access."""
    return render_template('servalliance.html')

@app.route("/analyze-frame", methods=['POST'])
def analyze_frame():
    """Analyze video frame from servalliance"""
    try:
        # Extract base64 image data from the request
        data = request.json
        image_data = data['image']
        image_data = image_data.split(",")[-1]  # Remove base64 prefix

        # Decode the image
        import base64
        from PIL import Image
        import io
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Save the image for debugging or logging (optional)
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], "current_frame.png")
        image.save(temp_file_path)

        # Predict
        prediction, confidence = pred_cot_dieas(temp_file_path)

        # Return prediction
        return jsonify({
            "prediction": prediction,
            "confidence": confidence
        })
    except Exception as e:
        print(f"Error analyzing frame: {e}")
        return jsonify({"error": "Error analyzing frame"}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    return "File is too large. Maximum size is 16MB", 413

if __name__ == "__main__":
    app.run(debug=True, threaded=False)
