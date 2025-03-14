from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
from werkzeug.utils import secure_filename
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload directory exists

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ML_MODEL_PATH = os.path.join("models", "heart_disease_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

# Load ML model (Heart Disease)
try:
    ml_model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    logger.info("ML model loaded successfully")
except Exception as e:
    logger.error(f"Error loading ML model: {str(e)}")
    ml_model = None
    scaler = None

# Load NN model (Fashion MNIST)
NN_MODEL_PATH = os.path.join("models", "fashion_mnist_model.h5")

try:
    nn_model = tf.keras.models.load_model('fashion_mnist_model.h5')
    logger.info("NN model loaded successfully")
except Exception as e:
    logger.error(f"Error loading NN model: {str(e)}")
    nn_model = None

# Fashion MNIST class names
fashion_mnist_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Routes for pages
@app.route('/')
def index():
    """Home page with navigation to other pages"""
    return render_template('index.html')

@app.route('/ml_theory')
def ml_theory():
    """Page explaining ML model development and data preparation"""
    return render_template('ml_theory.html')

@app.route('/nn_theory')
def nn_theory():
    """Page explaining Neural Network model development and data preparation"""
    return render_template('nn_theory.html')

@app.route('/ml_demo')
def ml_demo():
    """Demo page for Machine Learning model"""
    return render_template('ml_demo.html')

@app.route('/nn_demo')
def nn_demo():
    """Demo page for Neural Network model"""
    return render_template('nn_demo.html')

# API endpoints for predictions
@app.route('/predict_heart_disease', methods=['POST'])
def predict_heart_disease():
    """Endpoint for heart disease prediction"""
    try:
        if ml_model is None or scaler is None:
            logger.error("Model or scaler not loaded")
            return jsonify({'error': 'Model not loaded'}), 503

        # Get data from request
        data = request.get_json()
        
        # Prepare features (ensure order matches training data)
        features = np.array([[
            float(data['age']),
            float(data['sex']),
            float(data['cp']),
            float(data['trestbps']),
            float(data['chol']),
            float(data['thalach'])
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = ml_model.predict(features_scaled)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(ml_model.predict_proba(features_scaled)[0][1])
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/predict_fashion', methods=['POST'])
def predict_fashion():
    """Endpoint for fashion item classification"""
    try:
        if nn_model is None:
            return jsonify({'error': 'Model not loaded'}), 503

        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
            
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load and preprocess image
            img = tf.keras.preprocessing.image.load_img(
                filepath, color_mode='grayscale', target_size=(28, 28)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0  # Normalize
            img_array = img_array.reshape(1, 28, 28, 1)
            
            # Make prediction
            predictions = nn_model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Get top 3 predictions
            top_3_idx = np.argsort(predictions[0])[-3:][::-1]
            top_3 = [
                {
                    'class_name': fashion_mnist_classes[idx],
                    'confidence': float(predictions[0][idx])
                }
                for idx in top_3_idx
            ]
            
            return jsonify({
                'class_name': fashion_mnist_classes[predicted_class],
                'confidence': confidence,
                'top_3': top_3
            })
            
        finally:
            # Clean up temporary file
            os.remove(filepath)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
