from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier import logger
import os

app = Flask(__name__)

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager()
        self.model = None
        self.load_model()
    
    def load_model(self):
        try:
            # Load the trained model
            model_path = "artifacts/train_model/updated_model.h5"
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                logger.info("Model loaded successfully")
            else:
                logger.error(f"Model not found at {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def predict(self, image):
        try:
            # Preprocess image
            image = image.resize((224, 224))
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            # Make prediction
            prediction = self.model.predict(image_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = float(np.max(prediction))
            
            # Map class indices to labels
            class_labels = {0: 'Cyst', 1: 'Normal', 2: 'Stone', 3: 'Tumor'}
            predicted_label = class_labels.get(predicted_class, 'Unknown')
            
            return {
                'predicted_class': predicted_label,
                'confidence': confidence,
                'all_predictions': {
                    class_labels[i]: float(prediction[0][i]) 
                    for i in range(len(prediction[0]))
                }
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'error': str(e)}

# Initialize prediction pipeline
predictor = PredictionPipeline()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Open and process image
        image = Image.open(file.stream)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get prediction
        result = predictor.predict(image)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': predictor.model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
