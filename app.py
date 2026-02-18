from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load feature names
features_path = os.path.join(os.path.dirname(__file__), 'features.pkl')
with open(features_path, 'rb') as f:
    feature_names = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features in the correct order
        features = []
        for feature in feature_names:
            if feature in data:
                features.append(float(data[feature]))
            else:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
        
        # Make prediction
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        
        return jsonify({
            'predicted_price': round(prediction, 2),
            'predicted_price_formatted': f'${prediction * 100000:,.0f}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/features', methods=['GET'])
def get_features():
    """Return the list of required features and their descriptions"""
    feature_descriptions = {
        'bedrooms': 'Number of bedrooms',
        'bathrooms': 'Number of bathrooms',
        'sqft_living': 'Square footage of living space',
        'floors': 'Number of floors',
        'waterfront': 'Has waterfront (0=No, 1=Yes)',
        'view': 'View quality (0-4)',
        'condition': 'Condition rating (1-5)',
        'grade': 'Grade rating (4-12)',
        'sqft_above': 'Square footage above ground',
        'sqft_basement': 'Square footage of basement',
        'yr_built': 'Year built',
        'yr_renovated': 'Year renovated (0 if never)',
        'lat': 'Latitude',
        'long': 'Longitude',
        'sqft_living15': 'Living area in 2015'
    }
    
    # Get min/max values from training data for ranges
    ranges = {
        'bedrooms': {'min': 0, 'max': 7, 'step': 1},
        'bathrooms': {'min': 0, 'max': 5, 'step': 0.25},
        'sqft_living': {'min': 300, 'max': 7000, 'step': 10},
        'floors': {'min': 1, 'max': 3.5, 'step': 0.5},
        'waterfront': {'min': 0, 'max': 1, 'step': 1},
        'view': {'min': 0, 'max': 4, 'step': 1},
        'condition': {'min': 1, 'max': 5, 'step': 1},
        'grade': {'min': 4, 'max': 12, 'step': 1},
        'sqft_above': {'min': 300, 'max': 7000, 'step': 10},
        'sqft_basement': {'min': 0, 'max': 2500, 'step': 10},
        'yr_built': {'min': 1900, 'max': 2025, 'step': 1},
        'yr_renovated': {'min': 0, 'max': 2025, 'step': 1},
        'lat': {'min': 47.0, 'max': 48.0, 'step': 0.0001},
        'long': {'min': -123.0, 'max': -121.5, 'step': 0.0001},
        'sqft_living15': {'min': 500, 'max': 5000, 'step': 10}
    }
    
    return jsonify({
        'features': feature_names,
        'descriptions': feature_descriptions,
        'ranges': ranges
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
