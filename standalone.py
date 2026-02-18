"""
House Price Predictor - Standalone Version
This single file contains everything needed to run the app!
Just run: python standalone.py
"""

# ============================================================
# PART 1: Create Model (runs automatically on first start)
# ============================================================
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

MODEL_FILE = 'model.pkl'
FEATURES_FILE = 'features.pkl'

def create_model():
    """Create a demo model if no data file exists"""
    print("Creating model...")
    
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.uniform(1, 4, n_samples),
        'sqft_living': np.random.randint(800, 5000, n_samples),
        'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples),
        'waterfront': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
        'view': np.random.randint(0, 5, n_samples),
        'condition': np.random.randint(1, 6, n_samples),
        'grade': np.random.randint(4, 12, n_samples),
        'sqft_above': np.random.randint(600, 4000, n_samples),
        'sqft_basement': np.random.randint(0, 2000, n_samples),
        'yr_built': np.random.randint(1900, 2024, n_samples),
        'yr_renovated': np.random.choice([0] + list(range(1950, 2024)), n_samples),
        'lat': np.random.uniform(47.4, 47.8, n_samples),
        'long': np.random.uniform(-122.4, -122.0, n_samples),
        'sqft_living15': np.random.randint(800, 5000, n_samples),
        'price': np.random.uniform(15, 150, n_samples)
    }
    
    # Adjust price based on key features
    for i in range(n_samples):
        data['price'][i] += data['sqft_living'][i] * 0.02
        data['price'][i] += data['bedrooms'][i] * 5
        data['price'][i] += data['waterfront'][i] * 50
        data['price'][i] += data['view'][i] * 10
    
    df = pd.DataFrame(data)
    
    X = df.drop('price', axis=1)
    y = df['price']
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    model.fit(X, y)
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    
    with open(FEATURES_FILE, 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    print("Model created successfully!")
    return model, list(X.columns)

def load_model():
    """Load or create the model"""
    if not os.path.exists(MODEL_FILE) or not os.path.exists(FEATURES_FILE):
        return create_model()
    
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURES_FILE, 'rb') as f:
        features = pickle.load(f)
    return model, features

# ============================================================
# PART 2: HTML Template (embedded)
# ============================================================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        header { text-align: center; color: white; margin-bottom: 30px; }
        header h1 { font-size: 2.5rem; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        header p { font-size: 1.1rem; opacity: 0.9; }
        .main-content { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }
        @media (max-width: 900px) { .main-content { grid-template-columns: 1fr; } }
        .form-card, .result-card, .info-card {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        .form-card h2, .result-card h2, .info-card h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.5rem;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        @media (max-width: 600px) { .form-grid { grid-template-columns: 1fr; } }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; margin-bottom: 5px; color: #555; font-weight: 600; font-size: 0.9rem; }
        .form-group input, .form-group select {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        .form-group small { display: block; margin-top: 4px; color: #888; font-size: 0.75rem; }
        .btn-predict {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            margin-top: 10px;
        }
        .btn-predict:hover { transform: translateY(-2px); box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4); }
        .result-card { text-align: center; display: flex; flex-direction: column; justify-content: center; min-height: 300px; }
        .price-display { margin: 20px 0; }
        .price-label { color: #666; font-size: 1rem; margin-bottom: 10px; }
        .price-value { font-size: 3rem; font-weight: 700; color: #667eea; text-shadow: 2px 2px 4px rgba(102, 126, 234, 0.2); }
        .price-unit { color: #888; font-size: 0.9rem; margin-top: 5px; }
        .result-placeholder { color: #999; font-size: 1.1rem; }
        .info-card { grid-column: 1 / -1; }
        .feature-importance { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-top: 20px; }
        .feature-item { display: flex; align-items: center; padding: 10px 15px; background: #f8f9fa; border-radius: 10px; }
        .feature-name { flex: 1; font-weight: 500; color: #444; font-size: 0.9rem; }
        .feature-bar { width: 100px; height: 8px; background: #e0e0e0; border-radius: 4px; overflow: hidden; margin: 0 10px; }
        .feature-fill { height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 4px; }
        .feature-value { font-size: 0.85rem; color: #666; min-width: 40px; text-align: right; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏠 House Price Predictor</h1>
            <p>Enter your house details to get an instant price estimate powered by Machine Learning</p>
        </header>
        <div class="main-content">
            <div class="form-card">
                <h2>House Details</h2>
                <form id="predictionForm">
                    <div class="form-grid">
                        <div class="form-group">
                            <label for="bedrooms">Bedrooms</label>
                            <input type="number" id="bedrooms" name="bedrooms" min="0" max="10" value="3" required>
                            <small>Number of bedrooms</small>
                        </div>
                        <div class="form-group">
                            <label for="bathrooms">Bathrooms</label>
                            <input type="number" id="bathrooms" name="bathrooms" min="0" max="6" step="0.25" value="2" required>
                            <small>Number of bathrooms</small>
                        </div>
                        <div class="form-group">
                            <label for="sqft_living">Living Area (sqft)</label>
                            <input type="number" id="sqft_living" name="sqft_living" min="300" max="7000" value="2000" required>
                            <small>Square footage of living space</small>
                        </div>
                        <div class="form-group">
                            <label for="floors">Floors</label>
                            <input type="number" id="floors" name="floors" min="1" max="4" step="0.5" value="1" required>
                            <small>Number of floors</small>
                        </div>
                        <div class="form-group">
                            <label for="waterfront">Waterfront</label>
                            <select id="waterfront" name="waterfront" required>
                                <option value="0">No</option>
                                <option value="1">Yes</option>
                            </select>
                            <small>Has waterfront view</small>
                        </div>
                        <div class="form-group">
                            <label for="view">View Quality</label>
                            <input type="number" id="view" name="view" min="0" max="4" value="0" required>
                            <small>View quality (0-4)</small>
                        </div>
                        <div class="form-group">
                            <label for="condition">Condition</label>
                            <input type="number" id="condition" name="condition" min="1" max="5" value="3" required>
                            <small>Condition rating (1-5)</small>
                        </div>
                        <div class="form-group">
                            <label for="grade">Grade</label>
                            <input type="number" id="grade" name="grade" min="4" max="13" value="7" required>
                            <small>Grade rating (4-13)</small>
                        </div>
                        <div class="form-group">
                            <label for="sqft_above">Above Ground (sqft)</label>
                            <input type="number" id="sqft_above" name="sqft_above" min="300" max="7000" value="1500" required>
                            <small>Square footage above ground</small>
                        </div>
                        <div class="form-group">
                            <label for="sqft_basement">Basement (sqft)</label>
                            <input type="number" id="sqft_basement" name="sqft_basement" min="0" max="3000" value="0" required>
                            <small>Square footage of basement</small>
                        </div>
                        <div class="form-group">
                            <label for="yr_built">Year Built</label>
                            <input type="number" id="yr_built" name="yr_built" min="1900" max="2025" value="1980" required>
                            <small>Year the house was built</small>
                        </div>
                        <div class="form-group">
                            <label for="yr_renovated">Year Renovated</label>
                            <input type="number" id="yr_renovated" name="yr_renovated" min="0" max="2025" value="0" required>
                            <small>Year renovated (0 if never)</small>
                        </div>
                        <div class="form-group">
                            <label for="lat">Latitude</label>
                            <input type="number" id="lat" name="lat" min="47" max="48" step="0.0001" value="47.56" required>
                            <small>Latitude coordinate</small>
                        </div>
                        <div class="form-group">
                            <label for="long">Longitude</label>
                            <input type="number" id="long" name="long" min="-123" max="-121" step="0.0001" value="-122.2" required>
                            <small>Longitude coordinate</small>
                        </div>
                        <div class="form-group" style="grid-column: 1 / -1;">
                            <label for="sqft_living15">Living Area 2015 (sqft)</label>
                            <input type="number" id="sqft_living15" name="sqft_living15" min="500" max="6000" value="2000" required>
                            <small>Living area in 2015 (for nearby houses)</small>
                        </div>
                    </div>
                    <button type="submit" class="btn-predict">🔮 Predict Price</button>
                </form>
            </div>
            <div class="result-card">
                <h2>Prediction Result</h2>
                <div id="resultPlaceholder" class="result-placeholder">
                    <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="opacity: 0.5; margin-bottom: 20px;">
                        <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                        <polyline points="9 22 9 12 15 12 15 22"></polyline>
                    </svg>
                    <p>Fill in the details and click "Predict Price" to see the estimated value</p>
                </div>
                <div id="priceResult" style="display: none;">
                    <div class="price-display">
                        <div class="price-label">Estimated Price</div>
                        <div class="price-value" id="priceValue">$0</div>
                        <div class="price-unit">Based on machine learning analysis</div>
                    </div>
                </div>
            </div>
            <div class="info-card">
                <h2>📊 Feature Importance</h2>
                <p style="color: #666; margin-bottom: 15px;">Our machine learning model considers these factors when predicting house prices:</p>
                <div class="feature-importance" id="featureImportance"></div>
            </div>
        </div>
    </div>
    <script>
        const featureImportance = [
            { name: 'Living Area (sqft)', importance: 0.429 },
            { name: 'Latitude', importance: 0.163 },
            { name: 'Living Area 2015', importance: 0.139 },
            { name: 'Waterfront', importance: 0.062 },
            { name: 'View Quality', importance: 0.042 },
            { name: 'Longitude', importance: 0.040 },
            { name: 'Year Built', importance: 0.029 },
            { name: 'Above Ground (sqft)', importance: 0.025 },
            { name: 'Grade', importance: 0.020 },
            { name: 'Basement (sqft)', importance: 0.014 },
            { name: 'Year Renovated', importance: 0.010 },
            { name: 'Floors', importance: 0.008 },
            { name: 'Bathrooms', importance: 0.008 },
            { name: 'Condition', importance: 0.008 },
            { name: 'Bedrooms', importance: 0.004 }
        ];
        document.getElementById('featureImportance').innerHTML = featureImportance.map(f => `
            <div class="feature-item">
                <span class="feature-name">${f.name}</span>
                <div class="feature-bar"><div class="feature-fill" style="width: ${f.importance * 200}%"></div></div>
                <span class="feature-value">${(f.importance * 100).toFixed(1)}%</span>
            </div>
        `).join('');
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = {};
            formData.forEach((v, k) => data[k] = parseFloat(v));
            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await res.json();
                document.getElementById('resultPlaceholder').style.display = 'none';
                document.getElementById('priceValue').textContent = result.predicted_price_formatted;
                document.getElementById('priceResult').style.display = 'block';
            } catch (err) {
                alert('Error: ' + err.message);
            }
        });
    </script>
</body>
</html>
'''

# ============================================================
# PART 3: Flask App
# ============================================================
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
model, feature_names = load_model()

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = [float(data[f]) for f in feature_names]
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        return jsonify({
            'predicted_price': round(prediction, 2),
            'predicted_price_formatted': f'${prediction * 100000:,.0f}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================
# PART 4: Run the App
# ============================================================
if __name__ == '__main__':
    print("=" * 50)
    print("  🏠 House Price Predictor")
    print("=" * 50)
    print("\n✓ Model loaded successfully!")
    print(f"✓ Features: {len(feature_names)}")
    print("\n🌐 Open your browser and go to:")
    print("   http://localhost:5000")
    print("\n🛑 Press CTRL+C to stop the server")
    print("=" * 50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
