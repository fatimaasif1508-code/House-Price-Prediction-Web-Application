# 🪟 Windows Setup Guide for House Price Predictor

## Quick Start (3 Steps)

### Step 1: Open Command Prompt
Press `Win + R`, type `cmd`, press **Enter**

### Step 2: Navigate to Your Folder
```cmd
cd "C:\Users\MAHAD ENTERPRISES"
```

### Step 3: Run These Commands
```cmd
mkdir house_price_app
cd house_price_app
```

---

## 📁 Create Project Files

### 1. Create `app.py`
Right-click in folder → New → Text Document → Name it `app.py`
Paste this content:

```python
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
        
        features = []
        for feature in feature_names:
            if feature in data:
                features.append(float(data[feature]))
            else:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
        
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        
        return jsonify({
            'predicted_price': round(prediction, 2),
            'predicted_price_formatted': f'${prediction * 100000:,.0f}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### 2. Create `requirements.txt`
```
Flask==3.0.0
scikit-learn==1.3.2
numpy==1.26.2
pandas==2.1.4
```

### 3. Create `setup.py`
(See the setup.py file in this folder)

### 4. Create `templates` folder
```cmd
mkdir templates
```

### 5. Create `templates/index.html`
(See the index.html file in the templates folder)

---

## 🚀 Run the Application

### Option A: Using setup.py (Recommended)

```cmd
cd "C:\Users\MAHAD ENTERPRISES\house_price_app"
python setup.py
python app.py
```

### Option B: Direct (if you have houses.csv)

```cmd
cd "C:\Users\MAHAD ENTERPRISES\house_price_app"
python -c "import pickle; pickle.dump(['bedrooms','bathrooms','sqft_living','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','lat','long','sqft_living15'], open('features.pkl','wb'))"
python app.py
```

---

## 🌐 Access the Website

Once running, open your browser and go to:
```
http://localhost:5000
```

---

## 🛑 Stop the Server

Press `CTRL + C` in the Command Prompt window.

---

## 🔄 Restart After Closing

```cmd
cd "C:\Users\MAHAD ENTERPRISES\house_price_app"
python app.py
```

---

## ❌ Common Errors & Fixes

### Error: "python is not recognized"
**Fix:** Install Python from https://python.org and check "Add Python to PATH"

### Error: "No module named 'flask'"
**Fix:** Run: `pip install flask scikit-learn numpy pandas`

### Error: "model.pkl not found"
**Fix:** Run `python setup.py` first to create the model files

---

## 📞 Need Help?

1. Make sure Python is installed: `python --version`
2. Make sure pip is working: `pip --version`
3. Check you're in the right folder: `cd "C:\Users\MAHAD ENTERPRISES\house_price_app"`
