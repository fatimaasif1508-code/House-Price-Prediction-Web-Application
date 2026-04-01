# House Price Prediction Website

A machine learning-powered web application that predicts house prices based on various features like bedrooms, bathrooms, square footage, location, and more.



Live Demo: 
file:///C:/Users/Giga%20Tech/OneDrive/Desktop/Entreprenaurship/house_price_app/index.html

## 📁 Project Structure

```
house_price_app/
├── index.html              # Main website (static version with JS ML model)
├── app.py                  # Flask backend (for API version)
├── model.pkl               # Trained Random Forest model
├── features.pkl            # Feature names list
├── static_model.json       # Linear regression model for JS
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Flask template
└── README.md               # This file
```

## 🚀 Features

- **Interactive Form**: Enter house details with validation
- **Real-time Prediction**: Get instant price estimates
- **Feature Importance Visualization**: See which factors affect price most
- **Responsive Design**: Works on desktop and mobile devices
- **Machine Learning Model**: Trained on 999 real estate records

## 🛠️ Technologies Used

### Static Website (Deployed)
- HTML5, CSS3, JavaScript (Vanilla)
- Linear Regression Model (embedded in JavaScript)
- R² Score: 0.66

### Flask Backend (Optional)
- Python 3.12+
- Flask 3.0
- scikit-learn (Random Forest)
- R² Score: 0.80

## 📊 Model Performance

| Model | R² Score | RMSE |
|-------|----------|------|
| Random Forest (Flask) | 0.80 | 13.27 |
| Linear Regression (JS) | 0.66 | ~18.5 |

## 🏃 Running Locally

### Option 1: Static Website (Simple)
Just open `index.html` in any modern web browser. No server required!

### Option 2: Flask Backend (More Accurate)

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask app:
```bash
python app.py
```

3. Open http://localhost:5000 in your browser

## 📋 Input Features

| Feature | Description | Range |
|---------|-------------|-------|
| bedrooms | Number of bedrooms | 0-10 |
| bathrooms | Number of bathrooms | 0-6 |
| sqft_living | Living area square footage | 300-7000 |
| floors | Number of floors | 1-3.5 |
| waterfront | Has waterfront view | 0-1 |
| view | View quality rating | 0-4 |
| condition | Condition rating | 1-5 |
| grade | Grade rating | 4-13 |
| sqft_above | Above ground square footage | 300-7000 |
| sqft_basement | Basement square footage | 0-3000 |
| yr_built | Year built | 1900-2025 |
| yr_renovated | Year renovated (0=never) | 0-2025 |
| lat | Latitude | 47.0-48.0 |
| long | Longitude | -123.0--121.5 |
| sqft_living15 | Living area in 2015 | 500-6000 |

## 🔍 Feature Importance

The most important factors affecting house prices:

1. **Living Area (42.9%)** - Square footage of living space
2. **Latitude (16.3%)** - Location north/south
3. **Living Area 2015 (13.9%)** - Neighbor's living area
4. **Waterfront (6.2%)** - Waterfront property premium
5. **View Quality (4.2%)** - Quality of the view

## 📄 License

This project is open source and available for personal and educational use.
