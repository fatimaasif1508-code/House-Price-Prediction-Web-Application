"""
Setup script to create model files for House Price Predictor
Run this first: python setup.py
"""
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import os

print("=" * 50)
print("House Price Predictor - Setup")
print("=" * 50)

# Check if data file exists
if not os.path.exists('houses.csv'):
    print("\n❌ houses.csv not found!")
    print("Please make sure houses.csv is in the same folder as this script.")
    print("\nCreating a demo model with sample data instead...")
    
    # Create sample data for demo
    import numpy as np
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
        'price': np.random.uniform(15, 150, n_samples)  # in $100k
    }
    
    # Adjust price based on key features
    for i in range(n_samples):
        data['price'][i] += data['sqft_living'][i] * 0.02  # $200 per sqft
        data['price'][i] += data['bedrooms'][i] * 5  # $500k per bedroom
        data['price'][i] += data['waterfront'][i] * 50  # $5M for waterfront
        data['price'][i] += data['view'][i] * 10  # $1M per view level
    
    df = pd.DataFrame(data)
    print(f"✓ Created sample dataset with {len(df)} records")
else:
    df = pd.read_csv('houses.csv')
    print(f"✓ Loaded houses.csv with {len(df)} records")

# Prepare features and target
X = df.drop('price', axis=1)
y = df['price']

print(f"✓ Features: {list(X.columns)}")

# Train model
print("\nTraining Linear Regression model...")
model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

model.fit(X, y)

# Evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"✓ Model trained!")
print(f"  R² Score: {r2:.4f}")
print(f"  RMSE: {rmse:.2f}")

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print(f"✓ Saved model.pkl")

# Save feature names
with open('features.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
print(f"✓ Saved features.pkl")

print("\n" + "=" * 50)
print("Setup complete! You can now run: python app.py")
print("=" * 50)
