import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import dill
import os

print("--- Training Disease-Conditioned Yield Impact Model ---")

# 1. Load Data
dataset_path = 'data/global_disease_detection.csv'
df = pd.read_csv(dataset_path)

# 2. Select Features
# We want to predict yield, so Milk_Yield_L is our target.
# We include Disease_Status as the crucial feature to let the model learn the exact impact.
features = [
    'Body_Temperature_C', 
    'Heart_Rate_bpm', 
    'Respiratory_Rate',
    'Feed_Quantity_kg', 
    'Ambient_Temperature_C', 
    'Humidity_percent',
    'Disease_Status'
]
target = 'Milk_Yield_L'

X = df[features].copy()
y = df[target]

# 3. Encode Disease Status
# We need to save this encoder so the API knows which number maps to which disease.
le = LabelEncoder()
X['Disease_Status'] = le.fit_transform(X['Disease_Status'])

# 4. Train-Test Split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train XGBoost Regressor
print("Training XGBoost Regressor (this may take a minute)...")
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# 6. Evaluate
preds = model.predict(X_test)
mae = np.mean(np.abs(y_test - preds))
print(f"Validation Mean Absolute Error (MAE): {mae:.2f} Liters")

# 7. Save Model and Encoder
os.makedirs('models', exist_ok=True)
model_path = 'models/yield_impact_model.pkl'
encoder_path = 'models/impact_disease_encoder.pkl'

with open(model_path, 'wb') as f:
    dill.dump(model, f)

with open(encoder_path, 'wb') as f:
    dill.dump(le, f)

# Also save a list of just the diseases to a quick JSON for the frontend to populate the dropdown
import json
disease_names = sorted(df['Disease_Status'].unique().tolist())
with open('data/disease_list.json', 'w') as f:
    json.dump(disease_names, f)

print(f"--- Training Complete! Saved to {model_path} ---")
