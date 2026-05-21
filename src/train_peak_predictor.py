import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import dill
import os

print("--- Training Next-Lactation Peak Predictor ---")

# 1. Load Data
dataset_path = 'data/processed_lactations.csv'
df = pd.read_csv(dataset_path)

# Drop any rows with NaNs in our specific columns
features = ['lactation_number', 'length_of_lactation', 'days_dry', 'total_milk_yield']
target = 'peak_yield'

df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# 2. Train-Test Split
print(f"Data shape after cleaning: {X.shape}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Gradient Boosting Regressor
print("Training Gradient Boosting Regressor...")
model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# 4. Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"Validation MAE: {mae:.2f} Liters")
print(f"Validation R2 Score: {r2:.2f}")

# 5. Save Model
os.makedirs('models', exist_ok=True)
model_path = 'models/peak_predictor_model.pkl'

with open(model_path, 'wb') as f:
    dill.dump(model, f)

print(f"--- Training Complete! Saved to {model_path} ---")
