import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import dill
import os
import sys

print("--- Training Multi-Output Milk Quality Predictor ---")

# 1. Load Data
dataset_path = 'data/milk_quality_performance.xlsx'
try:
    df = pd.read_excel(dataset_path)
except Exception as e:
    print(f"Error loading Excel file: {e}")
    # Provide a fallback just in case pandas excel engine fails
    sys.exit(1)

# Keep relevant columns
# Breed, Parity, Calving Season, CalvInt -> F (%), P (%)
features = ['Breed', 'Parity', 'Calving Season', 'CalvInt']
targets = ['F (%)', 'P (%)']

df = df[features + targets].dropna()

# 2. Encode Categorical Data
le_breed = LabelEncoder()
le_season = LabelEncoder()

df['Breed'] = le_breed.fit_transform(df['Breed'])
df['Calving Season'] = le_season.fit_transform(df['Calving Season'])

X = df[features]
y = df[targets]

print(f"Dataset shape after cleaning: {X.shape}")

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Multi-Output Random Forest Regressor
print("Training Multi-Output RandomForestRegressor...")
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
preds = model.predict(X_test)
mae_f = mean_absolute_error(y_test['F (%)'], preds[:, 0])
mae_p = mean_absolute_error(y_test['P (%)'], preds[:, 1])

r2_f = r2_score(y_test['F (%)'], preds[:, 0])
r2_p = r2_score(y_test['P (%)'], preds[:, 1])

print(f"Validation MAE (Fat %): {mae_f:.3f}")
print(f"Validation R2 Score (Fat %): {r2_f:.2f}")
print(f"Validation MAE (Protein %): {mae_p:.3f}")
print(f"Validation R2 Score (Protein %): {r2_p:.2f}")

# 6. Save Model and Encoders
os.makedirs('models', exist_ok=True)
model_path = 'models/milk_quality_model.pkl'
encoders_path = 'models/quality_label_encoders.pkl'

with open(model_path, 'wb') as f:
    dill.dump(model, f)

with open(encoders_path, 'wb') as f:
    dill.dump({
        'breed': le_breed,
        'season': le_season
    }, f)

print(f"--- Training Complete! Saved to {model_path} ---")
