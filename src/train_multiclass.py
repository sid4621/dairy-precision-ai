import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import dill
import os

print("--- Starting Multi-Class Symptom Checker Training ---")

# 1. Load Data
dataset_path = 'data/global_disease_detection.csv'
print(f"Loading dataset from {dataset_path}...")
df = pd.read_csv(dataset_path)

# 2. Select Features for a simple "Symptom Checker"
# We want features that a farmer can easily measure/input
features = [
    'Body_Temperature_C', 
    'Heart_Rate_bpm', 
    'Respiratory_Rate',
    'Feed_Quantity_kg', 
    'Milk_Yield_L',
    'Ambient_Temperature_C', 
    'Humidity_percent'
]
target = 'Disease_Status'

X = df[features]
y_raw = df[target]

print(f"Dataset shape: {X.shape}")
print(f"Number of unique classes: {y_raw.nunique()}")

# 3. Encode the Target (45 Classes)
le = LabelEncoder()
y = le.fit_transform(y_raw)

# 4. Train-Test Split (small sample of 50k for speed, stratify to keep rare diseases)
print("Splitting data...")
# We take a sample to make training fast on the user's local machine
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.3, stratify=y, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

# 5. Train XGBoost Multi-Class Model
print("Training XGBoost Multi-Class Classifier (this may take a minute)...")
clf = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(le.classes_),
    eval_metric='mlogloss',
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    n_jobs=-1,
    random_state=42
)

clf.fit(X_train, y_train)

# 6. Evaluate
preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Validation Accuracy: {acc * 100:.2f}%")

# 7. Save Model and Encoder
os.makedirs('models', exist_ok=True)
model_path = 'models/multiclass_disease_model.pkl'
encoder_path = 'models/disease_label_encoder.pkl'

print(f"Saving model to {model_path}...")
with open(model_path, 'wb') as f:
    dill.dump(clf, f)

print(f"Saving label encoder to {encoder_path}...")
with open(encoder_path, 'wb') as f:
    dill.dump(le, f)

print("--- Training Complete! ---")
print("Classes trained:", le.classes_[:5], "... and 40 more.")
