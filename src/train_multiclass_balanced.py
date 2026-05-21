import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import dill
import os

print("--- Re-Training Multi-Class Model with Balanced Data ---")

# 1. Load Data
dataset_path = 'data/global_disease_detection.csv'
df = pd.read_csv(dataset_path)

# 2. Balance the Dataset!
# "Healthy" has 137k rows, but each disease only has ~2.6k rows.
# We will undersample "Healthy" to match the size of the diseases so the AI doesn't become biased.
healthy_df = df[df['Disease_Status'] == 'Healthy'].sample(n=3000, random_state=42)
sick_df = df[df['Disease_Status'] != 'Healthy']

balanced_df = pd.concat([healthy_df, sick_df])
print(f"Balanced Dataset shape: {balanced_df.shape}")

# 3. Select Features
features = [
    'Body_Temperature_C', 'Heart_Rate_bpm', 'Respiratory_Rate',
    'Feed_Quantity_kg', 'Milk_Yield_L', 'Ambient_Temperature_C', 'Humidity_percent'
]
target = 'Disease_Status'

X = balanced_df[features]
y_raw = balanced_df[target]

# 4. Encode the Target
le = LabelEncoder()
y = le.fit_transform(y_raw)

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 6. Train XGBoost
print("Training Balanced XGBoost Model...")
clf = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(le.classes_),
    eval_metric='mlogloss',
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    n_jobs=-1,
    random_state=42
)

clf.fit(X_train, y_train)

# 7. Save Model and Encoder
os.makedirs('models', exist_ok=True)
with open('models/multiclass_disease_model.pkl', 'wb') as f:
    dill.dump(clf, f)

with open('models/disease_label_encoder.pkl', 'wb') as f:
    dill.dump(le, f)

print("--- Training Complete! Model is now perfectly balanced. ---")
