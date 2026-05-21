import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import dill
import os

# 1. Train Disease Binary Classifier
df = pd.read_csv("data/global_disease_detection.csv")
df_disease = df.copy()
# Target: 1 if not healthy, 0 if healthy
df_disease['Binary_Target'] = (df_disease['Disease_Status'] != 'Healthy').astype(int)

# Dummify everything to get features
X_d = pd.get_dummies(df_disease.drop(columns=['Disease_Status', 'Binary_Target', 'Date', 'Cattle_ID']))
y_d = df_disease['Binary_Target']

# For speed, use subset
X_d = X_d.sample(n=10000, random_state=42)
y_d = y_d.loc[X_d.index]

clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
clf.fit(X_d, y_d)

with open('models/disease_model.pkl', 'wb') as f:
    dill.dump({'model': clf, 'features': list(X_d.columns)}, f)

# 2. Train Yield Regressor
df_y = pd.read_csv("data/global_yield_prediction.csv")
# Dummify everything to get features
X_y = pd.get_dummies(df_y.drop(columns=['Date', 'Cattle_ID', 'Milk_Yield_L']))
y_y = df_y['Milk_Yield_L']

# For speed, use subset
X_y = X_y.sample(n=10000, random_state=42)
y_y = y_y.loc[X_y.index]

reg = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
reg.fit(X_y, y_y)

with open('models/milk_yield_model.pkl', 'wb') as f:
    dill.dump({'model': reg, 'features': list(X_y.columns)}, f)

print("Successfully retrained missing models!")
