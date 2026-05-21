import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_PATH = 'data/mmcows_daily_features.csv'

def load_data():
    return pd.read_csv(DATA_PATH)

def train_daily_model(df):
    print("Training Daily Yield Model with Health Sensors...")
    
    # Target variable
    target = 'milk_weight_kg'
    
    # Features (Health & Environment)
    features = ['DIM', 'avg_daily_temp_C', 'max_daily_temp_C', 'fever_flag', 
                'avg_daily_THI', 'avg_daily_barn_temp_F', 'heat_stress_flag']
    
    # Drop NaNs
    df_model = df.dropna(subset=[target] + features)
    
    X = df_model[features]
    y = df_model[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Daily Yield Prediction Results ---")
    print(f"RMSE: {rmse:.2f} kg/day")
    print(f"MAE: {mae:.2f} kg/day")
    print(f"R-Squared: {r2:.3f}")
    
    # Plot feature importance
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("What Impacts Daily Milk Yield the Most? (MmCows Sensor Model)")
    sns.barplot(x=[importances[i] for i in indices], y=[features[i] for i in indices], palette="magma")
    plt.xlabel('Relative Importance (Contribution to Prediction)')
    plt.tight_layout()
    plt.savefig('results/sensor_health_impact.png')
    plt.close()
    print("Saved feature importance to results/sensor_health_impact.png")
    
    import dill
    import os
    os.makedirs('models', exist_ok=True)
    with open('models/daily_yield_model.pkl', 'wb') as f:
        dill.dump({'model': rf_model, 'features': features}, f)
    print("Saved model to models/daily_yield_model.pkl")
    
if __name__ == "__main__":
    df = load_data()
    train_daily_model(df)
