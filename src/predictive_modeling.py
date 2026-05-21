import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

PROCESSED_DATA_PATH = 'data/processed_lactations.csv'

def load_data():
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError("Please run data_processing.py first.")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    return df

def feature_engineering(df):
    """
    Prepare data for the model. 
    Here is where we can add 'health' features later!
    """
    print("Performing feature engineering...")
    
    # Target variable
    target = 'total_milk_yield'
    
    # We drop NAs in target to train safely
    df = df.dropna(subset=[target])
    
    # Selected features for the current dataset
    # Note: 'health_status', 'THI' or 'disease_flag' can be added here once data is available.
    features = ['lactation_number', 'length_of_lactation', 'days_dry', 'peak_yield', 'days_to_peak']
    
    # Drop rows with NAs in features for simplicity of first model
    df_model = df.dropna(subset=features).copy()
    
    X = df_model[features]
    y = df_model[target]
    
    return X, y, features

def train_and_evaluate(X, y, features):
    print("Splitting data into 80% train, 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n--- Training Random Forest Regressor ---")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred_rf = rf_model.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    
    print(f"Random Forest RMSE: {rmse_rf:.2f} kg")
    print(f"Random Forest R-Squared: {r2_rf:.3f} (Explains {r2_rf*100:.1f}% of variance)")
    
    print("\n--- Training Gradient Boosting Regressor ---")
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    
    y_pred_gb = gb_model.predict(X_test)
    rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
    r2_gb = r2_score(y_test, y_pred_gb)
    
    print(f"Gradient Boosting RMSE: {rmse_gb:.2f} kg")
    print(f"Gradient Boosting R-Squared: {r2_gb:.3f} (Explains {r2_gb*100:.1f}% of variance)")
    
    # Choose best model to plot
    best_model = rf_model if r2_rf > r2_gb else gb_model
    y_pred_best = y_pred_rf if r2_rf > r2_gb else y_pred_gb
    model_name = "Random Forest" if r2_rf > r2_gb else "Gradient Boosting"
    print(f"\nBest Model: {model_name}")
    
    # Feature Importance Plot
    plot_feature_importance(best_model, features, model_name)
    
    # Actual vs Predicted Plot
    plot_actual_vs_predicted(y_test, y_pred_best, model_name)
    
    return best_model

def plot_feature_importance(model, features, model_name):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances - {model_name}")
    sns.barplot(x=[importances[i] for i in indices], y=[features[i] for i in indices], palette="viridis")
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('results/feature_importance.png')
    plt.close()
    print("Saved feature importance plot to results/feature_importance.png")

def plot_actual_vs_predicted(y_test, y_pred, model_name):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Total Milk Yield (kg)')
    plt.ylabel('Predicted Total Milk Yield (kg)')
    plt.title(f'Actual vs Predicted Yield ({model_name})')
    plt.tight_layout()
    plt.savefig('results/actual_vs_predicted.png')
    plt.close()
    print("Saved prediction plot to results/actual_vs_predicted.png")

if __name__ == "__main__":
    df = load_data()
    X, y, features = feature_engineering(df)
    model = train_and_evaluate(X, y, features)
