import pandas as pd
from xgboost import XGBClassifier
import joblib
import os

def train():
    print("Loading Mastitis Dataset...")
    path = r'C:\Users\sidda\Desktop\cattle milk yeild prediction\Clinical_Mastitis_cows_version2\Clinical_Mastitis_cows_version2\clinical_mastitis_cows.csv'
    df = pd.read_csv(path)

    # Features: Temperature, Udder Hardness, Pain, and EUFL (Exhale Udder Front Left limit proxy for swelling)
    # Target: class1 (0=Healthy, 1=Mastitis)
    
    features = ['Temperature', 'Hardness', 'Pain', 'EUFL']
    target = 'class1'

    X = df[features]
    y = df[target]

    print("Training XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X, y)
    
    # Calculate accuracy
    from sklearn.metrics import accuracy_score
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Training Accuracy: {acc * 100:.2f}%")

    # Save Model
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'mastitis_xgb.pkl')
    joblib.dump(model, model_path)
    print(f"Model successfully saved to {model_path}")

if __name__ == '__main__':
    train()
