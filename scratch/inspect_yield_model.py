import dill
import os

path = r"C:\Users\sidda\Desktop\cattle milk yeild prediction\Cattle Health and Feeding Data\models\milk_yield_model.pkl"
if os.path.exists(path):
    with open(path, 'rb') as f:
        model_dict = dill.load(f)
        features = model_dict.get('features', [])
        print(f"Count: {len(features)}")
        print("Features:", ",".join(features[:10]), "...")
        print("Model Type:", type(model_dict.get('model', 'Not found')))
else:
    print("Model not found")
