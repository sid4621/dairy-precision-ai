import os

replacements = [
    ("Cattle Health and Feeding Data/global_cattle_disease_detection_dataset.csv", "data/global_disease_detection.csv"),
    ("catlle milk protein fat xl sheet/Dataset cow performance.xlsx", "data/milk_quality_performance.xlsx"),
    (r"C:\Users\sidda\Desktop\cattle milk yeild prediction\Cattle Health and Feeding Data\models\disease_model.pkl", r"models\disease_model.pkl"),
    (r"C:\Users\sidda\Desktop\cattle milk yeild prediction\Cattle Health and Feeding Data\models\milk_yield_model.pkl", r"models\milk_yield_model.pkl"),
]

for filename in os.listdir("src"):
    if filename.endswith(".py"):
        filepath = os.path.join("src", filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        original_content = content
        for old, new in replacements:
            content = content.replace(old, new)
            
        if content != original_content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Updated paths in {filename}")

print("Done replacing paths.")
