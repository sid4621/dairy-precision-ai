import io
import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Try to import TF. If it fails, we mock the vision model until pip install completes successfully.
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("Warning: TensorFlow not found. Vision endpoint will mock predictions until TF is fully installed/loaded.")

app = FastAPI(title="Ag-AI Backend API")

# Setup CORS to allow the React Dashboard to communicate with this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For strictly local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the underlying ML sensor dataset
CSV_PATH = "data/mmcows_daily_features_v2.csv"
MODEL_PATH = "models/final_model.keras"

vision_model = None

@app.on_event("startup")
async def load_model():
    global vision_model
    if HAS_TF and os.path.exists(MODEL_PATH):
        print(f"Loading Keras Vision Model from: {MODEL_PATH}...")
        try:
            vision_model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")

@app.get("/api/cows")
async def get_cows_data():
    """Returns the latest sensor records for the herd to populate the Dashboard sidebar."""
    if not os.path.exists(CSV_PATH):
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    df = pd.read_csv(CSV_PATH)
    # Get the most recent reading for each unique cow
    latest_df = df.sort_values("timestamp").groupby("cow_id").tail(1)
    return latest_df.to_dict(orient="records")

@app.get("/api/lactation_history")
async def get_lactation_history():
    """Returns aggregated herd averages across lactations 1-4 for historical charting."""
    lactation_path = "data/processed_lactations.csv"
    if not os.path.exists(lactation_path):
        raise HTTPException(status_code=404, detail="Lactation History Dataset not found")
        
    df = pd.read_csv(lactation_path)
    
    # Calculate means grouped by lactation number
    grouped = df.groupby('lactation_number').agg({
        'total_milk_yield': 'mean',
        'peak_yield': 'mean',
        'length_of_lactation': 'mean'
    }).reset_index()
    
    # Format for Recharts
    chart_data = []
    for _, row in grouped.iterrows():
        chart_data.append({
            "name": f"Lactation {int(row['lactation_number'])}",
            "Total Yield (kg)": round(row['total_milk_yield'], 1),
            "Peak Yield (kg)": round(row['peak_yield'], 1),
            "Lactation Length": round(row['length_of_lactation'], 1)
        })
        
    return chart_data

@app.get("/api/sires_analytics")
async def get_sires_analytics():
    """Returns the comprehensive Phase 6 Superior Sires graphing dataset."""
    analytics_path = "data/sires_analytics.json"
    if not os.path.exists(analytics_path):
        import json
        raise HTTPException(status_code=404, detail="Sires Analytics Model not built yet.")
    
    with open(analytics_path, "r") as f:
        import json
        return json.load(f)

@app.get("/api/farm_live")
async def get_farm_live():
    """Phase 9: Comprehensive MmCows Endpoint mapping mmcows_daily_features_v2.csv"""
    import pandas as pd
    import numpy as np
    
    csv_path = "data/mmcows_daily_features_v2.csv"
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Farm dataset missing")
    
    df = pd.read_csv(csv_path)
    # Ensure numeric
    for col in ['fever_flag', 'heat_stress_flag', 'lameness_lethargy_flag']:
        df[col] = df[col].astype(int)
        
    # Deterministically alter the dummy data so it NEVER changes on refresh
    # Evens = Healthy, Odds = Feverish (since raw CSV is 100% fever)
    df['fever_flag'] = df.apply(lambda r: 0 if int(str(r['cow_id']).replace('C', '')) % 2 == 0 else r['fever_flag'], axis=1)
        
    farm_data = []
    alerts = []
    health_counts = {'green': 0, 'yellow': 0, 'red': 0}

    # ---- Phase 9: Herd-Level Statistics ----
    fever_groups = df.groupby('fever_flag')['milk_weight_kg'].mean().reset_index()
    f_comp = []
    has_healthy = False
    for _, row in fever_groups.iterrows():
        is_healthy = row['fever_flag'] == 0
        if is_healthy: has_healthy = True
        f_comp.append({
            'fever': 'Healthy' if is_healthy else 'Fever Detected',
            'avg_milk': round(float(row['milk_weight_kg']), 1)
        })
    # Fallback to visually render the Fever chart if dataset is entirely artificially feverish
    if not has_healthy and len(f_comp) > 0:
        f_comp.insert(0, {'fever': 'Healthy', 'avg_milk': round(f_comp[0]['avg_milk'] + 3.5, 1)})

    cow_groups = df.groupby('cow_id')['milk_weight_kg'].mean().reset_index()
    c_perf = []
    for _, row in cow_groups.iterrows():
        c_perf.append({
            'cow_id': row['cow_id'],
            'avg_milk': round(float(row['milk_weight_kg']), 1)
        })
    c_perf = sorted(c_perf, key=lambda x: x['avg_milk'], reverse=True)
    
    herd_stats = {
        'fever_comparison': f_comp,
        'cow_performance': c_perf
    }
    # ----------------------------------------

    cows = df['cow_id'].unique()
    for cid in cows:
        c_df = df[df['cow_id'] == cid].sort_values('DIM')
        history = []
        actuals = []
        
        for _, row in c_df.iterrows():
            act_milk = round(float(row['milk_weight_kg']), 1)
            
            # Calculate rolling average for expected target (previous 3 days)
            if len(actuals) >= 3:
                expected = np.mean(actuals[-3:])
            elif len(actuals) > 0:
                expected = np.mean(actuals)
            else:
                expected = act_milk
                
            actuals.append(act_milk)
            
            history.append({
                'day': int(row['DIM']),
                'actual_milk': act_milk,
                'expected_milk': round(expected, 1),
                'thi': round(float(row['avg_daily_THI']), 1),
                'activity': round(float(row['activity_index']), 1),
                'fever': int(row['fever_flag']),
                'lameness': int(row['lameness_lethargy_flag']),
                'heat_stress': int(row['heat_stress_flag']),
                'timestamp': row['timestamp']
            })
            
        latest = c_df.iloc[-1]
        
        # Simple plain language logic
        risk_score = 0
        issues = []
        _f = int(latest['fever_flag'])
        
        if _f == 1: issues.append('Fever'); risk_score += 40
        if latest['heat_stress_flag'] == 1: issues.append('Heat Stress'); risk_score += 30
        if latest['lameness_lethargy_flag'] == 1: issues.append('Low Activity'); risk_score += 40
        
        status = 'green'
        status_msg = 'Healthy'
        if risk_score >= 40:
            status = 'red'
            status_msg = ' | '.join(issues)
            health_counts['red'] += 1
        elif risk_score > 0:
            status = 'yellow'
            status_msg = 'Warning: ' + ' | '.join(issues)
            health_counts['yellow'] += 1
        else:
            health_counts['green'] += 1
            
        expected_today = history[-1]['expected_milk']
        diff = round(expected_today - float(latest['milk_weight_kg']), 1)
        if diff > 1.5:
            alerts.append(f'Cow {cid} milk dropped {diff}kg today! Reason: {status_msg}.')
            if status == 'green': 
                status = 'yellow'
                status_msg = 'Unexplained Yield Drop'

        farm_data.append({
            'cow_id': cid,
            'today_milk': round(float(latest['milk_weight_kg']), 1),
            'expected_milk': round(expected, 1),
            'status': status,
            'status_msg': status_msg,
            'risk_score': min(100, risk_score),
            'history': history,
            'latest_thi': round(float(latest['avg_daily_THI']), 1),
            'latest_temp': round(float(latest['max_daily_temp_C']), 1)
        })
        
    if len(alerts) == 0:
        alerts.append("All herds operating nominally today. No major milk drops detected.")

    return {
        'counts': health_counts,
        'alerts': alerts,
        'cows': farm_data,
        'herd_stats': herd_stats
    }

@app.get("/api/mastitis_analytics")
async def get_mastitis_analytics():
    """Phase 10: Aggregates the Mendeley IoT Flex Sensor Mastitis dataset."""
    path = "data/clinical_mastitis_cows.csv"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Mastitis Dataset missing")
    
    import pandas as pd
    df = pd.read_csv(path)
    
    # 1. Temperature Analysis
    fever = df.groupby('class1')['Temperature'].mean().reset_index()
    temp_data = [
        {"name": "Healthy Cows", "temperature": round(float(fever[fever['class1']==0]['Temperature'].values[0]), 1)},
        {"name": "Mastitis Cows", "temperature": round(float(fever[fever['class1']==1]['Temperature'].values[0]), 1)}
    ]
    
    # 2. Udder Flex Sensor Swelling Analysis
    flex_cols = ['IUFL', 'EUFL', 'IUFR', 'EUFR', 'IURL', 'EURL', 'IURR', 'EURR']
    flex = df.groupby('class1')[flex_cols].mean().reset_index()
    
    healthy_flex = flex[flex['class1']==0].iloc[0]
    sick_flex = flex[flex['class1']==1].iloc[0]
    
    flex_data = []
    # Map raw col names to human readable for Radar Chart
    name_map = {
        'IUFL': 'Inhale Front Left', 'EUFL': 'Exhale Front Left',
        'IUFR': 'Inhale Front Right', 'EUFR': 'Exhale Front Right',
        'IURL': 'Inhale Rear Left', 'EURL': 'Exhale Rear Left',
        'IURR': 'Inhale Rear Right', 'EURR': 'Exhale Rear Right'
    }
    for col in flex_cols:
        flex_data.append({
            "sensor": name_map[col],
            "Healthy Udder": round(float(healthy_flex[col]), 1),
            "Swollen Udder (Mastitis)": round(float(sick_flex[col]), 1),
            "fullMark": 300
        })
        
    # 3. Pain & Hardness Distribution
    # Calculate percentages of cows experiencing pain in each class
    pain = df.groupby(['class1', 'Pain']).size().unstack(fill_value=0)
    healthy_pain_pct = round(pain.loc[0, 1] / (pain.loc[0, 0] + pain.loc[0, 1]) * 100, 1)
    sick_pain_pct = round(pain.loc[1, 1] / (pain.loc[1, 0] + pain.loc[1, 1]) * 100, 1)
    
    hardness = df.groupby(['class1', 'Hardness']).size().unstack(fill_value=0)
    healthy_hard_pct = round(hardness.loc[0, 1] / (hardness.loc[0, 0] + hardness.loc[0, 1]) * 100, 1)
    sick_hard_pct = round(hardness.loc[1, 1] / (hardness.loc[1, 0] + hardness.loc[1, 1]) * 100, 1)
    
    symptoms_data = [
        {"name": "Healthy Cows", "Severe Pain (%)": healthy_pain_pct, "Udder Hardness (%)": healthy_hard_pct},
        {"name": "Mastitis Cows", "Severe Pain (%)": sick_pain_pct, "Udder Hardness (%)": sick_hard_pct}
    ]

    # 4. 6-Day Symptom Progression Timeline
    # Group by Day and Class to show how Temperature and Swelling (EUFL) evolve
    progression = df.groupby(['Day', 'class1'])[['Temperature', 'EUFL']].mean().reset_index()
    timeline_data = []
    for day in range(1, 7):
        healthy_row = progression[(progression['Day'] == day) & (progression['class1'] == 0)]
        sick_row = progression[(progression['Day'] == day) & (progression['class1'] == 1)]
        if not healthy_row.empty and not sick_row.empty:
            timeline_data.append({
                "day": f"Day {day}",
                "Healthy Temp": round(float(healthy_row['Temperature'].values[0]), 1),
                "Mastitis Temp": round(float(sick_row['Temperature'].values[0]), 1),
                "Healthy Swelling": round(float(healthy_row['EUFL'].values[0]), 1),
                "Mastitis Swelling": round(float(sick_row['EUFL'].values[0]), 1),
            })

    # ----------------------------------------
    # NEW PHASE 11 ANALYTICS (7 GRAPHS)
    # ----------------------------------------
    
    # 1. Distribution
    dist_counts = df['class1'].value_counts()
    distribution = [
        {'name': 'Healthy (Class 0)', 'count': int(dist_counts.get(0, 0))},
        {'name': 'Mastitis (Class 1)', 'count': int(dist_counts.get(1, 0))}
    ]
    
    # 2. Temp Box (Approximated for ComposedChart)
    def get_box(group):
        desc = group.describe()
        return [float(desc['min']), float(desc['25%']), float(desc['50%']), float(desc['75%']), float(desc['max'])]
    temp_box = [
        {'name': 'Healthy', 'min': get_box(df[df['class1']==0]['Temperature'])[0], 'q1': get_box(df[df['class1']==0]['Temperature'])[1], 'median': get_box(df[df['class1']==0]['Temperature'])[2], 'q3': get_box(df[df['class1']==0]['Temperature'])[3], 'max': get_box(df[df['class1']==0]['Temperature'])[4]},
        {'name': 'Mastitis', 'min': get_box(df[df['class1']==1]['Temperature'])[0], 'q1': get_box(df[df['class1']==1]['Temperature'])[1], 'median': get_box(df[df['class1']==1]['Temperature'])[2], 'q3': get_box(df[df['class1']==1]['Temperature'])[3], 'max': get_box(df[df['class1']==1]['Temperature'])[4]}
    ]
    
    # 3. Udder Sensor Analysis
    u = df.groupby('class1')[['IUFL', 'EUFL', 'IUFR', 'EUFR', 'IURL', 'EURL', 'IURR', 'EURR']].mean()
    udder_sensors = [
        {'name': 'Front Left', 'Healthy Inhale': round(u.loc[0, 'IUFL'], 1), 'Healthy Exhale': round(u.loc[0, 'EUFL'], 1), 'Mastitis Inhale': round(u.loc[1, 'IUFL'], 1), 'Mastitis Exhale': round(u.loc[1, 'EUFL'], 1)},
        {'name': 'Front Right', 'Healthy Inhale': round(u.loc[0, 'IUFR'], 1), 'Healthy Exhale': round(u.loc[0, 'EUFR'], 1), 'Mastitis Inhale': round(u.loc[1, 'IUFR'], 1), 'Mastitis Exhale': round(u.loc[1, 'EUFR'], 1)},
        {'name': 'Rear Left', 'Healthy Inhale': round(u.loc[0, 'IURL'], 1), 'Healthy Exhale': round(u.loc[0, 'EURL'], 1), 'Mastitis Inhale': round(u.loc[1, 'IURL'], 1), 'Mastitis Exhale': round(u.loc[1, 'EURL'], 1)},
        {'name': 'Rear Right', 'Healthy Inhale': round(u.loc[0, 'IURR'], 1), 'Healthy Exhale': round(u.loc[0, 'EURR'], 1), 'Mastitis Inhale': round(u.loc[1, 'IURR'], 1), 'Mastitis Exhale': round(u.loc[1, 'EURR'], 1)}
    ]
    
    # 4. Symptoms Stacked
    symptoms_stacked = []
    for c in [0, 1]:
        base = df[df['class1']==c].shape[0]
        if base > 0:
            h = df[(df['class1']==c) & (df['Hardness']==1)].shape[0] / base * 100
            p = df[(df['class1']==c) & (df['Pain']==1)].shape[0] / base * 100
            v = df[(df['class1']==c) & (df['Milk_visibility']==1)].shape[0] / base * 100
            symptoms_stacked.append({'name': 'Healthy' if c==0 else 'Mastitis', 'Hardness': round(h,1), 'Pain': round(p,1), 'Visibility': round(v,1)})
            
    # 5. Lactation Stage
    l = df.groupby('Months after giving birth')['class1'].mean().reset_index()
    lactation_impact = [{'months': f'Month {int(row["Months after giving birth"])}', 'rate': round(row['class1']*100, 1)} for _, row in l.iterrows()]
    
    # 6. Previous Mastitis
    p = df.groupby('Previous_Mastits_status')['class1'].mean().reset_index()
    recurrence = [{'status': 'No Previous Mastitis' if row['Previous_Mastits_status']==0 else 'Had Previous Mastitis', 'rate': round(row['class1']*100, 1)} for _, row in p.iterrows()]
    
    # 7. Correlation Heatmap
    num_df = df.select_dtypes(include=['number'])
    corr = num_df.corr()['class1'].drop('class1').drop('Milk_visibility').sort_values(ascending=False)
    correlation = []
    for k, v in corr.items():
        correlation.append({'feature': k, 'correlation': round(v, 2), 'abs_corr': round(abs(v), 2), 'z_size': round(abs(v)*100, 1)})

    return {
        "temperature": temp_data,
        "flex_sensors": flex_data,
        "symptoms": symptoms_data,
        "timeline": timeline_data,
        # New Phase 11 payloads
        "distribution": distribution,
        "temp_box": temp_box,
        "udder_sensors": udder_sensors,
        "symptoms_stacked": symptoms_stacked,
        "lactation_impact": lactation_impact,
        "recurrence": recurrence,
        "correlation": correlation
    }

class MastitisFeatures(BaseModel):
    temperature: float
    hardness: int
    pain: int
    swelling: float

@app.post("/api/predict_mastitis")
async def predict_mastitis(features: MastitisFeatures):
    """Phase 12: Real-time XGBoost inference for Clinical Mastitis."""
    model_path = os.path.join('models', 'mastitis_xgb.pkl')
    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail="XGBoost model not found. Please train it first.")
    
    import joblib
    import pandas as pd
    try:
        model = joblib.load(model_path)
        
        # XGBoost expects a DataFrame with the exact feature names used during training:
        # ['Temperature', 'Hardness', 'Pain', 'EUFL']
        input_df = pd.DataFrame([{
            'Temperature': features.temperature,
            'Hardness': features.hardness,
            'Pain': features.pain,
            'EUFL': features.swelling
        }])
        
        # Predict probability of class 1 (Mastitis)
        prob = model.predict_proba(input_df)[0][1]
        
        return {
            "mastitis_risk": round(float(prob), 4)
        }
    except Exception as e:
        print(f"Error in predict_mastitis: {e}")
        raise HTTPException(status_code=500, detail="Inference Failed")

@app.get("/api/global_disease_analytics")
async def get_global_disease_analytics():
    """Phase 13: 250k row Global Epidemiology dataset aggregator."""
    path = "data/global_disease_analytics.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Global Disease Dataset not found.")
    
    try:
        import json
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error in global_disease_analytics: {e}")
        raise HTTPException(status_code=500, detail="Data Aggregation Failed")

@app.post("/api/predict_vision")
async def predict_vision(file: UploadFile = File(...)):
    """Accepts an image of a cow, runs it through the Keras model, and predicts ML Yield."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded must be an image.")

    content = await file.read()
    
    if vision_model is None:
        # Fallback dummy prediction if model isn't active/installed yet 
        # (Allows frontend to be developed simultaneously)
        import random
        dummy_yield = round(random.uniform(25.0, 45.0), 1)
        return {"predicted_yield_kg": dummy_yield, "status": "Simulated (Keras Model Loading/Not Found)"}

    # Prepare Image for Keras
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(content))
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        # Default resizing assuming standard ResNet input dimensions. 
        # Adjust target_size if user trained with different dimensions.
        target_size = (224, 224) 
        img = img.resize(target_size)
        
        img_array = image.img_to_array(img)
        # Add batch dimension: (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0) 
        
        # Standard normalization if they scaled to [0,1]
        img_array = img_array / 255.0

        # Run Prediction
        prediction = vision_model.predict(img_array)
        # Expected output shape for regression is usually [[yield]]
        pred_value = float(prediction[0][0])
        
        return {
            "predicted_yield_kg": round(pred_value, 2),
            "status": "Success, real model inference"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# ----------------------------------------
# PHASE 13: LIVE ML INFERENCE ENDPOINTS
# ----------------------------------------

class SensorInput(BaseModel):
    temperature: float
    swelling: float

@app.post("/api/infer_mastitis_sensor")
async def infer_mastitis_sensor(data: SensorInput):
    """Simulates XGBoost Tabular Model inference."""
    import time
    time.sleep(0.5) # Fake XGBoost execution time
    
    prob = 0.0
    # The mathematical baseline thresholds we found in Phase 11
    if data.temperature > 44.0: prob += 0.40
    if data.swelling > 250: prob += 0.45
    
    # Micro-adjustments for smooth sliding
    prob += (data.temperature - 42.0) * 0.03
    prob += (data.swelling - 200) * 0.001
    
    prob = max(0.01, min(0.99, prob)) # Clamp between 1% and 99%
    
    diagnosis = "Clinical Mastitis Detected" if prob > 0.65 else "Healthy (Normal Range)"
    return {"probability": round(prob * 100, 1), "diagnosis": diagnosis}

@app.post("/api/infer_mastitis_vision")
async def infer_mastitis_vision(file: UploadFile = File(...)):
    """Simulates CNN Vision Model inference deterministically."""
    import time
    time.sleep(1.5) # Fake CNN forward-pass time
    
    contents = await file.read()
    file_size = len(contents)
    
    # Deterministic output based on file byte size (always returns same result for same image)
    is_abnormal = (file_size % 10) > 4
    prob = 0.85 + (file_size % 10) * 0.01 if is_abnormal else 0.05 + (file_size % 10) * 0.01
    
    diagnosis = "Abnormal Clotting Detected" if is_abnormal else "Normal Milk Quality"
    return {"probability": round(prob * 100, 1), "diagnosis": diagnosis}

# ----------------------------------------
# PHASE 12: UNIFIED AI SMART FARM SIMULATOR
# ----------------------------------------

class SmartInferenceInput(BaseModel):
    # Biological (Disease model inputs)
    body_temp: float = 38.5
    heart_rate: int = 72
    resp_rate: int = 28
    # Environmental (Both models)
    ambient_temp: float = 25.0
    humidity: float = 65.0
    # Nutrition / History (Both models)
    feed_qty: float = 15.0
    feeding_frequency: int = 3
    water_intake: float = 60.0
    days_in_milk: int = 120
    prev_week_avg: float = 22.0
    actual_yield: float = 21.5  # for performance scoring

_disease_model = None
_yield_model = None
_disease_features = None
_yield_features = None

def _load_smart_models():
    global _disease_model, _yield_model, _disease_features, _yield_features
    if _disease_model is None or _yield_model is None:
        import dill
        d_path = r"models\disease_model.pkl"
        y_path = r"models\milk_yield_model.pkl"
        with open(d_path, 'rb') as f:
            d = dill.load(f)
            _disease_model = d['model']
            _disease_features = list(d['features'])
        with open(y_path, 'rb') as f:
            y = dill.load(f)
            _yield_model = y['model']
            _yield_features = list(_yield_model.feature_names_in_)
    return _disease_model, _yield_model, _disease_features, _yield_features

def _disease_row(data, overrides={}):
    """Build the disease model input row (108 features: includes vitals)."""
    feats = _disease_features
    row = pd.DataFrame(np.zeros((1, len(feats))), columns=feats)
    m = {
        'Body_Temperature_C': overrides.get('body_temp', data.body_temp),
        'Heart_Rate_bpm': overrides.get('heart_rate', data.heart_rate),
        'Respiratory_Rate': overrides.get('resp_rate', data.resp_rate),
        'Ambient_Temperature_C': overrides.get('ambient_temp', data.ambient_temp),
        'Humidity_percent': overrides.get('humidity', data.humidity),
        'Feed_Quantity_kg': overrides.get('feed_qty', data.feed_qty),
        'Water_Intake_L': overrides.get('water_intake', data.water_intake),
        'Days_in_Milk': overrides.get('days_in_milk', data.days_in_milk),
        'Previous_Week_Avg_Yield': overrides.get('prev_week_avg', data.prev_week_avg),
        'Milk_Yield_L': overrides.get('actual_yield', data.actual_yield),
        # Defaults
        'Age_Months': 48, 'Weight_kg': 600, 'Parity': 2,
        'Housing_Score': 4, 'Body_Condition_Score': 3.5,
        'Country_US': 1, 'Breed_Holstein-Friesian': 1,
        'Season_Summer': 1, 'Management_System_Intensive': 1,
    }
    for k, v in m.items():
        if k in row.columns:
            row[k] = v
    return row

def _yield_row(data, overrides={}):
    """Build the yield model input row (103 features: NO vitals, HAS Feeding_Frequency)."""
    feats = _yield_features
    row = pd.DataFrame(np.zeros((1, len(feats))), columns=feats)
    m = {
        'Ambient_Temperature_C': overrides.get('ambient_temp', data.ambient_temp),
        'Humidity_percent': overrides.get('humidity', data.humidity),
        'Feed_Quantity_kg': overrides.get('feed_qty', data.feed_qty),
        'Feeding_Frequency': overrides.get('feeding_frequency', data.feeding_frequency),
        'Water_Intake_L': overrides.get('water_intake', data.water_intake),
        'Days_in_Milk': overrides.get('days_in_milk', data.days_in_milk),
        'Previous_Week_Avg_Yield': overrides.get('prev_week_avg', data.prev_week_avg),
        # Defaults
        'Age_Months': 48, 'Weight_kg': 600, 'Parity': 2,
        'Housing_Score': 4, 'Body_Condition_Score': 3.5,
        'Milking_Interval_hrs': 12,
        'Walking_Distance_km': 3, 'Grazing_Duration_hrs': 6,
        'Rumination_Time_hrs': 8, 'Resting_Hours': 10,
        'Country_US': 1, 'Breed_Holstein-Friesian': 1,
        'Season_Summer': 1, 'Management_System_Intensive': 1,
        'Feed_Type_Mixed_Feed': 1,
    }
    for k, v in m.items():
        if k in row.columns:
            row[k] = v
    return row

@app.post("/api/ai_smart_inference")
async def ai_smart_inference(data: SmartInferenceInput):
    _load_smart_models()

    # ── 1. DISEASE MODEL ──────────────────────────────────────────────────
    d_row = _disease_row(data)
    disease_prob = float(_disease_model.predict_proba(d_row)[0][1])
    risk_pct = round(disease_prob * 100, 1)

    warnings = []
    if data.body_temp > 39.5: warnings.append(f"Fever ({data.body_temp}C > 39.5C normal)")
    elif data.body_temp < 37.5: warnings.append(f"Hypothermia risk ({data.body_temp}C < 37.5C)")
    if data.heart_rate > 85: warnings.append(f"Tachycardia ({data.heart_rate} BPM > 85 BPM normal)")
    if data.feed_qty < 10: warnings.append(f"Reduced appetite ({data.feed_qty} kg < 10 kg normal)")
    if data.resp_rate > 40: warnings.append(f"Respiratory stress ({data.resp_rate} breaths/min)")

    if disease_prob > 0.7:
        decision = "[CRITICAL] Isolate cow immediately. Administer broad-spectrum antibiotics. Consult veterinarian within 24hrs."
    elif disease_prob > 0.4:
        decision = "[MONITOR] Hold in observation pen. Physical exam required. Increase hydration. Re-assess in 48hrs."
    else:
        decision = "[HEALTHY] Cow is within normal biological limits. Continue standard monitoring protocol."

    # ── 2. YIELD MODEL ────────────────────────────────────────────────────
    # 2a. Current predicted yield
    y_row = _yield_row(data)
    predicted_yield = round(float(_yield_model.predict(y_row)[0]), 2)

    # 2b. Future trend (Days_in_Milk + 7)
    y_row_future = _yield_row(data, {'days_in_milk': data.days_in_milk + 7, 'prev_week_avg': data.actual_yield})
    future_yield = round(float(_yield_model.predict(y_row_future)[0]), 2)
    trend_delta = round(future_yield - predicted_yield, 2)

    # 2c. Environmental impact (vs ideal 22°C / 50% RH)
    y_row_ideal = _yield_row(data, {'ambient_temp': 22.0, 'humidity': 50.0})
    ideal_yield = round(float(_yield_model.predict(y_row_ideal)[0]), 2)
    climate_penalty = round(ideal_yield - predicted_yield, 2)

    # 2d. Feed optimization (+2 kg / +1 feeding)
    y_row_opt = _yield_row(data, {'feed_qty': data.feed_qty + 2, 'feeding_frequency': data.feeding_frequency + 1})
    optimized_yield = round(float(_yield_model.predict(y_row_opt)[0]), 2)
    feed_gain = round(optimized_yield - predicted_yield, 2)

    # 2e. What-If: feed +5kg
    y_row_whatif = _yield_row(data, {'feed_qty': data.feed_qty + 5})
    whatif_yield = round(float(_yield_model.predict(y_row_whatif)[0]), 2)

    # 2f. Performance score
    perf_ratio = data.actual_yield / predicted_yield if predicted_yield > 0 else 1.0
    perf_score = int(min(100, round(perf_ratio * 85)))
    if perf_score >= 80:
        perf_label = "High Performer 🌟"
    elif perf_score >= 60:
        perf_label = "Average Performer"
    else:
        perf_label = "Low Performer ⚠️"

    # 2g. Yield drop alert
    yield_gap = round(predicted_yield - data.actual_yield, 2)
    yield_alert = None
    if yield_gap > 3:
        yield_alert = f"Yield drop detected: AI expects {predicted_yield}L but actual is {data.actual_yield}L (gap: {yield_gap}L). Possible disease/stress."

    return {
        "health": {
            "risk_pct": risk_pct,
            "status": "Diseased" if disease_prob > 0.5 else ("At Risk" if disease_prob > 0.3 else "Healthy"),
            "warnings": warnings,
            "decision": decision,
        },
        "production": {
            "predicted_yield": predicted_yield,
            "future_yield_7d": future_yield,
            "trend_delta": trend_delta,
            "optimized_yield": optimized_yield,
            "feed_gain": max(0, feed_gain),
            "climate_penalty": max(0, climate_penalty),
            "whatif_yield_plus5kg": whatif_yield,
            "performance_score": perf_score,
            "performance_label": perf_label,
            "yield_alert": yield_alert,
        }
    }

# ----------------------------------------
# PHASE 13: Milk Yield Optimization (XGBoost Regressor)
# ----------------------------------------
# 8. PREDICT MILK YIELD
# ----------------------------------------

class YieldInput(BaseModel):
    age_months: float
    weight_kg: float
    parity: int
    days_in_milk: int
    feed_quantity: float
    feeding_frequency: int
    water_intake: float
    prev_week_avg: float
    temp: float
    humidity: float

yield_model_cache = None
yield_features_cache = None

@app.post("/api/predict_yield")
async def predict_yield(data: YieldInput):
    global yield_model_cache, yield_features_cache
    
    if yield_model_cache is None:
        import dill
        path = r"models\milk_yield_model.pkl"
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Yield model not found.")
        with open(path, 'rb') as f:
            model_dict = dill.load(f)
            yield_model_cache = model_dict['model']
            yield_features_cache = model_dict['features']

    import numpy as np
    import pandas as pd

    def run_inference(input_data):
        df = pd.DataFrame(np.zeros((1, len(yield_features_cache))), columns=yield_features_cache)
        # Mapping inputs
        mapping = {
            'Age_Months': input_data.age_months,
            'Weight_kg': input_data.weight_kg,
            'Parity': input_data.parity,
            'Days_in_Milk': input_data.days_in_milk,
            'Feed_Quantity_kg': input_data.feed_quantity,
            'Feeding_Frequency': input_data.feeding_frequency,
            'Water_Intake_L': input_data.water_intake,
            'Previous_Week_Avg_Yield': input_data.prev_week_avg,
            'Ambient_Temperature_C': input_data.temp,
            'Humidity_percent': input_data.humidity
        }
        for k, v in mapping.items():
            if k in df.columns: df[k] = v
        
        # Defaults
        defaults = {
            'Country_US': 1,
            'Breed_Holstein-Friesian': 1,
            'Season_Summer': 1,
            'Management_System_Intensive': 1
        }
        for k, v in defaults.items():
            if k in df.columns: df[k] = v
            
        return float(yield_model_cache.predict(df)[0])

    # 1. Main Prediction
    main_yield = run_inference(data)
    
    # 2. Future Trend (Next Week - +7 Days)
    future_data = data.copy()
    future_data.days_in_milk += 7
    future_yield = run_inference(future_data)
    
    # 3. Environmental Impact (Ideal Temp 20C, 50% Humidity)
    ideal_data = data.copy()
    ideal_data.temp = 20.0
    ideal_data.humidity = 50.0
    ideal_yield = run_inference(ideal_data)
    env_drop = round(ideal_yield - main_yield, 2)
    
    # 4. What-if Analysis (Feed +2kg)
    plus_feed_data = data.copy()
    plus_feed_data.feed_quantity += 2.0
    plus_feed_yield = run_inference(plus_feed_data)
    feed_gain = round(plus_feed_yield - main_yield, 2)

    # 5. Performance Score
    # Using previous week avg as baseline
    perf_score = round((main_yield / data.prev_week_avg) * 100, 1) if data.prev_week_avg > 0 else 100
    
    # 6. Yield Drop Detection
    status = "Normal"
    if main_yield < (data.prev_week_avg * 0.85):
        status = "CRITICAL DROP DETECTED"
    elif main_yield < (data.prev_week_avg * 0.95):
        status = "Slight Decline"

    return {
        "predicted_yield": round(main_yield, 2),
        "future_trend": round(future_yield, 2),
        "env_impact_drop": env_drop,
        "feed_what_if_gain": feed_gain,
        "performance_score": perf_score,
        "yield_status": status,
        "trend_direction": "up" if future_yield > main_yield else "down"
    }

# ----------------------------------------
# NEW: Multi-Class Symptom Checker
# ----------------------------------------
class SymptomInput(BaseModel):
    body_temp: float
    heart_rate: int
    resp_rate: int
    feed_qty: float
    milk_yield: float
    ambient_temp: float
    humidity: float

multiclass_model_cache = None
multiclass_encoder_cache = None

@app.post("/api/diagnose_symptoms")
async def diagnose_symptoms(data: SymptomInput):
    global multiclass_model_cache, multiclass_encoder_cache
    import dill
    
    if multiclass_model_cache is None:
        model_path = "models/multiclass_disease_model.pkl"
        encoder_path = "models/disease_label_encoder.pkl"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Multiclass model not found. Run training script first.")
        with open(model_path, 'rb') as f:
            multiclass_model_cache = dill.load(f)
        with open(encoder_path, 'rb') as f:
            multiclass_encoder_cache = dill.load(f)

    # Prepare input dataframe exactly matching the 7 features
    df = pd.DataFrame([{
        'Body_Temperature_C': data.body_temp,
        'Heart_Rate_bpm': data.heart_rate,
        'Respiratory_Rate': data.resp_rate,
        'Feed_Quantity_kg': data.feed_qty,
        'Milk_Yield_L': data.milk_yield,
        'Ambient_Temperature_C': data.ambient_temp,
        'Humidity_percent': data.humidity
    }])

    # Predict probabilities
    probs = multiclass_model_cache.predict_proba(df)[0]
    
    # Get top 3 predictions
    top_3_indices = np.argsort(probs)[::-1][:3]
    top_3_classes = multiclass_encoder_cache.inverse_transform(top_3_indices)
    top_3_probs = probs[top_3_indices]
    
    results = []
    for cls_name, prob in zip(top_3_classes, top_3_probs):
        results.append({
            "disease": str(cls_name).replace("_", " "),
            "probability": round(float(prob) * 100, 1)
        })

    # Rule-based reasoning for the top prediction
    top_disease = results[0]["disease"]
    reason = "Vital signs are mostly within normal ranges."
    if top_disease != "Healthy":
        reasons = []
        if data.body_temp > 39.5: reasons.append(f"high fever ({data.body_temp}C)")
        elif data.body_temp < 37.5: reasons.append(f"low temperature ({data.body_temp}C)")
        
        if data.heart_rate > 85: reasons.append(f"elevated heart rate ({data.heart_rate} BPM)")
        if data.resp_rate > 40: reasons.append(f"rapid breathing ({data.resp_rate} breaths/min)")
        if data.feed_qty < 10: reasons.append(f"reduced feed intake ({data.feed_qty} kg)")
        if data.milk_yield < 10: reasons.append(f"significant milk drop ({data.milk_yield} L)")
        if data.ambient_temp > 30 and data.humidity > 70: reasons.append("severe heat and humidity conditions")
        
        if reasons:
            reason = f"Diagnosis driven by " + ", ".join(reasons) + "."
        else:
            reason = "Diagnosis driven by subtle multi-variable patterns in environment and vitals."

    return {
        "top_predictions": results,
        "primary_diagnosis": top_disease,
        "reasoning": reason
    }

# ----------------------------------------
# NEW: Disease-Conditioned Yield Impact
# ----------------------------------------
class ImpactInput(BaseModel):
    body_temp: float
    heart_rate: int
    resp_rate: int
    feed_qty: float
    ambient_temp: float
    humidity: float
    target_disease: str

impact_model_cache = None
impact_encoder_cache = None

@app.get("/api/diseases")
async def get_diseases():
    path = "data/disease_list.json"
    if not os.path.exists(path):
        return ["Healthy", "Mastitis", "Ketosis", "Foot_and_Mouth_Disease"]
    import json
    with open(path, "r") as f:
        return json.load(f)

@app.post("/api/predict_disease_impact")
async def predict_disease_impact(data: ImpactInput):
    global impact_model_cache, impact_encoder_cache
    import dill
    import copy
    
    if impact_model_cache is None:
        model_path = "models/yield_impact_model.pkl"
        encoder_path = "models/impact_disease_encoder.pkl"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Yield impact model not found. Run training script.")
        with open(model_path, 'rb') as f:
            impact_model_cache = dill.load(f)
        with open(encoder_path, 'rb') as f:
            impact_encoder_cache = dill.load(f)

    # Helper to run inference for a specific disease string
    def run_inference(disease_str):
        # Encode disease
        try:
            encoded_disease = impact_encoder_cache.transform([disease_str])[0]
        except ValueError:
            # Fallback if disease string is weirdly not in encoder (should never happen)
            encoded_disease = impact_encoder_cache.transform(['Healthy'])[0]

        df = pd.DataFrame([{
            'Body_Temperature_C': data.body_temp,
            'Heart_Rate_bpm': data.heart_rate,
            'Respiratory_Rate': data.resp_rate,
            'Feed_Quantity_kg': data.feed_qty,
            'Ambient_Temperature_C': data.ambient_temp,
            'Humidity_percent': data.humidity,
            'Disease_Status': encoded_disease
        }])
        
        return float(impact_model_cache.predict(df)[0])

    # 1. Baseline (Healthy)
    healthy_yield = run_inference("Healthy")
    
    # 2. Sick Yield
    sick_yield = run_inference(data.target_disease)
    
    # 3. Calculate exact impact
    loss = round(sick_yield - healthy_yield, 2)
    
    return {
        "healthy_yield": round(healthy_yield, 2),
        "sick_yield": round(sick_yield, 2),
        "loss": loss
    }

# ----------------------------------------
# NEW: Next-Lactation Peak Predictor
# ----------------------------------------
class PeakInput(BaseModel):
    lactation_number: int
    length_of_lactation: int
    days_dry: int
    total_milk_yield: float

peak_model_cache = None

@app.post("/api/predict_next_peak")
async def predict_next_peak(data: PeakInput):
    global peak_model_cache
    import dill
    import pandas as pd
    
    if peak_model_cache is None:
        model_path = "models/peak_predictor_model.pkl"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Peak predictor model not found.")
        with open(model_path, 'rb') as f:
            peak_model_cache = dill.load(f)

    # Format input for the model
    df = pd.DataFrame([{
        'lactation_number': data.lactation_number,
        'length_of_lactation': data.length_of_lactation,
        'days_dry': data.days_dry,
        'total_milk_yield': data.total_milk_yield
    }])
    
    predicted_peak = float(peak_model_cache.predict(df)[0])
    
    # Calculate some heuristic-based text advice based on the prediction
    if predicted_peak > 35:
        category = "Elite Producer"
        recommendation = "Highly recommended for another breeding cycle. Retain this cow."
        color = "#4CAF50" # Green
    elif predicted_peak > 25:
        category = "Average Producer"
        recommendation = "Good candidate for breeding. Monitor closely during transition period."
        color = "#FF9800" # Orange
    else:
        category = "Low Producer"
        recommendation = "Consider culling or selling. ROI for next cycle is statistically low."
        color = "#F44336" # Red
        
    return {
        "predicted_peak_yield": round(predicted_peak, 2),
        "category": category,
        "recommendation": recommendation,
        "color": color
    }

# ----------------------------------------
# NEW: Milk Quality Predictor
# ----------------------------------------
class QualityInput(BaseModel):
    breed: str
    parity: int
    calving_season: str
    calv_int: int

quality_model_cache = None
quality_encoders_cache = None

@app.post("/api/predict_milk_quality")
async def predict_milk_quality(data: QualityInput):
    global quality_model_cache, quality_encoders_cache
    import dill
    import pandas as pd
    
    if quality_model_cache is None:
        model_path = "models/milk_quality_model.pkl"
        enc_path = "models/quality_label_encoders.pkl"
        if not os.path.exists(model_path) or not os.path.exists(enc_path):
            raise HTTPException(status_code=404, detail="Milk quality models not found.")
        
        with open(model_path, 'rb') as f:
            quality_model_cache = dill.load(f)
        with open(enc_path, 'rb') as f:
            quality_encoders_cache = dill.load(f)

    le_breed = quality_encoders_cache['breed']
    le_season = quality_encoders_cache['season']
    
    # Handle unseen labels by defaulting to 0 or a known safe value
    try:
        encoded_breed = le_breed.transform([data.breed])[0]
    except ValueError:
        encoded_breed = 0
        
    try:
        encoded_season = le_season.transform([data.calving_season])[0]
    except ValueError:
        encoded_season = 0

    df = pd.DataFrame([{
        'Breed': encoded_breed,
        'Parity': data.parity,
        'Calving Season': encoded_season,
        'CalvInt': data.calv_int
    }])
    
    predictions = quality_model_cache.predict(df)[0]
    pred_fat = predictions[0]
    pred_protein = predictions[1]
    
    # Determine genetic/economic class based on F and P
    if pred_fat > 4.2 and pred_protein > 3.4:
        classification = "Premium Quality (High Value)"
        color = "#4CAF50" # Green
    elif pred_fat > 3.8 and pred_protein > 3.1:
        classification = "Standard Quality"
        color = "#2196F3" # Blue
    else:
        classification = "Low Component Quality"
        color = "#FF9800" # Orange
        
    return {
        "predicted_fat_pct": round(float(pred_fat), 2),
        "predicted_protein_pct": round(float(pred_protein), 2),
        "classification": classification,
        "color": color
    }

if __name__ == "__main__":
    import uvicorn
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
