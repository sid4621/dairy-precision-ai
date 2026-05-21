import os
import json
import pandas as pd

def build_json():
    path = "data/global_disease_detection.csv"
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return
    
    print("Reading CSV... This may take a moment.")
    df = pd.read_csv(path)
    
    # 1. Distribution
    print("Processing Distribution...")
    dist_counts = df['Disease_Status'].value_counts()
    top_15 = dist_counts.head(15)
    distribution = [{'name': str(k), 'count': int(v)} for k, v in top_15.items()]
    distribution.append({'name': 'Other Illnesses (29 Types)', 'count': int(dist_counts[15:].sum())})

    # 2. Vitals
    print("Processing Vitals...")
    sick_df = df[df['Disease_Status'] != 'Healthy']
    top_10_sick = sick_df['Disease_Status'].value_counts().head(10).index
    vitals_df = df[df['Disease_Status'].isin(top_10_sick)].groupby('Disease_Status')[['Body_Temperature_C', 'Heart_Rate_bpm']].mean().reset_index()
    vitals = [{'disease': row['Disease_Status'].replace('_', ' '), 'Temperature': round(row['Body_Temperature_C'], 2), 'HeartRate': round(row['Heart_Rate_bpm'], 1)} for _, row in vitals_df.iterrows()]

    # 3. Environment Impact
    print("Processing Environment...")
    env = df.groupby(['Climate_Zone', 'Season']).apply(lambda x: (x['Disease_Status'] != 'Healthy').mean() * 100).reset_index(name='Infection_Rate')
    environment = []
    for cz in env['Climate_Zone'].unique():
        d = {'Climate': cz}
        for s in env['Season'].unique():
            rate = env[(env['Climate_Zone']==cz) & (env['Season']==s)]['Infection_Rate'].values
            d[s] = round(rate[0], 1) if len(rate) > 0 else 0
        environment.append(d)

    # 4. Nutrition Impact
    print("Processing Nutrition...")
    nut = df.groupby('Feed_Type').apply(lambda x: (x['Disease_Status'] != 'Healthy').mean() * 100).reset_index(name='rate')
    nutrition = [{'feed': row['Feed_Type'].replace('_', ' '), 'Disease Rate': round(row['rate'], 1)} for _, row in nut.iterrows()]

    # 5. Vaccination Efficacy
    print("Processing Vaccination...")
    vax = df.groupby('FMD_Vaccine').apply(lambda x: (x['Disease_Status'] == 'Foot_and_Mouth').mean() * 100).reset_index(name='rate')
    vaccination = [{'status': 'Unvaccinated' if row['FMD_Vaccine']==0 else 'Vaccinated', 'FMD Infection Rate': round(row['rate'], 2)} for _, row in vax.iterrows()]

    # 6. Activity Tracking
    print("Processing Activity...")
    act = df.groupby(df['Disease_Status'] == 'Healthy')[['Walking_Distance_km', 'Resting_Hours']].mean().reset_index()
    activity = [{'status': 'Healthy' if row['Disease_Status'] else 'Sick / Infected', 'Walking (km)': round(row['Walking_Distance_km'], 1), 'Resting (hrs)': round(row['Resting_Hours'], 1)} for _, row in act.iterrows()]

    # 7. Correlation Heatmap
    print("Processing Correlation...")
    df['is_sick'] = (df['Disease_Status'] != 'Healthy').astype(int)
    num_df = df.select_dtypes(include=['number'])
    corr = num_df.corr()['is_sick'].drop(['is_sick', 'Milk_Yield_L']).sort_values(ascending=False).head(15)
    correlation = [{'feature': k.replace('_', ' '), 'correlation': round(v, 3), 'abs_corr': round(abs(v), 3), 'z_size': round(abs(v)*200, 1)} for k, v in corr.items()]

    payload = {
        "distribution": distribution,
        "vitals": vitals,
        "environment": environment,
        "nutrition": nutrition,
        "vaccination": vaccination,
        "activity": activity,
        "correlation": correlation
    }
    
    out_path = "data/global_disease_analytics.json"
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=4)
        
    print(f"Successfully generated {out_path}!")

if __name__ == "__main__":
    build_json()
