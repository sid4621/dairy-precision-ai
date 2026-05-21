import pandas as pd
import json
import os
import numpy as np

def convert_np(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError

def generate_sires_analytics():
    print("Loading Superior Sires Excel File...")
    xl = pd.ExcelFile('raw data for performance of superior dairy cattle sires_2.xlsx')
    
    analytics = {}

    # 1. (Removed EBV)
    # 2. Attrition Funnel
    print("Calculating Attrition...")
    cows_per_lac = []
    all_dfs = []
    
    for i in range(1, 5):
        sheet_name = f'{i}st Lactation' if i==1 else f'{i}nd Lactation' if i==2 else f'{i}rd Lactation' if i==3 else '4rd Lactation'
        df = xl.parse(sheet_name)
        all_dfs.append(df)
        cows_per_lac.append({
            'name': f'Lactation {i}',
            'count': len(df['Animal Number'].dropna().unique())
        })
    analytics['attrition_funnel'] = cows_per_lac
    
    # 3. Peak Lactation Curve & Biological Radar
    print("Building Lifecycle Metrics...")
    lifecycle_data = []
    for idx, df in enumerate(all_dfs):
        lifecycle_data.append({
            'name': f'Lactation {idx+1}',
            'Days_to_Peak': round(float(df['Days To Peak (Days)'].mean()), 1) if 'Days To Peak (Days)' in df.columns else 0,
            'Peak_Yield': round(float(df['Peak Yield (Kg)'].mean()), 1) if 'Peak Yield (Kg)' in df.columns else 0,
            'Dry_Days': round(float(df['Days Dry (Days)'].mean()), 1) if 'Days Dry (Days)' in df.columns else 0
        })
    analytics['lifecycle'] = lifecycle_data
    
    # 4. Herd Yield Distribution (Histogram of Total Yield)
    print("Building Distribution...")
    combined_yields = pd.concat([d[['Total Milk Yield (Kg)']].dropna() for d in all_dfs])
    # Create simple bins for Recharts BarChart rendering a bell curve
    bins = np.linspace(3000, 13000, 20)
    hist, edges = np.histogram(combined_yields['Total Milk Yield (Kg)'], bins=bins)
    
    dist_data = []
    for i in range(len(hist)):
        dist_data.append({
            'range': f"{int(edges[i])}-{int(edges[i+1])}",
            'count': int(hist[i])
        })
    analytics['distribution'] = dist_data
    
    # 5. Predictive Cohort Analysis (L1-L5) & Confidence Bands
    print("Building Predictive Cohorts...")
    def get_ids(df):
        if 'Animal Number' not in df.columns: return set()
        ids = df['Animal Number'].dropna().astype(str).str.strip()
        ids = ids.apply(lambda x: x[:-2] if x.endswith('.0') else x)
        return set(ids)

    s1_ids = get_ids(all_dfs[0])
    s2_ids = get_ids(all_dfs[1])
    s3_ids = get_ids(all_dfs[2])
    s4_ids = get_ids(all_dfs[3])

    # True Cohorts
    c2 = (s1_ids & s2_ids) - s3_ids  # Made it to L2 but not L3
    c3 = (s1_ids & s2_ids & s3_ids) - s4_ids # Made it to L3 but not L4
    c4 = s1_ids & s2_ids & s3_ids & s4_ids # Elite

    survival_rates = {
        'Total_Unique': len(s1_ids | s2_ids | s3_ids | s4_ids),
        'L1_Count': len(s1_ids),
        'L2_Count': len(s1_ids & s2_ids),
        'L2_Pct': round(len(s1_ids & s2_ids) / len(s1_ids) * 100, 1) if len(s1_ids) else 0,
        'L3_Count': len(s1_ids & s2_ids & s3_ids),
        'L3_Pct': round(len(s1_ids & s2_ids & s3_ids) / len(s1_ids) * 100, 1) if len(s1_ids) else 0,
        'L4_Count': len(c4),
        'L4_Pct': round(len(c4) / len(s1_ids) * 100, 1) if len(s1_ids) else 0
    }

    def get_yield(df, cohort):
        df['clean_id'] = df['Animal Number'].dropna().astype(str).str.strip().apply(lambda x: x[:-2] if x.endswith('.0') else x)
        val = df[df['clean_id'].isin(cohort)]['Total Milk Yield (Kg)'].mean()
        return round(float(val), 1) if pd.notnull(val) else None
    
    c2_y1, c2_y2 = get_yield(all_dfs[0], c2), get_yield(all_dfs[1], c2)
    c3_y1, c3_y2, c3_y3 = get_yield(all_dfs[0], c3), get_yield(all_dfs[1], c3), get_yield(all_dfs[2], c3)
    c4_y1, c4_y2, c4_y3, c4_y4 = get_yield(all_dfs[0], c4), get_yield(all_dfs[1], c4), get_yield(all_dfs[2], c4), get_yield(all_dfs[3], c4)

    # ML Prediction for C4 (Elite)
    x = [1, 2, 3, 4]
    y = [c4_y1, c4_y2, c4_y3, c4_y4]
    poly = np.polyfit(x, y, 2)
    l5_pred = round(float(np.polyval(poly, 5)), 1)
    
    # Std Dev for Confidence Band
    residuals = y - np.polyval(poly, x)
    std_dev = max(100, np.std(residuals) * 2) # At least 100kg variance minimum
    
    chart_data = [
        {'name': 'Lactation 1', 'Cohort_2': c2_y1, 'Cohort_3': c3_y1, 'Elite_75': c4_y1},
        {'name': 'Lactation 2', 'Cohort_2': c2_y2, 'Cohort_3': c3_y2, 'Elite_75': c4_y2},
        {'name': 'Lactation 3', 'Cohort_3': c3_y3, 'Elite_75': c4_y3},
        {'name': 'Lactation 4', 'Elite_75': c4_y4, 'L5_Prediction': c4_y4, 'L5_Bounds': [c4_y4, c4_y4]},
        {'name': 'Lactation 5 (AI)', 'L5_Prediction': l5_pred, 'L5_Bounds': [round(l5_pred - std_dev, 1), round(l5_pred + std_dev, 1)]}
    ]
    
    analytics['cohorts'] = {
        'survival': survival_rates,
        'graph': chart_data
    }

    # 6. Dry Period Optimization (Scatter Plot)
    print("Building Dry Period Analytics...")
    l2_df = all_dfs[1].copy()
    if 'Days Dry (Days)' in l2_df.columns and 'Total Milk Yield (Kg)' in l2_df.columns:
        dp_df = l2_df[['Days Dry (Days)', 'Total Milk Yield (Kg)']].dropna()
        # Filter realistic dry days (e.g. 20 to 150)
        dp_df = dp_df[(dp_df['Days Dry (Days)'] > 20) & (dp_df['Days Dry (Days)'] < 150)]
        dp_df = dp_df.sample(n=min(300, len(dp_df)), random_state=42)
        dp_df['Days_Dry'] = dp_df['Days Dry (Days)'].astype(int)
        dp_df['Yield'] = dp_df['Total Milk Yield (Kg)'].round(1)
        analytics['dry_period'] = dp_df[['Days_Dry', 'Yield']].to_dict(orient='records')
    else:
        analytics['dry_period'] = []

    # 7. Lactation Persistency (Scatter Plot)
    print("Building Persistency Analytics...")
    l1_df = all_dfs[0].copy()
    if 'Peak Yield (Kg)' in l1_df.columns and 'Total Milk Yield (Kg)' in l1_df.columns:
        pers_df = l1_df[['Peak Yield (Kg)', 'Total Milk Yield (Kg)']].dropna()
        pers_df = pers_df.sample(n=min(300, len(pers_df)), random_state=42)
        pers_df['Peak'] = pers_df['Peak Yield (Kg)'].round(1)
        pers_df['Total'] = pers_df['Total Milk Yield (Kg)'].round(1)
        analytics['persistency'] = pers_df[['Peak', 'Total']].to_dict(orient='records')
    else:
        analytics['persistency'] = []

    # Save to JSON
    os.makedirs('data', exist_ok=True)
    with open('data/sires_analytics.json', 'w') as f:
        json.dump(analytics, f, indent=4, default=convert_np)
        
    print("Successfully built data/sires_analytics.json!")

if __name__ == "__main__":
    generate_sires_analytics()
