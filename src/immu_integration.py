import pandas as pd
import numpy as np
import os
import glob

# Paths
BASE_DIR = 'data/MmCows/MmCows Dairy Cows Dataset/sensor_data/sensor_data/main_data/immu'
DAILY_FEATURES_PATH = 'data/mmcows_daily_features.csv'
OUTPUT_PATH = 'data/mmcows_daily_features_v2.csv'

def process_immu():
    print("Processing IMMU Acceleration Data...")
    
    if not os.path.exists(DAILY_FEATURES_PATH):
        print("Error: Base daily features not found. Run sensor_integration.py first.")
        return
        
    master_df = pd.read_csv(DAILY_FEATURES_PATH)
    master_df['timestamp'] = pd.to_datetime(master_df['timestamp']).dt.normalize()
    
    immu_dirs = glob.glob(os.path.join(BASE_DIR, 'T*'))
    all_activity = []
    
    for dir_path in immu_dirs:
        # T01 typically maps to C01, etc.
        tag_id = os.path.basename(dir_path)
        cow_id = tag_id.replace('T', 'C') 
        
        csv_files = glob.glob(os.path.join(dir_path, '*.csv'))
        if not csv_files:
            continue
            
        print(f"Reading IMMU for {cow_id}...")
        df = pd.read_csv(csv_files[0])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate Vector Magnitude of Acceleration
        # VM = sqrt(x^2 + y^2 + z^2)
        df['activity_index'] = np.sqrt(df['accel_x_mps2']**2 + df['accel_y_mps2']**2 + df['accel_z_mps2']**2)
        
        df.set_index('timestamp', inplace=True)
        # Resample to daily average activity
        daily_act = df.resample('D').agg({'activity_index': 'sum'})
        daily_act.reset_index(inplace=True)
        daily_act['cow_id'] = cow_id
        
        # Lameness happens when activity drastically drops from average
        # We'll calculate a rolling mean if there are enough days, 
        # but for simplicity in this demo, say activity < 20th percentile is lethargy.
        threshold = daily_act['activity_index'].quantile(0.2)
        daily_act['lameness_lethargy_flag'] = (daily_act['activity_index'] < threshold).astype(int)
        
        all_activity.append(daily_act)
    
    if all_activity:
        activity_df = pd.concat(all_activity, ignore_index=True)
        activity_df['timestamp'] = activity_df['timestamp'].dt.normalize()
        
        print("Merging IMMU flags with master features...")
        final_df = pd.merge(master_df, activity_df, on=['timestamp', 'cow_id'], how='left')
        
        # Fill missing lameness flags with 0
        final_df['lameness_lethargy_flag'] = final_df['lameness_lethargy_flag'].fillna(0)
        final_df['activity_index'] = final_df['activity_index'].fillna(final_df['activity_index'].median())
        
        final_df.to_csv(OUTPUT_PATH, index=False)
        print(f"Version 2 ML Dataset saved to {OUTPUT_PATH}")
    else:
        print("No IMMU data found.")

if __name__ == "__main__":
    process_immu()
