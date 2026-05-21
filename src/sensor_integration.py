import pandas as pd
import numpy as np
import os
import glob

# Paths
BASE_DIR = 'data/MmCows/MmCows Dairy Cows Dataset/sensor_data/sensor_data/main_data'
CBT_DIR = os.path.join(BASE_DIR, 'cbt')
THI_DIR = os.path.join(BASE_DIR, 'thi')
MILK_DIR = os.path.join(BASE_DIR, 'milk')
OUTPUT_PATH = 'data/mmcows_daily_features.csv'

def process_cbt(cow_id):
    """Loads and aggregates Core Body Temperature (CBT) to daily metrics."""
    cbt_file = os.path.join(CBT_DIR, f'{cow_id}.csv')
    if not os.path.exists(cbt_file):
        return pd.DataFrame()
        
    df = pd.read_csv(cbt_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Set timestamp as index for fast resampling
    df.set_index('timestamp', inplace=True)
    
    # Resample continuous readings to Daily ('D')
    daily_cbt = df.resample('D').agg({
        'temperature_C': ['mean', 'max']
    })
    
    # Flatten multi-level columns
    daily_cbt.columns = ['avg_daily_temp_C', 'max_daily_temp_C']
    daily_cbt.reset_index(inplace=True)
    daily_cbt['cow_id'] = cow_id
    
    # Health Flag: Fever > 39.0 C
    daily_cbt['fever_flag'] = (daily_cbt['max_daily_temp_C'] > 39.0).astype(int)
    
    return daily_cbt

def process_milk(cow_id):
    """Loads daily milk yield."""
    milk_file = os.path.join(MILK_DIR, f'{cow_id}.csv')
    if not os.path.exists(milk_file):
        return pd.DataFrame()
        
    df = pd.read_csv(milk_file)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.normalize() # Ensure it's just the date
    df['cow_id'] = cow_id
    return df

def get_average_thi():
    """Loads the average barn THI."""
    thi_file = os.path.join(THI_DIR, 'average.csv')
    if not os.path.exists(thi_file):
        return pd.DataFrame()
        
    df = pd.read_csv(thi_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Resample continuous THI to daily
    daily_thi = df.resample('D').agg({
        'THI': 'mean',
        'temperature_F': 'mean'
    })
    
    daily_thi.columns = ['avg_daily_THI', 'avg_daily_barn_temp_F']
    daily_thi.reset_index(inplace=True)
    
    # Heat stress flag > 72 THI
    daily_thi['heat_stress_flag'] = (daily_thi['avg_daily_THI'] > 72.0).astype(int)
    return daily_thi

def integrate_data():
    print("Integrating MmCows Sensor Data...")
    
    all_cows_cbt = []
    all_cows_milk = []
    
    # We know cows are labeled C01 to C16 (from paper context), let's just find all files in milk dir
    cow_files = glob.glob(os.path.join(MILK_DIR, '*.csv'))
    cow_ids = [os.path.basename(f).replace('.csv', '') for f in cow_files]
    
    for cow_id in cow_ids:
        print(f"Processing {cow_id}...")
        cbt_df = process_cbt(cow_id)
        milk_df = process_milk(cow_id)
        
        # Merge CBT and Milk on timestamp
        if not cbt_df.empty and not milk_df.empty:
            merged_cow = pd.merge(milk_df, cbt_df, on=['timestamp', 'cow_id'], how='inner')
            all_cows_cbt.append(merged_cow)
            
    if not all_cows_cbt:
        print("Error: No data merged. Check file formats.")
        return
        
    master_df = pd.concat(all_cows_cbt, ignore_index=True)
    
    # Now merge the shared daily THI logically on timestamp
    thi_df = get_average_thi()
    if not thi_df.empty:
        # Normalize master_df timestamp to merge cleanly
        master_df['timestamp'] = master_df['timestamp'].dt.normalize()
        thi_df['timestamp'] = thi_df['timestamp'].dt.normalize()
        
        master_df = pd.merge(master_df, thi_df, on='timestamp', how='left')
    else:
        print("Warning: THI data not found.")
        
    # Drop rows where we have missing target (milk)
    master_df.dropna(subset=['milk_weight_kg'], inplace=True)
    
    print(f"Integration complete. Total daily records: {len(master_df)}")
    master_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    integrate_data()
