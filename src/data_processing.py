import pandas as pd
import numpy as np
import os

RAW_DATA_PATH = 'raw data for performance of superior dairy cattle sires_2.xlsx'
PROCESSED_DATA_PATH = 'data/processed_lactations.csv'

def clean_column_names(df):
    """Standardize column names for easier access."""
    cols = []
    for col in df.columns:
        # Lowercase, replace spaces with underscores, remove units and special chars
        c = str(col).lower().strip()
        c = c.replace(' (kg)', '').replace(' (days)', '').replace(',', '')
        c = c.replace('/', '_').replace(' ', '_')
        cols.append(c)
    df.columns = cols
    return df

def load_and_merge_lactations():
    """Loads lactations 1-4, cleans them, and merges vertically."""
    print("Loading data from Excel...")
    xl = pd.ExcelFile(RAW_DATA_PATH)
    
    all_lactations = []
    
    # Loop over the 4 lactation sheets
    for i in range(1, 5):
        sheet_name = f"{i}st Lactation" if i == 1 else f"{i}nd Lactation" if i == 2 else f"{i}rd Lactation" if i == 3 else f"{i}rd Lactation" 
        # Actually, user printed sheet names: ['1st Lactation', '2nd Lactation', '3rd Lactation', '4rd Lactation'] <- wait, '4rd Lactation'? 
        # I saw '4rd Lactation' in the output. I'll hardcode the exact names from the python check.
        sheet_names = ['1st Lactation', '2nd Lactation', '3rd Lactation', '4rd Lactation']
        
        df = xl.parse(sheet_names[i-1])
        df = clean_column_names(df)
        df['lactation_number'] = i
        all_lactations.append(df)
        print(f"Loaded {sheet_names[i-1]}: {len(df)} records")

    # Concatenate all lactations
    combined_df = pd.concat(all_lactations, ignore_index=True)
    
    # Convert numeric columns safely
    numeric_cols = ['total_milk_yield', 'corrected_milk', 'length_of_lactation', 
                    'days_dry', 'peak_yield', 'days_to_peak']
    
    for col in numeric_cols:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    
    print(f"Total Combined Records: {len(combined_df)}")
    return combined_df

def load_pedigree():
    xl = pd.ExcelFile(RAW_DATA_PATH)
    df = xl.parse('Pedigree')
    return clean_column_names(df)

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    
    combined_df = load_and_merge_lactations()
    combined_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved processed data to {PROCESSED_DATA_PATH}")
