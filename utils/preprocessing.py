import os
import glob
import pandas as pd
import hssm

def load_and_clean_mid_data(data_dir="../data", output_filename="mid_data_cleaned_hssm.csv"):
    """
    Reads a directory of MID task CSV files, gets subject, cue_type, RT, out_type columns. Converts the 
    'out_type' column into a binary HSSM 'response', cleans the data, and saves the output in new csv in same path. 
    """
    # 1. Check if the clean data already exists
    output_path = os.path.join(data_dir, output_filename)
    if os.path.exists(output_path):
        print(f"Cleaned dataset found at '{output_path}'.")
        print("Skipping preprocessing and loading directly...")
        return pd.read_csv(output_path)
        
    print(f"No cleaned dataset found. Searching for raw CSV files in '{data_dir}'...")
    
    # 2. Find all raw CSV files (excluding our output file)
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    csv_files = [f for f in csv_files if output_filename not in f] 
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the directory: {data_dir}")
        
    print(f"Found {len(csv_files)} files. Combining and cleaning...")
    
    # 3. Read and combine all CSVs
    df_list = [pd.read_csv(file) for file in csv_files]
    df_raw = pd.concat(df_list, ignore_index=True)
    
    # 4. Extract columns, then check for missing columns
    columns_to_keep = ['subject', 'cue_type', 'out_type', 'RT']
    
    missing_cols = [col for col in columns_to_keep if col not in df_raw.columns]
    if missing_cols:
        raise ValueError(f"Data is missing the following required columns: {missing_cols}")
        
    df_clean = df_raw[columns_to_keep].copy()
    
    # 5. Bring RT from ms -> seconds
    df_clean['rt'] = df_clean['RT'] / 1000.0  
    df_clean = df_clean.drop(columns=['RT']) 
    
    # 6. Map valid responses to a uniform boundary
    initial_row_count = len(df_clean)
    
    # Drop the 'TooFast' outliers (anticipatory presses)
    df_clean = df_clean[df_clean['out_type'] != 'TooFast']
    
    # Since this is a simple reaction time task (press button), all valid trials 
    # represent a single decision direction. We set all remaining responses to 1.
    df_clean['response'] = 1
    
    # Drop the old text outcome column since HSSM only wants 'response', 'rt', and covariates
    df_clean = df_clean.drop(columns=['out_type'])
    
    # 7. Final cleanup of missing/invalid trials
    df_clean = df_clean.dropna(subset=['rt', 'response'])
    # Keep the RT > 0.15s filter just in case there are other impossibly fast responses
    df_clean = df_clean[df_clean['rt'] > 0.15] 
    df_clean = df_clean.reset_index(drop=True)
    
    final_row_count = len(df_clean)
    
    # 8. Save the cleaned data
    df_clean.to_csv(output_path, index=False)
    
    print(f"\nData cleaning complete")
    print(f"Total trials loaded: {initial_row_count}")
    print(f"Trials dropped (TooFast/missing/invalid): {initial_row_count - final_row_count}")
    print(f"Final usable trials: {final_row_count}")
    print(f"SUCCESS: Cleaned data saved to '{output_path}'")
    
    return df_clean