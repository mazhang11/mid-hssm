import os
import re
import glob
import pandas as pd
import hssm

# This file reads all the individual csv files for each subject and extracts the columns we want,
# congregating them all in a new csv with subject name as a column.
# It also joins per-trial data with patient-level metadata (treatment arm, responder status)
# from mid_patient_data_summary.csv, keyed on subject ID and session number.


def load_patient_metadata(data_dir="../data", summary_filename="mid_patient_data_summary.csv"):
    """
    Loads mid_patient_data_summary.csv and reshapes it from wide format (one row per patient,
    separate Phase 1/Phase 2 columns) into long format (one row per patient x session),
    making it easy to join onto trial-level data.

    Returns a DataFrame with columns:
        subject       - patient ID (matches trial CSVs)
        session       - integer session number (1 or 2)
        treatment     - 1 if Bupropion XL, 0 if Placebo
        is_responder  - 1 if Responder, 0 if Non-Responder
    """
    summary_path = os.path.join(data_dir, summary_filename)
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Patient summary CSV not found at '{summary_path}'. "
            "Make sure 'mid_patient_data_summary.csv' is in the data directory."
        )

    meta = pd.read_csv(summary_path)
    print(f"Loaded patient metadata for {len(meta)} subjects from '{summary_path}'.")

    # Build one record per (subject, session) by iterating over both phases.
    # Phase number maps directly to session number (Phase 1 = Session 1, Phase 2 = Session 2).
    records = []
    for _, row in meta.iterrows():
        for phase_num, rand_col, resp_col in [
            (1, "Phase 1 Randomization", "Phase 1 Response"),
            (2, "Phase 2 Randomization", "Phase 2 Response"),
        ]:
            # Skip if either column is missing/NaN for this phase
            if pd.isna(row.get(rand_col)) or pd.isna(row.get(resp_col)):
                continue

            treatment_val = 1 if row[rand_col].strip() == "Bupropion XL" else 0
            response_val  = 1 if row[resp_col].strip()  == "Responder"    else 0

            records.append({
                "subject":      row["ID"],
                "session":      phase_num,
                "treatment":    treatment_val,   # 1=Bupropion XL, 0=Placebo
                "is_responder": response_val,    # 1=Responder, 0=Non-Responder
            })

    meta_long = pd.DataFrame(records)
    print(f"Reshaped metadata: {len(meta_long)} (subject, session) pairs.")
    return meta_long


def load_and_clean_mid_data(data_dir="../data", output_filename="mid_data_cleaned_hssm.csv"):
    """
    Reads a directory of MID task CSV files, gets subject, session, cue_type, RT, out_type columns.
    Converts the 'out_type' column into a binary HSSM 'response', cleans the data,
    joins patient-level treatment/response metadata, and saves the output in a new CSV.
    """
    # 1. Define output path (always re-run preprocessing)
    output_path = os.path.join(data_dir, output_filename)
    print(f"Searching for raw CSV files in '{data_dir}'...")

    # 2. Find all raw CSV files (excluding our output file and the summary file)
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    csv_files = [
        f for f in csv_files
        if output_filename not in f and "mid_patient_data_summary" not in f
    ]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the directory: {data_dir}")

    print(f"Found {len(csv_files)} files. Combining and cleaning...")

    # 3. Read all CSVs
    df_list = []
    for file in csv_files:
        df_sub = pd.read_csv(file)
        df_list.append(df_sub)

    df_raw = pd.concat(df_list, ignore_index=True)

    # 4. Extract columns; session is now included
    columns_to_keep = ['subject', 'session', 'cue_type', 'out_type', 'RT']

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
    df_clean = df_clean[df_clean['rt'] > 0.15]
    df_clean = df_clean.reset_index(drop=True)

    final_row_count = len(df_clean)

    # 8. Map categorical cue strings to continuous numerical values
    cue_mapping = {
        'neutral':       0.0,
        'small_reward':  0.5,
        'medium_reward': 1.0,
        'large_reward':  5.0
    }
    df_clean['cue_value'] = df_clean['cue_type'].map(cue_mapping)
    print("Successfully mapped continuous covariates.")

    # 9. Join patient-level metadata (treatment arm + responder status)
    meta_long = load_patient_metadata(data_dir=data_dir)

    # Ensure join keys are the same dtype before merging
    df_clean['session'] = pd.to_numeric(df_clean['session'], errors='coerce')
    meta_long['subject'] = meta_long['subject'].astype(df_clean['subject'].dtype)

    pre_join_count = len(df_clean)
    df_clean = df_clean.merge(meta_long, on=['subject', 'session'], how='left')

    unmatched = df_clean['treatment'].isna().sum()
    if unmatched > 0:
        print(f"  WARNING: {unmatched} trials could not be matched to patient metadata "
              "(subject ID or session not found in summary CSV). "
              "These rows will have NaN for 'treatment' and 'is_responder'.")

    print(f"Joined patient metadata. "
          f"Matched {pre_join_count - unmatched}/{pre_join_count} trials.")

    # 10. Save the cleaned data
    df_clean.to_csv(output_path, index=False)

    print(f"\nData cleaning complete")
    print(f"Total trials loaded:                      {initial_row_count}")
    print(f"Trials dropped (TooFast/missing/invalid): {initial_row_count - final_row_count}")
    print(f"Final usable trials:                      {final_row_count}")
    print(f"Columns in output: {list(df_clean.columns)}")
    print(f"SUCCESS: Cleaned data saved to '{output_path}'")

    return df_clean


def prepare_session1_controls(data_dir="../data", output_filename="mid_session1_controls_hssm.csv"):
    """
    Prepares a filtered, analysis-ready dataset for the simplified success_pooled model.

    Filters:
      - Session 1 only (Phase 1 visit)
      - Controls only  (treatment == 0, i.e. Placebo arm)

    Adds:
      - last_trial_success: 1 if the immediately preceding trial was a hit, 0 if a miss/skip.
        Computed BEFORE out_type is dropped so we capture the actual trial outcome.
        Defined as: out_type not in the failure set {'Miss', 'TooSlow', 'TooFast'}.
        The first trial for each subject has last_trial_success = 0 (no prior trial).

    z (starting bias) is intentionally NOT included as a covariate here — it will be fixed
    at 0.5 in the HSSM model formulation.
    """
    output_path = os.path.join(data_dir, output_filename)
    print(f"Searching for raw CSV files in '{data_dir}'...")

    # 1. Find all trial CSVs (exclude outputs and summary file)
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    csv_files = [
        f for f in csv_files
        if output_filename not in f
        and "mid_patient_data_summary" not in f
        and "mid_data_cleaned_hssm" not in f
    ]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the directory: {data_dir}")

    print(f"Found {len(csv_files)} files. Combining...")
    df_raw = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # 2. Validate required columns
    required = ['subject', 'session', 'cue_type', 'out_type', 'RT']
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Data is missing required columns: {missing}")

    df = df_raw[required].copy()

    # 3. Filter to Session 1 BEFORE joining metadata (smaller, faster merge)
    df['session'] = pd.to_numeric(df['session'], errors='coerce')
    df = df[df['session'] == 1].copy()
    print(f"After Session 1 filter: {len(df)} trials.")

    # 4. Join patient metadata and filter to controls (Placebo, treatment=0)
    meta_long = load_patient_metadata(data_dir=data_dir)
    meta_long['subject'] = meta_long['subject'].astype(df['subject'].dtype)
    df = df.merge(meta_long, on=['subject', 'session'], how='left')

    unmatched = df['treatment'].isna().sum()
    if unmatched > 0:
        print(f"  WARNING: {unmatched} trials had no metadata match and will be dropped.")
    df = df.dropna(subset=['treatment'])
    df = df[df['treatment'] == 0].copy()
    print(f"After controls-only filter (Placebo): {len(df)} trials "
          f"from {df['subject'].nunique()} subjects.")

    # 5. Compute last_trial_success BEFORE dropping out_type.
    #    Success = trial was a hit (not a miss or anticipatory press).
    #    Covers common MID task naming conventions for failure outcomes.
    failure_types = {'Miss', 'TooSlow', 'TooFast', 'miss', 'tooslow', 'toofast'}
    df = df.reset_index(drop=True)
    df['trial_success'] = df['out_type'].apply(
        lambda x: 0 if str(x).strip() in failure_types else 1
    )

    # Sort by subject to ensure within-subject trial order is preserved, then shift
    df = df.sort_values(['subject']).reset_index(drop=True)
    df['last_trial_success'] = (
        df.groupby('subject')['trial_success']
          .shift(1)
          .fillna(0)   # first trial of each subject has no prior trial → 0
          .astype(int)
    )
    df = df.drop(columns=['trial_success'])

    # 6. Drop TooFast outliers (anticipatory presses)
    initial_count = len(df)
    df = df[df['out_type'] != 'TooFast'].copy()

    # 7. Convert RT to seconds and set uniform response = 1
    df['rt'] = df['RT'] / 1000.0
    df = df.drop(columns=['RT', 'out_type'])
    df['response'] = 1

    # 8. Final RT validity filter
    df = df.dropna(subset=['rt', 'response'])
    df = df[df['rt'] > 0.15].reset_index(drop=True)
    final_count = len(df)

    # 9. Map cue conditions to continuous numerical values
    cue_mapping = {
        'neutral':       0.0,
        'small_reward':  0.5,
        'medium_reward': 1.0,
        'large_reward':  5.0
    }
    df['cue_value'] = df['cue_type'].map(cue_mapping)

    # 10. Save
    df.to_csv(output_path, index=False)

    print(f"\nSession 1 Controls preprocessing complete")
    print(f"  Trials before RT/TooFast filter: {initial_count}")
    print(f"  Trials dropped:                  {initial_count - final_count}")
    print(f"  Final usable trials:             {final_count}")
    print(f"  Subjects:                        {df['subject'].nunique()}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  last_trial_success distribution:\n"
          f"{df['last_trial_success'].value_counts().to_string()}")
    print(f"SUCCESS: Saved to '{output_path}'")

    return df


# --- Execution Block ---
if __name__ == "__main__":
    # Run the full dataset preprocessing
    df_all = load_and_clean_mid_data()

    # Also run the simplified Session 1 controls preprocessing for success_pooled.py
    df_s1 = prepare_session1_controls()