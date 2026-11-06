import os
import re
import glob
import pandas as pd

# This file reads all the individual csv files for each subject and extracts the columns we want,
# congregating them all in a new csv with subject name as a column.
# It also joins per-trial data with patient-level metadata (treatment arm, responder status)
# from mid_patient_data_summary.csv, keyed on subject ID and session number.


def load_patient_metadata(data_dir="../data", summary_filename="mid_patient_data_summary.csv",
                          require_response=True):
    """
    Loads mid_patient_data_summary.csv and reshapes it from wide format (one row per patient,
    separate Phase 1/Phase 2 columns) into long format (one row per patient x session),
    making it easy to join onto trial-level data.

    Returns a DataFrame with columns:
        subject       - patient ID (matches trial CSVs)
        session       - integer session number (1 or 2)
        treatment     - 1 if Bupropion XL, 0 if Placebo
        is_responder  - 1 if Responder, 0 if Non-Responder (NaN if unknown and
                        require_response=False)

    By default a (patient, phase) is skipped when its Randomization or Response is missing.
    With require_response=False only a missing Randomization skips it, so subjects whose
    treatment arm is known but whose response status is not are kept.
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
            if pd.isna(row.get(rand_col)) or (require_response and pd.isna(row.get(resp_col))):
                continue

            treatment_val = 1 if row[rand_col].strip() == "Bupropion XL" else 0
            if pd.isna(row.get(resp_col)):
                response_val = float("nan")
            else:
                response_val = 1 if row[resp_col].strip() == "Responder" else 0

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
        if output_filename not in f
        and "mid_patient_data_summary" not in f
        and "cleaned_rl_model" not in f
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
        and "cleaned_rl_model" not in f
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


def prepare_session1_controls_rl(data_dir="../data", output_filename="cleaned_rl_model_controls.csv"):
    """
    Prepares a balanced-panel dataset for the RL-DDM (hssm.RLSSM) model in models/rl_ddm.py.

    Filters:
      - Session 1 only (Phase 1 visit)
      - Controls only  (treatment == 0, i.e. Placebo arm)

    RLSSM needs every subject to have the same number of trials, contiguous and in
    trial order, so NO trials are dropped. Instead, invalid trials (TooFast, NoResponse /
    missing RT, rt < 0.20 s) are kept in the learning sequence and masked out of the
    likelihood by setting response = -1 (and a placeholder rt). This task has only one
    possible response, so -1 never occurs naturally; the masked decision process in
    rl_ddm.py contributes logp = 0 for those rows.

    Adds:
      - cue_idx:   0 neutral, 1 small, 2 medium, 3 large (index into the learned Q table)
      - hit:       1 if the trial was rewarded/neutral feedback (out_type names the payout)
      - feedback:  hit * cue_value / 5, i.e. reward received scaled to [0, 1]
      - valid:     1 if the trial enters the likelihood, 0 if masked
      - trial_idx: 1..N trial position within subject (across all 4 blocks)
    """
    output_path = os.path.join(data_dir, output_filename)
    print(f"Searching for raw block CSV files in '{data_dir}'...")

    # 1. Only the raw per-block files (never derived outputs)
    csv_files = glob.glob(os.path.join(data_dir, "Placebo_MID_Reward_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No raw block CSV files found in the directory: {data_dir}")

    print(f"Found {len(csv_files)} files. Combining...")
    df_raw = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    required = ['subject', 'session', 'run', 'trial', 'cue_type', 'out_type', 'RT']
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Data is missing required columns: {missing}")
    df = df_raw[required].copy()

    # 2. Session 1, then join metadata and keep controls (Placebo, treatment=0)
    df['session'] = pd.to_numeric(df['session'], errors='coerce')
    df = df[df['session'] == 1].copy()
    print(f"After Session 1 filter: {len(df)} trials.")

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

    return _finalize_rl_panel(df, output_path, "Session 1 Controls (RL)")


def _finalize_rl_panel(df, output_path, title):
    """
    Shared final steps for the RL-DDM datasets: trial order, learning covariates, masking of
    invalid trials, balanced-panel check, save. `df` holds one session's raw trials for the
    chosen subjects (columns subject, session, run, trial, cue_type, out_type, RT, plus
    any metadata columns).
    """
    # 3. Sort into true trial order. Each raw CSV is one block (run = reward_1..reward_4),
    #    and the learned values carry over across blocks within the session.
    df = df.sort_values(['subject', 'run', 'trial'], kind='stable').reset_index(drop=True)
    if df.duplicated(['subject', 'run', 'trial']).any():
        raise ValueError("Duplicate (subject, run, trial) rows found.")

    # 4. Learning covariates, computed on the full sequence
    cue_mapping = {
        'neutral':       0.0,
        'small_reward':  0.5,
        'medium_reward': 1.0,
        'large_reward':  5.0
    }
    cue_index = {'neutral': 0, 'small_reward': 1, 'medium_reward': 2, 'large_reward': 3}
    df['cue_value'] = df['cue_type'].map(cue_mapping)
    df['cue_idx'] = df['cue_type'].map(cue_index)
    if df['cue_idx'].isna().any():
        raise ValueError(f"Unknown cue_type values: {df.loc[df['cue_idx'].isna(), 'cue_type'].unique()}")
    df['cue_idx'] = df['cue_idx'].astype(int)

    hit_types = {'NoIncentive', 'Quarters', 'Dollar', 'FiveDollars'}
    df['hit'] = df['out_type'].astype(str).str.strip().isin(hit_types).astype(int)
    df['feedback'] = df['hit'] * df['cue_value'] / 5.0
    df['trial_idx'] = df.groupby('subject').cumcount() + 1

    # 5. Mask invalid trials instead of dropping them. The RT floor is 0.20 s here (the non-RL
    #    datasets use 0.15 s): the DDM cannot produce an RT below a subject's non-decision
    #    time, so one anticipatory response just above 0.15 s walls off that subject's t and
    #    makes the hierarchical fit diverge. No controls RT falls in 0.15-0.20 s.
    rt = df['RT'] / 1000.0
    invalid = (df['out_type'] == 'TooFast') | rt.isna() | (rt < 0.20)
    df['valid'] = (~invalid).astype(int)
    df['response'] = df['valid'].map({1: 1, 0: -1})
    df['rt'] = rt.where(df['valid'] == 1, 1.0)   # placeholder rt for masked rows
    df = df.drop(columns=['RT'])

    # 6. Balanced-panel check
    counts = df.groupby('subject').size()
    if counts.nunique() != 1:
        raise ValueError(f"Unbalanced panel: trial counts per subject = {counts.value_counts().to_dict()}")
    masked = (1 - df['valid']).groupby(df['subject']).sum()

    # 7. Save
    df.to_csv(output_path, index=False)

    print(f"\n{title} preprocessing complete")
    print(f"  Subjects:            {df['subject'].nunique()}")
    print(f"  Trials per subject:  {counts.iloc[0]}")
    print(f"  Masked trials:       {int(masked.sum())} total "
          f"(per subject min {int(masked.min())}, max {int(masked.max())})")
    print(f"  Columns: {list(df.columns)}")
    print(f"SUCCESS: Saved to '{output_path}'")

    return df


# Raw block filenames: Placebo_MID_Reward_<ID>_session<S>_block<B>_<YYYY_Mon_DD>_<HHMM>.csv
RAW_BLOCK_PATTERN = re.compile(
    r"Placebo_MID_Reward_(?P<id>.+?)_session(?P<session>[\d.]+)_block(?P<block>\d)_"
    r"(?P<date>\d{4}_[A-Za-z]{3}_\d{2})_\d{4}\.csv$"
)


def normalize_subject_id(raw_id):
    """
    Maps the raw IDs found in filenames onto the metadata IDs (e.g. 'PLAABC123'). Some raw
    IDs carry a session suffix ('_2', '_02', '.2') or lack the 'PLA' prefix ('ABC123').
    """
    sid = re.split(r"[_.]", str(raw_id).strip().upper())[0]
    if re.fullmatch(r"[A-Z]{3}\d{3}", sid):
        sid = "PLA" + sid
    return sid


def load_raw_blocks(data_dir="../data"):
    """
    Reads every raw block file, taking subject, session and block from the FILENAME rather
    than from the file's own columns, which are unreliable in places:
      - subject IDs inside files are sometimes malformed ('_2' suffixes, bare numbers)
      - session labels include 1.5, 2.5 and 20, and 4 subjects' second visits are
        labelled session 1
      - one file is tab-delimited
    Session is assigned by visit order: a subject with two visit dates gets session 1 for
    the earlier and 2 for the later (sessions are ~3 weeks apart); a subject with one visit
    keeps the filename's label (1.5 -> 1, 2.5 -> 2).

    Returns one row per trial with columns subject, session, run, trial, cue_type,
    out_type, RT.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "Placebo_MID_Reward_*.csv")))
    if not files:
        raise FileNotFoundError(f"No raw block CSV files found in the directory: {data_dir}")

    frames = []
    for f in files:
        m = RAW_BLOCK_PATTERN.search(os.path.basename(f))
        if m is None:
            raise ValueError(f"Unexpected raw filename: {os.path.basename(f)}")
        block = pd.read_csv(f, sep=None, engine="python")  # sniffs comma vs tab
        block = block[['trial', 'cue_type', 'out_type', 'RT']].copy()
        block['subject'] = normalize_subject_id(m['id'])
        block['session_label'] = float(m['session'])
        block['run'] = f"reward_{m['block']}"
        block['date'] = pd.to_datetime(m['date'], format="%Y_%b_%d")
        frames.append(block)
    df = pd.concat(frames, ignore_index=True)

    n_dates = df.groupby('subject')['date'].transform('nunique')
    if n_dates.max() > 2:
        raise ValueError("A subject has more than two visit dates; session assignment is ambiguous.")
    visit = df.groupby('subject')['date'].rank(method='dense')
    df['session'] = visit.where(n_dates == 2, df['session_label'] // 1).astype(int)

    per_visit = df.groupby(['subject', 'session'])['run'].nunique()
    if (per_visit != 4).any():
        print(f"  WARNING: {int((per_visit != 4).sum())} (subject, session) pairs do not have 4 blocks.")
    print(f"Loaded {len(files)} raw block files: {df['subject'].nunique()} subjects, "
          f"{len(per_visit)} (subject, session) visits.")
    return df[['subject', 'session', 'run', 'trial', 'cue_type', 'out_type', 'RT']]


def prepare_rl_dataset(session, treatment, output_filename, data_dir="../data"):
    """
    Balanced-panel RL-DDM dataset for one session and one treatment arm
    (treatment 0 = Placebo, 1 = Bupropion XL), with the same columns, trial order and
    masking as prepare_session1_controls_rl. Raw files are read with load_raw_blocks
    (IDs and sessions from the filenames). Subjects are kept when their arm for that
    phase is known, even if their responder status is missing (is_responder = NaN).
    """
    arm = {0: "Placebo", 1: "Bupropion XL"}[treatment]
    output_path = os.path.join(data_dir, output_filename)

    df = load_raw_blocks(data_dir)
    df = df[df['session'] == session].copy()
    print(f"Session {session}: {len(df)} trials from {df['subject'].nunique()} subjects.")

    meta_long = load_patient_metadata(data_dir=data_dir, require_response=False)
    df = df.merge(meta_long, on=['subject', 'session'], how='left')
    no_meta = df.loc[df['treatment'].isna(), 'subject'].nunique()
    if no_meta:
        print(f"  WARNING: {no_meta} subjects have no Phase {session} randomization and are dropped.")
    df = df[df['treatment'] == treatment].copy()
    print(f"After {arm} filter: {len(df)} trials from {df['subject'].nunique()} subjects "
          f"({df.groupby('subject')['is_responder'].first().isna().sum()} with unknown responder status).")
    if df.empty:
        raise ValueError(f"No Session {session} {arm} subjects found.")

    return _finalize_rl_panel(df, output_path, f"Session {session} {arm} (RL)")


# --- Execution Block ---
if __name__ == "__main__":
    # Run the full dataset preprocessing
    df_all = load_and_clean_mid_data()

    # Also run the simplified Session 1 controls preprocessing for success_pooled.py
    df_s1 = prepare_session1_controls()

    # Balanced-panel Session 1 controls dataset for the RL-DDM (rl_ddm.py)
    df_rl = prepare_session1_controls_rl()

    # Same RL-DDM dataset for Session 2 Bupropion (rl_ddm_session2_bupropion.py)
    df_rl_s2_bupropion = prepare_rl_dataset(session=2, treatment=1,
                                            output_filename="cleaned_rl_model_session2_bupropion.csv")

    # Not in use: Session 2 Placebo (rl_ddm_session2_placebo.py, 6 subjects) and
    # Session 1 Bupropion (rl_ddm_session1_bupropion.py, 9 subjects)
    # df_rl_s2_placebo = prepare_rl_dataset(session=2, treatment=0,
    #                                       output_filename="cleaned_rl_model_session2_placebo.csv")
    # df_rl_s1_bupropion = prepare_rl_dataset(session=1, treatment=1,
    #                                         output_filename="cleaned_rl_model_session1_bupropion.csv")