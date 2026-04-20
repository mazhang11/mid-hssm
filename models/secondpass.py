import sys
import pandas as pd
import hssm
import arviz as az
import matplotlib.pyplot as plt

def prepare_continuous_covariates(data_path="../data/mid_data_cleaned_hssm.csv"):
    """
    Loads the cleaned data and maps the categorical cue types 
    to a continuous numerical scale for regression modeling.
    """
    print(f"Loading cleaned data from '{data_path}'...")
    df = pd.read_csv(data_path)
    
    # Map categorical cue strings to continuous numerical values
    cue_mapping = {
      'neutral': 0.0,
      'small_reward': 0.5,
      'medium_reward': 1.0,
      'large_reward': 5.0
    } 
    
    df['cue_value'] = df['cue_type'].map(cue_mapping)
    print("Successfully mapped continuous covariates.")
    
    return df

def fit_continuous_model(df, subject_idx):
    """
    Fits a DDM treating the incentive as a continuous linear predictor.
    Bias (z) is omitted, defaulting to a fixed 0.5.
    """
    # --- MODIFIED FOR ARRAY JOB: Isolate exactly ONE subject based on the array ID ---
    all_subjects = df['subject'].unique()
    
    # Safety check: if the array ID is larger than our number of subjects, exit cleanly
    if subject_idx >= len(all_subjects):
        print(f"Task ID {subject_idx} is out of bounds (only {len(all_subjects)} subjects). Exiting.")
        sys.exit(0)
        
    target_subject = all_subjects[subject_idx]
    subset_data = df[df['subject'] == target_subject].copy()
    
    print("\n" + "="*50)
    # --- MODIFIED FOR OSCAR: Updated print statement ---
    print(f" Fitting Continuous Regression Model (Single Subject: {target_subject})")
    print("="*50)
    
    # Initialize the continuous model
    # 'cue_value' is numerical, so the model calculates a single slope (beta weight)
    # for how much v increases per $1 increase in reward.
    # NOTE: Random effects like (1|subject) are removed because we are fitting one individual.
    continuous_model = hssm.HSSM(
        data=subset_data,
        model="ddm",
        include=[
            {"name": "v", "formula": "v ~ 1 + cue_value"}, 
            {"name": "a", "formula": "a ~ 1"},
            {"name": "t", "formula": "t ~ 1"}
            # z is omitted, so it defaults to fixed 0.5 and will not update
        ]
    )
    
    # --- MODIFIED FOR OSCAR: Increased chains and cores to 4 ---
    continuous_model.sample(tune=1000, draws=1000, chains=4, cores=4)
    print(f"Finished sampling continuous model for subject {target_subject}.")
    return continuous_model, target_subject

def plot_model_posteriors(model, model_name="Model", subject_id=""):
    """
    Uses ArviZ to plot marginal posteriors and pair plots 
    to visually inspect parameter estimates and tradeoffs.
    """
    print(f"\nGenerating plots for {model_name} (Subject {subject_id})...")
    
    # --- MODIFIED: Routing to your Git-tracked plots folder ---
    file_prefix = f"plots/Sub_{subject_id}_{model_name.replace(' ', '_')}"
    
    # 1. Plot the marginal posteriors for the global intercepts
    # This shows the confidence range for the group average of each parameter
    az.plot_posterior(
        model.traces, 
        var_names=['v_Intercept', 'a_Intercept', 't_Intercept']
    )
    plt.suptitle(f"{model_name} (Sub {subject_id}): Marginal Posteriors")
    
    # --- MODIFIED FOR OSCAR: Swapped plt.show() for plt.savefig() and plt.close() ---
    # plt.show()
    plt.savefig(f"{file_prefix}_marginals.png", dpi=300, bbox_inches='tight')
    plt.close() # Close the plot so it doesn't take up headless memory
    
    # 2. Plot pair plots to check for parameter tradeoffs
    # kind='kde' creates a contour plot. A strong diagonal stretch means high tradeoff.
    az.plot_pair(
        model.traces, 
        var_names=['v_Intercept', 'a_Intercept'], 
        kind='kde', 
        marginals=True
    )
    plt.suptitle(f"{model_name} (Sub {subject_id}): Tradeoff Check (v vs a)")
    
    # --- MODIFIED FOR OSCAR: Swapped plt.show() for plt.savefig() and plt.close() ---
    # plt.show()
    plt.savefig(f"{file_prefix}_tradeoffs.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plots for {model_name} (Subject {subject_id}) into models/plots/.")

if __name__ == "__main__":
    # --- MODIFIED FOR ARRAY JOB: Grab the array task ID passed from SLURM ---
    if len(sys.argv) > 1:
        subject_idx = int(sys.argv[1])
    else:
        # Fallback for testing locally without an array ID
        subject_idx = 0 
        
    # 1. Load data and create the continuous covariate column
    df = prepare_continuous_covariates()
    
    # 3. Fit Continuous Model for this specific subject
    continuous_model, subject_id = fit_continuous_model(df, subject_idx)
    plot_model_posteriors(continuous_model, "Continuous Model", subject_id)