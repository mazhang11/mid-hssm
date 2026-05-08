import sys
import os
import datetime
import pandas as pd
import hssm
import arviz as az
import matplotlib.pyplot as plt

# The prepare_continuous_covariates function has been moved to utils/preprocessing.py

def fit_baseline_model(df, subject_idx):
    """
    Fits a baseline DDM with no cue covariates. Used as the null model for az.compare.
    """
    all_subjects = df['subject'].unique()
    target_subject = all_subjects[subject_idx]
    subset_data = df[df['subject'] == target_subject].copy()
    
    if subject_idx % 50 == 0:
        print(f" Fitting Baseline Model (Subject: {target_subject})")
    
    baseline_model = hssm.HSSM(
        data=subset_data,
        model="ddm",
        include=[
            {"name": "v", "formula": "v ~ 1"},
            {"name": "a", "formula": "a ~ 1"},
            {"name": "t", "formula": "t ~ 1"},
            {"name": "z", "formula": "z ~ 1"}
        ]
    )
    show_progress = (subject_idx % 50 == 0)
    baseline_model.sample(
        tune=1000, draws=1000, chains=4, cores=4,
        sampler="nuts_numpyro", progressbar=show_progress
    )
    if subject_idx % 50 == 0:
        print(f"Finished sampling baseline model for subject {target_subject}.")
    return baseline_model

def fit_continuous_model(df, subject_idx):
    """
    Fits a DDM treating the incentive as a continuous linear predictor for v and z.
    """
    # --- MODIFIED FOR ARRAY JOB: Isolate exactly ONE subject based on the array ID ---
    all_subjects = df['subject'].unique()
    
    # Safety check: if the array ID is larger than our number of subjects, exit cleanly
    if subject_idx >= len(all_subjects):
        print(f"Task ID {subject_idx} is out of bounds (only {len(all_subjects)} subjects). Exiting.")
        sys.exit(0)
        
    target_subject = all_subjects[subject_idx]
    subset_data = df[df['subject'] == target_subject].copy()
    
    if subject_idx % 50 == 0:
        print("\n" + "="*50)
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
            {"name": "t", "formula": "t ~ 1"},
            {"name": "z", "formula": "z ~ 1 + cue_value"}  # z now responds to incentive
        ]
    )
    
    show_progress = (subject_idx % 50 == 0)
    continuous_model.sample(
        tune=1000, draws=1000, chains=4, cores=4,
        sampler="nuts_numpyro", progressbar=show_progress
    )
    
    if subject_idx % 50 == 0:
        print(f"Finished sampling continuous model for subject {target_subject}.")
    return continuous_model, target_subject

def plot_model_posteriors(model, model_name="Model", subject_id="", output_dir="plots"):
    """
    Uses ArviZ to plot marginal posteriors and pair plots 
    to visually inspect parameter estimates and tradeoffs.
    """
    if subject_id % 50 == 0 if isinstance(subject_id, int) else True:
        print(f"\nGenerating plots for {model_name} (Subject {subject_id})...")
    
    # --- MODIFIED: Routing to your designated output folder ---
    os.makedirs(output_dir, exist_ok=True)
    file_prefix = os.path.join(output_dir, f"Sub_{subject_id}_{model_name.replace(' ', '_')}")
    
    # 1. Plot the marginal posteriors for the global intercepts
    # This shows the confidence range for the group average of each parameter
    az.plot_posterior(
        model.traces, 
        var_names=['v_Intercept', 'v_cue_value', 'a_Intercept', 't_Intercept', 'z_Intercept', 'z_cue_value']
    )
    plt.suptitle(f"{model_name} (Sub {subject_id}): Marginal Posteriors")
    plt.savefig(f"{file_prefix}_marginals.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot pair plots to check for parameter tradeoffs
    # kind='kde' creates a contour plot. A strong diagonal stretch means high tradeoff.
    az.plot_pair(
        model.traces, 
        var_names=['v_Intercept', 'a_Intercept', 'z_Intercept'], 
        kind='kde', 
        marginals=True
    )
    plt.suptitle(f"{model_name} (Sub {subject_id}): Tradeoff Check (v, a, z)")
    plt.savefig(f"{file_prefix}_tradeoffs.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    if subject_id % 50 == 0 if isinstance(subject_id, int) else True:
        print(f"Saved plots for {model_name} (Subject {subject_id}) into {output_dir}/.")

if __name__ == "__main__":
    # --- MODIFIED FOR ARRAY JOB: Grab the array task ID passed from SLURM ---
    if len(sys.argv) > 1:
        subject_idx = int(sys.argv[1])
    else:
        subject_idx = 0 
        
    # 1. Load data (covariates are now pre-mapped in preprocessing)
    data_path = "../data/mid_data_cleaned_hssm.csv"
    if subject_idx % 50 == 0:
        print(f"Loading cleaned data from '{data_path}'...")
    df = pd.read_csv(data_path)
    
    # 2. Fit Baseline Model (null: intercepts only, no cue covariate)
    # --- DISABLED: Baseline training skipped to conserve compute. Re-enable for model comparison. ---
    # baseline_model = fit_baseline_model(df, subject_idx)
    
    # 3. Fit Continuous Model for this specific subject
    continuous_model, subject_id = fit_continuous_model(df, subject_idx)
    
    # --- SAVE MODEL INFO ---
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Create the output directory at the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    slurm_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    if slurm_job_id:
        # For array jobs, we group all 50 tasks into the same folder using the unique SLURM Job ID.
        # This prevents nasty race conditions where Task 5 might accidentally create a "v2" folder 
        # while Task 0 is writing to "v1" because they started a fraction of a second apart.
        output_dir = os.path.join(project_root, "outputs", f"continuous_unpooled_{date_str}_Job{slurm_job_id}")
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Local fallback uses simple v1, v2 enumeration
        base_dir = os.path.join(project_root, "outputs", f"continuous_unpooled_{date_str}")
        counter = 1
        while True:
            output_dir = f"{base_dir}_v{counter}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                break
            counter += 1
    
    # Create the models subfolder
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Sample posterior predictive (embeds simulated data into trace for PPC plots in evaluate_models.py)
    continuous_model.sample_posterior_predictive()
    
    # Save continuous trace (baseline disabled — re-enable when running model comparison)
    # az.to_netcdf(baseline_model.traces, os.path.join(models_dir, f"Sub_{subject_id}_baseline_model_trace.nc"))
    continuous_trace_path = os.path.join(models_dir, f"Sub_{subject_id}_continuous_model_trace.nc")
    az.to_netcdf(continuous_model.traces, continuous_trace_path)
    if subject_idx % 50 == 0:
        print(f"Saved trace to {models_dir}")
    
    # Save free_RVs once (same for all subjects since the model structure is identical)
    if subject_idx == 0:
        free_rvs_path = os.path.join(output_dir, "free_variables.txt")
        with open(free_rvs_path, "w") as f:
            # f.write("=== Baseline Model Free RVs ===\n")
            # for rv in baseline_model.pymc_model.free_RVs:
            #     f.write(f"  {rv}\n")
            f.write("=== Continuous Model Free RVs ===\n")
            for rv in continuous_model.pymc_model.free_RVs:
                f.write(f"  {rv}\n")
        print(f"Saved free variable list to {free_rvs_path}")
    
    # Save plots to a 'plots' subfolder
    plots_dir = os.path.join(output_dir, "plots")
    plot_model_posteriors(continuous_model, "Continuous Model", subject_id, output_dir=plots_dir)