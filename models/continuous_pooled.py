import sys
import os
import datetime
import pandas as pd
import hssm
import arviz as az
import matplotlib.pyplot as plt

def fit_baseline_pooled_model(df):
    """
    Fits a hierarchical baseline DDM with no cue covariates. Used as the null model for az.compare.
    """
    print(" Fitting Baseline Hierarchical (Pooled) Model")
    baseline_model = hssm.HSSM(
        data=df,
        model="ddm",
        include=[
            {"name": "v", "formula": "v ~ 1 + (1 | subject)"},
            {"name": "a", "formula": "a ~ 1 + (1 | subject)"},
            {"name": "t", "formula": "t ~ 1 + (1 | subject)"},
            {"name": "z", "formula": "z ~ 1 + (1 | subject)"}
        ]
    )
    baseline_model.sample(
        tune=1000, draws=1000, chains=4, cores=4,
        target_accept=0.9, sampler="nuts_numpyro"
    )
    print("Finished sampling baseline pooled model.")
    return baseline_model

def fit_pooled_model(df):
    print("\n" + "="*50)
    print(" Fitting Continuous Hierarchical (Pooled) Model")
    print("="*50)
    
    # We include (1|subject) to model random intercepts per subject
    # We include (0 + cue_value|subject) to allow the slope of the cue_value to vary per subject
    # This hierarchical structure allows for partial pooling across all subjects.
    pooled_model = hssm.HSSM(
        data=df,
        model="ddm",
        include=[
            {"name": "v", "formula": "v ~ 1 + cue_value + (1 + cue_value | subject)"}, 
            {"name": "a", "formula": "a ~ 1 + (1 | subject)"},
            {"name": "t", "formula": "t ~ 1 + (1 | subject)"},
            {"name": "z", "formula": "z ~ 1 + cue_value + (1 + cue_value | subject)"}  # z responds to incentive
        ]
    )
    
    # We use sampler="nuts_numpyro" to leverage JAX, which automatically detects and runs on the GPU!
    pooled_model.sample(
        tune=1000, 
        draws=1000, 
        chains=4, 
        cores=4, 
        target_accept=0.9,
        sampler="nuts_numpyro"
    )
    print("Finished sampling pooled model.")
    return pooled_model

def plot_model_posteriors(model, output_dir):
    print("\nGenerating plots for Pooled Model...")
    os.makedirs(output_dir, exist_ok=True)
    file_prefix = os.path.join(output_dir, "Pooled_Model")
    
    # Marginal posteriors for global fixed effects, now including z
    az.plot_posterior(
        model.traces, 
        var_names=['v_Intercept', 'v_cue_value', 'a_Intercept', 't_Intercept', 'z_Intercept', 'z_cue_value']
    )
    plt.suptitle("Pooled Model: Global Fixed Effects")
    plt.savefig(f"{file_prefix}_marginals.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Pair plot including z to check for v/z tradeoff
    az.plot_pair(
        model.traces,
        var_names=['v_Intercept', 'a_Intercept', 'z_Intercept'],
        kind='kde',
        marginals=True
    )
    plt.suptitle("Pooled Model: Tradeoff Check (v, a, z)")
    plt.savefig(f"{file_prefix}_tradeoffs.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plots into {output_dir}/")

if __name__ == "__main__":
    # 1. Load data
    data_path = "../data/mid_data_cleaned_hssm.csv"
    print(f"Loading cleaned data from '{data_path}'...")
    df = pd.read_csv(data_path)
    
    # 2. Fit Baseline Pooled Model (null: no cue covariate)
    # --- DISABLED: Baseline training skipped to conserve compute. Re-enable for model comparison. ---
    # baseline_model = fit_baseline_pooled_model(df)
    
    # 3. Fit Continuous Hierarchical Model
    pooled_model = fit_pooled_model(df)
    
    # --- SAVE MODEL INFO ---
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_dir = os.path.join(project_root, "outputs", f"continuous_pooled_{date_str}")
    
    # Enumerate folders so we don't overwrite identical runs on the same day
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
    pooled_model.sample_posterior_predictive()
    
    # Save pooled trace (baseline disabled — re-enable when running model comparison)
    # az.to_netcdf(baseline_model.traces, os.path.join(models_dir, "continuous_pooled_baseline_trace.nc"))
    pooled_trace_path = os.path.join(models_dir, "continuous_pooled_model_trace.nc")
    az.to_netcdf(pooled_model.traces, pooled_trace_path)
    print(f"Saved trace to {models_dir}")
    
    # Save free_RVs to the top-level output folder
    free_rvs_path = os.path.join(output_dir, "free_variables.txt")
    with open(free_rvs_path, "w") as f:
        # f.write("=== Baseline Pooled Model Free RVs ===\n")
        # for rv in baseline_model.pymc_model.free_RVs:
        #     f.write(f"  {rv}\n")
        f.write("=== Continuous Pooled Model Free RVs ===\n")
        for rv in pooled_model.pymc_model.free_RVs:
            f.write(f"  {rv}\n")
    print(f"Saved free variable list to {free_rvs_path}")
    
    # Save plots
    plots_dir = os.path.join(output_dir, "plots")
    plot_model_posteriors(pooled_model, output_dir=plots_dir)
