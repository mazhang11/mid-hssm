"""
med_pooled.py

A hierarchical DDM model that extends continuous_pooled.py by adding:
  - treatment    (1 = Bupropion XL, 0 = Placebo)
  - is_responder (1 = Responder,    0 = Non-Responder)

as fixed-effect covariates alongside the monetary incentive (cue_value).

Scientific motivation:
  - The fixed effect of 'treatment' asks: does Bupropion XL shift the group-level
    drift rate / bias relative to placebo, independent of reward level?
  - The fixed effect of 'is_responder' asks: do clinical responders differ in their
    baseline evidence accumulation or starting bias?
  - The interaction term 'cue_value:treatment' (commented out below) would ask:
    does the drug amplify or dampen reward sensitivity specifically.

Random effects (1 + cue_value | subject) allow each subject's intercept and
reward slope to vary, while the fixed effects above explain group-level differences.
"""

import sys
import os
import datetime
import pandas as pd
import hssm
import arviz as az
import matplotlib.pyplot as plt


def fit_med_pooled_model(df):
    """
    Fits the full medication-augmented hierarchical DDM.

    Fixed effects on v and z:
      - Intercept        : baseline drift / bias at neutral cue, placebo, non-responder
      - cue_value        : reward sensitivity slope
      - treatment        : main effect of Bupropion XL vs. Placebo
      - is_responder     : main effect of clinical response status

    To add an interaction (reward sensitivity x treatment), uncomment
    the 'cue_value:treatment' term in the formulas below.

    Random effects on v and z:
      - (1 + cue_value | subject): per-subject intercept and reward slope deviations
    """
    print("\n" + "="*60)
    print(" Fitting Medication-Augmented Hierarchical (Pooled) DDM")
    print("="*60)

    model = hssm.HSSM(
        data=df,
        model="ddm",
        include=[
            {
                "name": "v",
                "formula": (
                    "v ~ 1 + cue_value + treatment + is_responder"
                    # + " + cue_value:treatment"   # Uncomment to test reward x drug interaction
                    # + " + cue_value:is_responder" # Uncomment to test reward x response interaction
                    " + (1 + cue_value | subject)"
                )
            },
            {
                "name": "a",
                # Boundary may reflect caution; treatment could shift it
                "formula": "a ~ 1 + treatment + is_responder + (1 | subject)"
            },
            {
                "name": "t",
                # Non-decision time is unlikely to vary with treatment; keep simple
                "formula": "t ~ 1 + (1 | subject)"
            },
            {
                "name": "z",
                "formula": (
                    "z ~ 1 + cue_value + treatment + is_responder"
                    # + " + cue_value:treatment"   # Uncomment to test bias x drug interaction
                    " + (1 + cue_value | subject)"
                )
            },
        ]
    )

    # nuts_numpyro uses JAX backend for GPU acceleration (requested in the bash script)
    # target_accept=0.9 reduces divergences in complex hierarchical models
    model.sample(
        tune=1000,
        draws=1000,
        chains=4,
        cores=4,
        target_accept=0.9,
        sampler="nuts_numpyro"
    )
    print("Finished sampling medication-augmented pooled model.")
    return model


def plot_model_results(model, output_dir):
    """
    Saves diagnostic plots for the medication-augmented model.
    Focuses on the fixed effects most relevant to the clinical question.
    """
    print("\nGenerating plots for Medication-Augmented Pooled Model...")
    os.makedirs(output_dir, exist_ok=True)
    file_prefix = os.path.join(output_dir, "Med_Pooled_Model")

    # 1. Marginal posteriors for all global fixed effects
    # These are the headline results: does treatment / responder status shift DDM params?
    az.plot_posterior(
        model.traces,
        var_names=[
            "v_Intercept", "v_cue_value", "v_treatment", "v_is_responder",
            "z_Intercept", "z_cue_value", "z_treatment", "z_is_responder",
            "a_Intercept", "a_treatment", "a_is_responder",
            "t_Intercept"
        ]
    )
    plt.suptitle("Med-Augmented Pooled DDM: Global Fixed Effects")
    plt.savefig(f"{file_prefix}_marginals.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Pair plot: key parameters to check for tradeoffs
    az.plot_pair(
        model.traces,
        var_names=["v_Intercept", "v_treatment", "v_is_responder", "z_Intercept"],
        kind="kde",
        marginals=True
    )
    plt.suptitle("Med-Augmented Pooled DDM: Tradeoff Check")
    plt.savefig(f"{file_prefix}_tradeoffs.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Forest plot of treatment and responder fixed effects across all parameters
    # This gives a compact "effect size" summary across the whole model
    az.plot_forest(
        model.traces,
        var_names=["v_treatment", "v_is_responder", "a_treatment", "a_is_responder",
                   "z_treatment", "z_is_responder"],
        combined=True,
        hdi_prob=0.95
    )
    plt.suptitle("Med-Augmented Pooled DDM: Treatment & Responder Effects (95% HDI)")
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_forest_treatment_effects.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plots into {output_dir}/")


if __name__ == "__main__":
    # 1. Load data (must have been preprocessed with treatment/is_responder columns)
    data_path = "../data/mid_data_cleaned_hssm.csv"
    print(f"Loading cleaned data from '{data_path}'...")
    df = pd.read_csv(data_path)

    # Validate that the required columns were added by preprocessing
    required_cols = ["treatment", "is_responder", "cue_value", "subject", "session"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Data is missing columns: {missing}. "
            "Re-run utils/preprocessing.py to generate the full cleaned CSV."
        )

    n_treated  = df.drop_duplicates("subject")["treatment"].sum()
    n_subjects = df["subject"].nunique()
    print(f"Data loaded: {n_subjects} subjects, "
          f"{n_treated} treated (Bupropion XL), {n_subjects - n_treated} placebo.")

    # 2. Fit the medication-augmented hierarchical model
    med_model = fit_med_pooled_model(df)

    # --- SAVE MODEL INFO ---
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_dir     = os.path.join(project_root, "outputs", f"med_pooled_{date_str}")

    # Enumerate so repeated runs on the same day don't overwrite each other
    counter = 1
    while True:
        output_dir = f"{base_dir}_v{counter}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            break
        counter += 1

    # Create models/ subfolder
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Sample posterior predictive (embeds simulated data into trace for PPC in evaluate_models.py)
    med_model.sample_posterior_predictive()

    # Save trace
    trace_path = os.path.join(models_dir, "med_pooled_model_trace.nc")
    az.to_netcdf(med_model.traces, trace_path)
    print(f"Saved trace to {trace_path}")

    # Save free_RVs
    free_rvs_path = os.path.join(output_dir, "free_variables.txt")
    with open(free_rvs_path, "w") as f:
        f.write("=== Med-Augmented Pooled Model Free RVs ===\n")
        for rv in med_model.pymc_model.free_RVs:
            f.write(f"  {rv}\n")
    print(f"Saved free variable list to {free_rvs_path}")

    # Save plots
    plots_dir = os.path.join(output_dir, "plots")
    plot_model_results(med_model, output_dir=plots_dir)
