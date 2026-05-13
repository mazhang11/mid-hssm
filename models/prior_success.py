"""
prior_success.py  (renamed from success_pooled.py)

A simplified hierarchical DDM fit on Session 1, Placebo-arm subjects only.

Design decisions (per TODO 8 May 2026):
  - Session 1 only, controls only: reduces confounds from medication and crossover.
  - z (starting bias) fixed at 0.5: simplifies the model; bias is not a primary target here.
  - last_trial_success added as a trial-level covariate on v: captures sequential RT dynamics
    (subjects typically respond faster after a successful trial).
  - a and t are kept as intercept-only with random subject effects (see rationale below).
  - Quintile + posterior predictive plots: used to evaluate model accuracy via data generation.

Why a and t do NOT depend on cue_value or last_trial_success:
  - a (boundary separation): Represents the subject's speed-accuracy threshold — how cautious
    they are before committing. In a simple go/no-go RT task like MID, this is theoretically
    stable within a session rather than flexibly tracking trial-by-trial incentives. Critically,
    allowing both v and a to vary with cue_value creates a strong identifiability problem: both
    parameters move RTs in the same direction, making them hard for the sampler to distinguish.
  - t (non-decision time): Captures perceptual encoding + motor execution time — the fixed
    sensorimotor overhead. This is not expected to vary with incentive magnitude or sequential
    feedback. Making t depend on cue_value would be theoretically hard to justify and would
    introduce further tradeoffs with v.
  The principle is: put covariates where there is both theoretical motivation and statistical
  separability. If model comparison (az.compare) later reveals systematic residuals in a or t,
  those covariates can be added incrementally.

Model formulation:
  v ~ 1 + cue_value + last_trial_success + (1 + cue_value | subject)
  a ~ 1 + (1 | subject)
  t ~ 1 + (1 | subject)
  z = 0.5  (fixed, not estimated)
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import hssm
import arviz as az
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless Oscar runs
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_prior_success_model(df):
    """
    Fits the simplified hierarchical DDM for Session 1 controls.

    Fixed effects on v:
      - Intercept          : baseline drift at neutral cue, after a failed trial
      - cue_value          : reward sensitivity slope
      - last_trial_success : sequential dynamics — does prior success speed up RT?

    z is omitted from include[], which fixes it at the HSSM default of 0.5.
    """
    print("\n" + "="*60)
    print(" Fitting Prior-Success Pooled DDM (Session 1, Controls)")
    print("="*60)

    model = hssm.HSSM(
        data=df,
        model="ddm",
        include=[
            {
                "name": "v",
                "formula": "v ~ 1 + cue_value + last_trial_success + (1 + cue_value | subject)"
            },
            {
                "name": "a",
                "formula": "a ~ 1 + (1 | subject)"
            },
            {
                "name": "t",
                "formula": "t ~ 1 + (1 | subject)"
            },
            # z is intentionally omitted → fixed at 0.5
        ]
    )

    model.sample(
        tune=1000,
        draws=1000,
        chains=4,
        cores=4,
        target_accept=0.9,
        sampler="nuts_numpyro"
    )
    print("Finished sampling prior-success pooled model.")
    return model


# ---------------------------------------------------------------------------
# Plotting: posterior marginals and pair plots
# ---------------------------------------------------------------------------

def plot_posterior_marginals(model, output_dir):
    """Saves marginal posterior plots for the global fixed effects."""
    os.makedirs(output_dir, exist_ok=True)
    file_prefix = os.path.join(output_dir, "Prior_Success_Pooled")

    # 1. Marginal posteriors — the key fixed effects
    az.plot_posterior(
        model.traces,
        var_names=["v_Intercept", "v_cue_value", "v_last_trial_success",
                   "a_Intercept", "t_Intercept"]
    )
    plt.suptitle("Prior-Success Pooled DDM: Global Fixed Effects")
    plt.savefig(f"{file_prefix}_marginals.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Pair plot — check for tradeoffs between drift intercept, cue slope, and success effect
    az.plot_pair(
        model.traces,
        var_names=["v_Intercept", "v_cue_value", "v_last_trial_success", "a_Intercept"],
        kind="kde",
        marginals=True
    )
    plt.suptitle("Prior-Success Pooled DDM: Tradeoff Check")
    plt.savefig(f"{file_prefix}_tradeoffs.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved posterior marginal and tradeoff plots.")


# ---------------------------------------------------------------------------
# Plotting: posterior predictive check
# ---------------------------------------------------------------------------

def plot_posterior_predictive(model, output_dir):
    """
    Generates a posterior predictive check (PPC) plot.
    Overlays model-simulated RT distributions on top of observed RTs.
    """
    os.makedirs(output_dir, exist_ok=True)

    if "posterior_predictive" not in model.traces:
        print("  NOTE: No posterior predictive samples found. "
              "Call model.sample_posterior_predictive() before plotting.")
        return

    ax = az.plot_ppc(model.traces, observed_rug=True)
    ax.set_title("Prior-Success Pooled DDM: Posterior Predictive Check")
    ax.set_xlabel("Reaction Time (s)")
    plt.savefig(os.path.join(output_dir, "Prior_Success_Pooled_ppc.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved PPC plot.")


# ---------------------------------------------------------------------------
# Plotting: RT quantile plot
# ---------------------------------------------------------------------------

def plot_rt_quintiles(model, observed_df, output_dir):
    """
    Quantile-Probability (Q-P) plot comparing observed and model-predicted RT quantiles.

    For each cue condition and each quintile (10th, 30th, 50th, 70th, 90th percentile),
    we compare the observed RT at that quantile to the distribution of model-predicted
    RTs at the same quantile. A well-fitting model will cluster tightly around the
    diagonal (predicted ≈ observed).

    This is a standard DDM diagnostic used in Ratcliff & McKoon (2008) and Matzke & Wagenmakers (2009).
    """
    os.makedirs(output_dir, exist_ok=True)

    if "posterior_predictive" not in model.traces:
        print("  NOTE: No posterior predictive samples found. Skipping quintile plot.")
        return

    quantiles = [0.10, 0.30, 0.50, 0.70, 0.90]
    cue_labels = {0.0: "Neutral", 0.5: "Small", 1.0: "Medium", 5.0: "Large"}

    # Extract posterior predictive RT samples — shape: (chain, draw, trial)
    ppc_rt = model.traces.posterior_predictive["rt"].values
    ppc_rt_flat = ppc_rt.reshape(-1, ppc_rt.shape[-1])  # (n_samples, n_trials)

    obs_rt  = observed_df['rt'].values
    obs_cue = observed_df['cue_value'].values

    fig, axes = plt.subplots(1, len(cue_labels), figsize=(16, 4), sharey=True)
    fig.suptitle("RT Quintile Plot: Observed vs. Model-Predicted (by Cue Condition)",
                 fontsize=13)

    for ax, (cue_val, cue_name) in zip(axes, cue_labels.items()):
        mask  = obs_cue == cue_val
        obs_q = np.quantile(obs_rt[mask], quantiles)

        # Predicted quantiles across all posterior samples
        pred_qs     = np.quantile(ppc_rt_flat[:, mask], quantiles, axis=1).T
        pred_q_mean = pred_qs.mean(axis=0)
        pred_q_low  = np.percentile(pred_qs, 2.5, axis=0)
        pred_q_high = np.percentile(pred_qs, 97.5, axis=0)

        ax.plot([0, 2], [0, 2], 'k--', alpha=0.4, label='Perfect fit')
        ax.errorbar(
            obs_q, pred_q_mean,
            yerr=[pred_q_mean - pred_q_low, pred_q_high - pred_q_mean],
            fmt='o', capsize=4, color='steelblue', label='Quintiles ± 95% CI'
        )
        ax.set_title(f"{cue_name} Reward\n(cue_value={cue_val})")
        ax.set_xlabel("Observed RT Quintile (s)")
        if ax == axes[0]:
            ax.set_ylabel("Predicted RT Quintile (s)")
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Prior_Success_Pooled_rt_quintiles.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved RT quintile plot.")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Load the Session 1 controls dataset
    data_path = "../data/mid_session1_controls_hssm.csv"
    print(f"Loading data from '{data_path}'...")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. "
            "Run: python utils/preprocessing.py  (which also calls prepare_session1_controls())"
        )

    df = pd.read_csv(data_path)

    # Validate required columns
    required = ["subject", "cue_value", "last_trial_success", "rt", "response"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Data is missing columns: {missing}. "
            "Re-run utils/preprocessing.py to regenerate the cleaned CSV."
        )

    print(f"Loaded {len(df)} trials from {df['subject'].nunique()} subjects.")
    print(f"last_trial_success: "
          f"{df['last_trial_success'].sum()} prior successes / "
          f"{(df['last_trial_success'] == 0).sum()} prior failures.")

    # 2. Fit the model
    model = fit_prior_success_model(df)

    # 3. Set up output directory
    date_str     = datetime.datetime.now().strftime("%Y-%m-%d")
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_dir     = os.path.join(project_root, "outputs", f"prior_success_{date_str}")

    counter = 1
    while True:
        output_dir = f"{base_dir}_v{counter}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            break
        counter += 1

    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # 4. Sample posterior predictive (needed for PPC and quintile plots)
    print("\nSampling posterior predictive...")
    model.sample_posterior_predictive()

    # 5. Save trace
    trace_path = os.path.join(models_dir, "prior_success_model_trace.nc")
    az.to_netcdf(model.traces, trace_path)
    print(f"Saved trace to {trace_path}")

    # 6. Save free RVs
    free_rvs_path = os.path.join(output_dir, "free_variables.txt")
    with open(free_rvs_path, "w") as f:
        f.write("=== Prior-Success Pooled Model Free RVs ===\n")
        for rv in model.pymc_model.free_RVs:
            f.write(f"  {rv}\n")
    print(f"Saved free variable list to {free_rvs_path}")

    # 7. Generate all plots
    plots_dir = os.path.join(output_dir, "plots")
    print("\nGenerating plots...")
    plot_posterior_marginals(model, output_dir=plots_dir)
    plot_posterior_predictive(model, output_dir=plots_dir)
    plot_rt_quintiles(model, observed_df=df, output_dir=plots_dir)

    print(f"\nAll outputs saved to: {output_dir}")
