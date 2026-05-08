"""
evaluate_models.py

A post-hoc evaluation script that runs AFTER training is complete.
It loads saved traces from an output folder, then produces:
  1. Posterior Predictive Check (PPC) plots — does the model predict the observed data?
  2. az.compare table — which model (Baseline vs Continuous) fits better per subject?

Usage:
    python evaluate_models.py <path_to_output_folder>

Example:
    python evaluate_models.py ../outputs/continuous_unpooled_2026-05-07_Job123456
    python evaluate_models.py ../outputs/continuous_pooled_2026-05-07_v1

This script is intentionally separate from training to keep evaluation flexible and repeatable.
"""

import sys
import os
import glob
import pandas as pd
import arviz as az
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless cluster use
import matplotlib.pyplot as plt


def run_ppc_and_compare(output_dir, data_path):
    """
    For each subject in the output folder, loads the baseline and continuous traces,
    runs PPC plots, runs az.compare, and saves all results back into the output folder.
    """
    models_dir = os.path.join(output_dir, "models")
    eval_dir = os.path.join(output_dir, "evaluation")
    ppc_dir = os.path.join(eval_dir, "ppc_plots")
    os.makedirs(ppc_dir, exist_ok=True)

    # Load the original data for PPC comparison
    print(f"Loading data from '{data_path}'...")
    df = pd.read_csv(data_path)

    # Find all baseline traces — each corresponds to one subject
    baseline_traces = sorted(glob.glob(os.path.join(models_dir, "*_baseline_model_trace.nc")))

    if not baseline_traces:
        print(f"No baseline traces found in '{models_dir}'. Exiting.")
        sys.exit(1)

    print(f"Found {len(baseline_traces)} subject(s) to evaluate.\n")

    compare_results = []

    for baseline_path in baseline_traces:
        # Derive the subject ID and paired continuous trace path from the filename
        fname = os.path.basename(baseline_path)
        subject_id = fname.replace("_baseline_model_trace.nc", "").replace("Sub_", "")
        continuous_path = os.path.join(models_dir, f"Sub_{subject_id}_continuous_model_trace.nc")

        if not os.path.exists(continuous_path):
            print(f"  WARNING: No continuous trace found for Subject {subject_id}. Skipping.")
            continue

        print(f"Evaluating Subject {subject_id}...")

        # Load traces from disk
        baseline_trace = az.from_netcdf(baseline_path)
        continuous_trace = az.from_netcdf(continuous_path)

        # -----------------------------------------------------------------------
        # 1. Posterior Predictive Check (PPC)
        # -----------------------------------------------------------------------
        # az.plot_ppc overlays the model's simulated RT distributions (posterior predictive)
        # on top of the actual observed RT data, so you can see if the model is well-calibrated.
        # NOTE: HSSM stores posterior predictive samples in the trace when
        # sample_posterior_predictive() is called during training. If the trace
        # doesn't contain 'posterior_predictive', this plot will be skipped.
        for trace, label in [(baseline_trace, "Baseline"), (continuous_trace, "Continuous")]:
            if "posterior_predictive" in trace:
                ax = az.plot_ppc(trace, observed_rug=True)
                ax.set_title(f"Sub {subject_id} — {label} Model: Posterior Predictive Check")
                ppc_path = os.path.join(ppc_dir, f"Sub_{subject_id}_{label}_ppc.png")
                plt.savefig(ppc_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  Saved PPC plot: {ppc_path}")
            else:
                print(f"  NOTE: No posterior predictive samples in {label} trace for Subject {subject_id}.")
                print(f"  To generate PPC plots, call model.sample_posterior_predictive() during training.")

        # -----------------------------------------------------------------------
        # 2. Model Comparison via az.compare (LOO cross-validation)
        # -----------------------------------------------------------------------
        # az.compare ranks models by their ELPD (Expected Log Predictive Density).
        # A higher ELPD means the model predicts held-out data better.
        # The 'weight' column shows the relative evidence for each model.
        comparison_df = az.compare(
            {"Baseline": baseline_trace, "Continuous": continuous_trace},
            ic="loo"
        )
        comparison_df.insert(0, "subject_id", subject_id)
        compare_results.append(comparison_df.reset_index().rename(columns={"index": "model"}))
        print(f"  Model comparison for Subject {subject_id}:\n{comparison_df}\n")

    # -----------------------------------------------------------------------
    # 3. Save combined comparison table across all subjects
    # -----------------------------------------------------------------------
    if compare_results:
        combined = pd.concat(compare_results, ignore_index=True)
        comparison_path = os.path.join(eval_dir, "model_comparison.csv")
        combined.to_csv(comparison_path, index=False)
        print(f"\nSaved combined model comparison table to '{comparison_path}'")

        # Print a quick winner summary
        winners = combined[combined["rank"] == 0][["subject_id", "model", "elpd_loo", "p_loo"]]
        print("\n=== Winning Model Per Subject (by LOO ELPD) ===")
        print(winners.to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_models.py <path_to_output_folder>")
        print("Example: python evaluate_models.py ../outputs/continuous_unpooled_2026-05-07_Job123")
        sys.exit(1)

    output_dir = sys.argv[1]
    if not os.path.isdir(output_dir):
        print(f"ERROR: Output folder '{output_dir}' does not exist.")
        sys.exit(1)

    # Data path relative to the models/ directory where this script lives
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../data/mid_data_cleaned_hssm.csv")

    run_ppc_and_compare(output_dir, data_path)
