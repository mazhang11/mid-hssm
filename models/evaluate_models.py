"""
evaluate_models.py

A post-hoc evaluation script that runs AFTER training is complete.
Loads a saved trace and the original observed data, reconstructs an
hssm.HSSM model object, then uses HSSM's native plotting API to produce:

  1. Global posterior predictive plot  (hssm.plotting.plot_predictive)
  2. Per-subject × condition PPC grid  (custom, looping over subjects)
  3. Quantile probability plot         (hssm.plotting.plot_quantile_probability)
  4. Marginal posterior / trace plot   (model.plot_trace via ArviZ)
  5. Model graph                       (model.graph)
  6. t_1|subject_offset outlier report (identify extreme non-decision time subjects)
  7. az.compare table [commented out — requires a baseline trace]

Usage:
    python evaluate_models.py <path_to_output_folder>

Example:
    python evaluate_models.py ../outputs/prior_success_2026-05-12_v1
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
import hssm
import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Model reconstruction
# ---------------------------------------------------------------------------

def rebuild_model(trace, df, model_name):
    """
    Reconstructs a minimal hssm.HSSM object whose .traces attribute is the
    loaded InferenceData object.  The model formula must match what was used
    during training so that HSSM can attach the trace correctly.

    Returns the reconstructed model, or None on failure.
    """
    try:
        if "rl_wald" in model_name:
            # RLSSM: the spec lives in rl_wald.build_model (imported lazily, since importing
            # it registers the RL model and sets floatX to float32)
            import rl_wald
            model = rl_wald.build_model(df)
        elif "rl_ddm_analytical" in model_name:
            # RLSSM with the analytical DDM (hssm >= 0.5.1); spec lives in rl_ddm.build_model
            import rl_ddm
            model = rl_ddm.build_model(df)
        elif "prior_success" in model_name:
            model = hssm.HSSM(
                data=df,
                model="ddm",
                include=[
                    {
                        "name": "v",
                        "formula": "v ~ 1 + cue_value + last_trial_success + (1 + cue_value | subject)"
                    },
                    {"name": "a", "formula": "a ~ 1 + (1 | subject)"},
                    {"name": "t", "formula": "t ~ 1 + (1 | subject)"},
                    # z fixed at 0.5 — intentionally omitted
                ]
            )
        elif "continuous_pooled" in model_name:
            model = hssm.HSSM(
                data=df,
                model="ddm",
                include=[
                    {"name": "v", "formula": "v ~ 1 + cue_value + (1 + cue_value | subject)"},
                    {"name": "a", "formula": "a ~ 1 + (1 | subject)"},
                    {"name": "t", "formula": "t ~ 1 + (1 | subject)"},
                    {"name": "z", "formula": "z ~ 1 + cue_value + (1 + cue_value | subject)"},
                ]
            )
        elif "med_pooled" in model_name:
            model = hssm.HSSM(
                data=df,
                model="ddm",
                include=[
                    {
                        "name": "v",
                        "formula": "v ~ 1 + cue_value + treatment + is_responder + (1 + cue_value | subject)"
                    },
                    {
                        "name": "a",
                        "formula": "a ~ 1 + treatment + is_responder + (1 | subject)"
                    },
                    {"name": "t", "formula": "t ~ 1 + (1 | subject)"},
                    {
                        "name": "z",
                        "formula": "z ~ 1 + cue_value + treatment + is_responder + (1 + cue_value | subject)"
                    },
                ]
            )
        else:
            print(f"  WARNING: No known formula for '{model_name}'. Skipping model reconstruction.")
            return None

        # Inject the trace into the model object so HSSM plotting functions can find it
        model.restore_traces(trace)
        return model

    except Exception as e:
        print(f"  ERROR reconstructing model '{model_name}': {e}")
        return None


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def save_predictive_plot(model, output_dir, label, dt=None, data=None):
    """hssm.plotting.plot_predictive — global RT distribution overlay (all subjects pooled)."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        ax = hssm.plotting.plot_predictive(model, dt=dt, data=data, predictive_group="posterior_predictive")
        plt.title(f"{label}: Posterior Predictive Check (Global)")
        plt.savefig(os.path.join(output_dir, f"{label}_ppc_global.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved global PPC plot.")
    except Exception as e:
        print(f"  ERROR generating global PPC plot: {e}")


def save_per_subject_ppc(model, df, output_dir, label):
    """
    Per-subject × condition posterior predictive plot.

    For each subject, draws a small grid of panels — one per cue_type condition
    — overlaying the observed RT histogram (positive = correct, negative = error)
    against a sample of posterior-predictive RT draws.

    Observed RTs are signed: positive when response==1, negative when response==0
    (mirroring the HSSM encoding).  Posterior predictive values from the
    'rt,response' variable are already signed in that convention.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Pull posterior predictive samples — shape (chain, draw, obs, 2)
    # The last dimension packs [rt, response]; index 0 is the signed RT.
    if not hasattr(model.traces, 'posterior_predictive') or \
            'rt,response' not in model.traces.posterior_predictive:
        print("  WARNING: posterior_predictive group not found — "
              "run sample_posterior_predictive first. Skipping per-subject PPC.")
        return

    pp_raw = model.traces.posterior_predictive['rt,response'].values  # (chain, draw, obs, 2)
    pp_vals = pp_raw[..., 0]  # extract signed RT only → (chain, draw, obs)
    n_chains, n_draws, n_obs = pp_vals.shape
    # Flatten chains × draws → one RT per obs per posterior sample
    pp_flat = pp_vals.reshape(n_chains * n_draws, n_obs)  # (S, N)

    conditions = sorted(df['cue_type'].unique())
    subjects   = sorted(df['subject'].unique())
    n_conds    = len(conditions)

    print(f"  Generating per-subject PPC plots ({len(subjects)} subjects × "
          f"{n_conds} conditions)...")

    # Signed observed RTs: positive = chose upper boundary, negative = lower
    obs_signed = df['rt'].values * np.where(df['response'].values == 1, 1, -1)

    # Use up to 50 posterior draws to keep rendering fast
    rng = np.random.default_rng(42)
    draw_idx = rng.choice(n_chains * n_draws, size=min(50, n_chains * n_draws), replace=False)

    fig_rows = (len(subjects) + 3) // 4          # 4 subjects per row
    cols_per_row = min(4, len(subjects))

    for cond in conditions:
        cond_mask = (df['cue_type'] == cond).values
        fig, axes = plt.subplots(
            (len(subjects) + cols_per_row - 1) // cols_per_row,
            cols_per_row,
            figsize=(cols_per_row * 3.5, ((len(subjects) + cols_per_row - 1) // cols_per_row) * 2.8),
            squeeze=False
        )
        fig.suptitle(f"{label}\nPer-Subject PPC — {cond}", fontsize=11, fontweight='bold', y=1.02)

        for s_idx, subj in enumerate(subjects):
            row = s_idx // cols_per_row
            col = s_idx % cols_per_row
            ax  = axes[row][col]

            subj_mask = (df['subject'] == subj).values
            mask = subj_mask & cond_mask

            if mask.sum() == 0:
                ax.set_visible(False)
                continue

            obs_rt = obs_signed[mask]
            rt_min = obs_rt.min() - 0.1
            rt_max = obs_rt.max() + 0.1
            bins = np.linspace(rt_min, rt_max, 25)

            # Observed histogram
            ax.hist(obs_rt, bins=bins, density=True, color='steelblue',
                    alpha=0.6, label='observed')

            # Posterior predictive overlay
            for d in draw_idx:
                pp_rt = pp_flat[d, mask]
                ax.hist(pp_rt, bins=bins, density=True,
                        histtype='step', color='tomato', alpha=0.15, linewidth=0.6)

            ax.set_title(subj, fontsize=6.5)
            ax.set_xlabel('RT (s)', fontsize=6)
            ax.tick_params(labelsize=5)
            ax.axvline(0, color='gray', linewidth=0.6, linestyle='--')

        # Hide any leftover empty panels
        for extra in range(len(subjects), fig_rows * cols_per_row):
            r, c = extra // cols_per_row, extra % cols_per_row
            if r < axes.shape[0] and c < axes.shape[1]:
                axes[r][c].set_visible(False)

        plt.tight_layout()
        safe_cond = cond.replace(' ', '_').replace('/', '-')
        save_path = os.path.join(output_dir, f"{label}_ppc_by_subject_{safe_cond}.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

    print(f"  Saved per-subject PPC plots ({n_conds} condition files).")


def save_quantile_probability_plot(model, output_dir, label, cond_col="cue_value", dt=None, data=None):
    """hssm.plotting.plot_quantile_probability — quantile probability plot."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        ax = hssm.plotting.plot_quantile_probability(
            model,
            cond=cond_col,
            dt=dt,
            data=data,
            quantile_by=["subject"],
            predictive_style="ellipse",
            ellipse_confidence=0.95,
        )
        plt.title(f"{label}: Quantile Probability Plot (by {cond_col})")
        plt.savefig(os.path.join(output_dir, f"{label}_quantile_probability.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved quantile probability plot.")
    except Exception as e:
        print(f"  ERROR generating quantile probability plot: {e}")


def save_trace_plot(model, output_dir, label):
    """model.plot_trace() — ArviZ trace plot for all global fixed effects."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        model.plot_trace()
        plt.savefig(os.path.join(output_dir, f"{label}_trace.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved trace plot.")
    except Exception as e:
        print(f"  ERROR generating trace plot: {e}")


def save_model_graph(model, output_dir, label):
    """model.graph() — graphviz DAG of the model structure."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        graph = model.graph()
        graph.render(
            filename=os.path.join(output_dir, f"{label}_model_graph"),
            format="png",
            cleanup=True,
        )
        print(f"  Saved model graph.")
    except Exception as e:
        print(f"  ERROR generating model graph (graphviz may not be installed): {e}")


def save_t_offset_outlier_diagnostics(trace, df, output_dir, label, z_thresh=2.5):
    """
    Identifies subjects whose t_1|subject_offset (non-decision time random effect)
    is unusually extreme (|z| > z_thresh) and produces:

      (a) A forest plot of ALL subjects' t offsets — so you can see who the
          outliers are in context.
      (b) For each flagged outlier: their raw RT distribution (by condition)
          and overall accuracy vs the group median, so you can decide
          whether to exclude them.

    Outputs a text report and PNG files.
    """
    os.makedirs(output_dir, exist_ok=True)
    t_var = 't_1|subject_offset'

    if t_var not in trace.posterior:
        print(f"  WARNING: '{t_var}' not found in posterior. Skipping outlier diagnostic.")
        return

    # --- extract posterior means per subject ---
    raw   = trace.posterior[t_var].values          # (chain, draw, subject)
    flat  = raw.reshape(-1, raw.shape[-1])          # (S, n_subjects)
    means = flat.mean(axis=0)                       # (n_subjects,)
    sds   = flat.std(axis=0)

    subjects = df['subject'].unique()               # ordered as HSSM encoded them
    # HSSM encodes subjects in sorted order matching the trace dimension
    hssm_subjects = np.array(sorted(df['subject'].unique()))

    z_scores = (means - means.mean()) / means.std()
    outlier_mask = np.abs(z_scores) > z_thresh
    outlier_subjects = hssm_subjects[outlier_mask]

    # ----------------------------------------------------------------
    # (a) Forest plot of all subject t offsets
    # ----------------------------------------------------------------
    sorted_order = np.argsort(means)
    fig, ax = plt.subplots(figsize=(10, max(6, len(hssm_subjects) * 0.22)))

    colors = ['tomato' if outlier_mask[i] else 'steelblue' for i in sorted_order]
    y_pos  = np.arange(len(hssm_subjects))
    ci_lo  = np.percentile(flat, 2.5,  axis=0)[sorted_order]
    ci_hi  = np.percentile(flat, 97.5, axis=0)[sorted_order]
    m_sort = means[sorted_order]
    labs   = hssm_subjects[sorted_order]

    ax.barh(y_pos, m_sort, xerr=[m_sort - ci_lo, ci_hi - m_sort],
            color=colors, alpha=0.75, error_kw=dict(elinewidth=0.8, capsize=2))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labs, fontsize=6)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('t_1|subject_offset (posterior mean ± 95% CI)')
    ax.set_title(f'{label}: Non-Decision Time Subject Offsets\n'
                 f'(red = |z| > {z_thresh}; outliers: {", ".join(outlier_subjects) if len(outlier_subjects) else "none"})',
                 fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{label}_t_offset_forest.png"), dpi=250, bbox_inches='tight')
    plt.close()
    print(f"  Saved t_1|subject_offset forest plot.")

    # Print text report
    print(f"\n  === t_1|subject_offset outlier report (|z| > {z_thresh}) ===")
    if len(outlier_subjects) == 0:
        print("  No outliers detected.")
    else:
        for subj in outlier_subjects:
            idx = np.where(hssm_subjects == subj)[0][0]
            print(f"  {subj}: mean offset = {means[idx]:.3f}, "
                  f"z = {z_scores[idx]:.2f}, "
                  f"95% CI = [{np.percentile(flat[:, idx], 2.5):.3f}, "
                  f"{np.percentile(flat[:, idx], 97.5):.3f}]")

    # ----------------------------------------------------------------
    # (b) RT diagnostics for each outlier subject
    # ----------------------------------------------------------------
    if len(outlier_subjects) == 0:
        return

    grp_accuracy = df.groupby('subject')['response'].mean()
    grp_median_rt = df.groupby('subject')['rt'].median()

    for subj in outlier_subjects:
        subj_df = df[df['subject'] == subj]
        conditions = sorted(subj_df['cue_type'].unique())
        n_conds = len(conditions)

        fig, axes = plt.subplots(1, n_conds + 1,
                                 figsize=((n_conds + 1) * 3.5, 4),
                                 gridspec_kw={'width_ratios': [1] * n_conds + [0.9]})

        for c_idx, cond in enumerate(conditions):
            ax = axes[c_idx]
            cond_df = subj_df[subj_df['cue_type'] == cond]
            all_df  = df[df['cue_type'] == cond]
            ax.hist(all_df['rt'],   bins=20, density=True,
                    color='lightgray', alpha=0.8, label='all subjects')
            ax.hist(cond_df['rt'],  bins=15, density=True,
                    color='tomato',    alpha=0.7, label=subj)
            ax.set_title(cond, fontsize=8)
            ax.set_xlabel('RT (s)', fontsize=7)
            ax.legend(fontsize=6)

        # Accuracy comparison panel
        ax = axes[-1]
        overall_acc = grp_accuracy.values
        subj_acc    = grp_accuracy.get(subj, np.nan)
        ax.hist(overall_acc, bins=15, color='lightgray', alpha=0.8, label='group')
        ax.axvline(subj_acc, color='tomato', linewidth=2, label=f'{subj}\n({subj_acc:.2f})')
        ax.set_title('Accuracy', fontsize=8)
        ax.set_xlabel('P(correct)', fontsize=7)
        ax.legend(fontsize=6)

        idx = np.where(hssm_subjects == subj)[0][0]
        fig.suptitle(f"Outlier Diagnostic: {subj}\n"
                     f"t offset = {means[idx]:.3f} (z = {z_scores[idx]:.2f})",
                     fontsize=9, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{label}_outlier_{subj}.png"),
                    dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved outlier diagnostic for {subj}.")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(output_dir, data_path):
    models_dir = os.path.join(output_dir, "models")
    eval_dir = os.path.join(output_dir, "evaluation")

    print(f"Loading data from '{data_path}'...")
    try:
        df = pd.read_csv(data_path)
        print(f"  Loaded {len(df)} trials from {df['subject'].nunique()} subjects.")
    except FileNotFoundError:
        print(f"  ERROR: Could not load data from '{data_path}'. Exiting.")
        sys.exit(1)

    trace_files = sorted(glob.glob(os.path.join(models_dir, "*.nc")))
    if not trace_files:
        print(f"No .nc trace files found in '{models_dir}'. Exiting.")
        sys.exit(1)

    print(f"\nFound {len(trace_files)} trace(s) to evaluate.\n")

    for trace_path in trace_files:
        label = os.path.basename(trace_path).replace(".nc", "")
        print(f"{'='*60}")
        print(f"Evaluating: {label}")
        print(f"{'='*60}")

        # 1. Load trace from disk
        print("  Loading trace from disk...")
        trace = az.from_netcdf(trace_path)

        # 2. Reconstruct the HSSM model object and attach the trace
        print("  Reconstructing HSSM model object...")
        model = rebuild_model(trace, df, label)
        if model is None:
            print("  Skipping plots for this trace.\n")
            continue

        label_dir = os.path.join(eval_dir, label)

        # RL models keep masked (invalid) trials in the data with placeholder RTs.
        # Evaluate the posterior predictive on valid trials only, on both sides.
        ppc_dt, ppc_df = None, df
        if "valid" in df.columns and "posterior_predictive" in model.traces:
            if "rl_ddm" in label:
                import rl_ddm as rl_module
            else:
                import rl_wald as rl_module
            ppc_dt, ppc_df = rl_module.valid_view(model.traces, df)
            model.traces["posterior_predictive"] = ppc_dt["posterior_predictive"]

        # 3. Global posterior predictive plot (pooled across subjects)
        save_predictive_plot(model, label_dir, label, dt=ppc_dt, data=ppc_df if ppc_dt is not None else None)

        # 4. Per-subject × condition PPC grid
        save_per_subject_ppc(model, ppc_df, label_dir, label)

        # 5. Quantile probability plot
        save_quantile_probability_plot(model, label_dir, label, cond_col="cue_value",
                                       dt=ppc_dt, data=ppc_df if ppc_dt is not None else None)

        # 6. Trace plot (marginal posteriors)
        save_trace_plot(model, label_dir, label)

        # 7. Model graph
        save_model_graph(model, label_dir, label)

        # 8. t_1|subject_offset outlier diagnostic
        save_t_offset_outlier_diagnostics(trace, df, label_dir, label)

        print(f"  All plots saved to: {label_dir}\n")

    # -----------------------------------------------------------------------
    # MODEL COMPARISON — requires both baseline and continuous traces.
    # Re-enable when both are available in the models/ directory.
    # -----------------------------------------------------------------------
    '''
    baseline_traces = sorted(glob.glob(os.path.join(models_dir, "*_baseline_model_trace.nc")))
    compare_results = []

    for baseline_path in baseline_traces:
        fname = os.path.basename(baseline_path)
        subject_id = fname.replace("_baseline_model_trace.nc", "").replace("Sub_", "")
        continuous_path = os.path.join(models_dir, f"Sub_{subject_id}_continuous_model_trace.nc")

        if os.path.exists(continuous_path):
            baseline_trace = az.from_netcdf(baseline_path)
            continuous_trace = az.from_netcdf(continuous_path)

            comparison_df = az.compare(
                {"Baseline": baseline_trace, "Continuous": continuous_trace},
                ic="loo"
            )
            comparison_df.insert(0, "subject_id", subject_id)
            compare_results.append(comparison_df.reset_index().rename(columns={"index": "model"}))
            print(f"  Model comparison for Subject {subject_id}:\n{comparison_df}\n")

    if compare_results:
        combined = pd.concat(compare_results, ignore_index=True)
        comparison_path = os.path.join(eval_dir, "model_comparison.csv")
        combined.to_csv(comparison_path, index=False)
        print(f"\nSaved model comparison table to '{comparison_path}'")
        winners = combined[combined["rank"] == 0][["subject_id", "model", "elpd_loo", "p_loo"]]
        print("\n=== Winning Model Per Subject (by LOO ELPD) ===")
        print(winners.to_string(index=False))
    '''


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_models.py <path_to_output_folder>")
        sys.exit(1)

    output_dir = sys.argv[1]
    if not os.path.isdir(output_dir):
        print(f"ERROR: Output folder '{output_dir}' does not exist.")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Automatically select the right data CSV based on the folder name
    if "rl_wald" in output_dir or "rl_ddm" in output_dir:
        data_path = os.path.join(script_dir, "../data/cleaned_rl_model_controls.csv")
    elif "prior_success" in output_dir:
        data_path = os.path.join(script_dir, "../data/mid_session1_controls_hssm.csv")
    else:
        data_path = os.path.join(script_dir, "../data/mid_data_cleaned_hssm.csv")

    run_evaluation(output_dir, data_path)
