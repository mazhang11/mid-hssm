"""
rl_ddm.py

A hierarchical reinforcement-learning DDM (hssm.RLSSM) fit on Session 1, Placebo-arm subjects,
using the exact analytical DDM likelihood (requires hssm >= 0.5.1).

Participants must learn each cue's reward by trial and error, so a Rescorla-Wagner learner
tracks a value Q for each of the 4 cues from the feedback actually received, and Q drives
the drift trial by trial:

  v_t       = v0 + scaler * Q_t[cue_t]
  Q_{t+1}   = Q_t[cue_t] + rl_alpha * (feedback_t - Q_t[cue_t])
  feedback  = hit * cue_value / 5    (reward received, scaled to [0, 1]; 0 on failures)
  Q_0       = 0 for every cue

Decision process ("ddm_analytical_masked"):
  hssm.likelihoods.logp_ddm_jax, the JAX port of HSSM's analytical DDM (Navarro & Fuss, 2009;
  the same likelihood prior_success.py uses through hssm.HSSM). Exact, so there is no drift
  cap. An earlier version used RLSSM's LAN "ddm", which is only valid for v in [-3, 3] and
  a in [0.3, 2.5]; this data needs v ~ 7 (prior_success: v_Intercept ~ 6.9), so that fit
  pinned a, t and v at their limits and never converged.

  Note the task has a single possible response: every real response is an upper-boundary
  hit, and the DDM can only explain the absence of lower-boundary responses with a high
  drift. rl_wald.py models the same learner with a one-boundary (Wald) decision process.

Masking:
  RLSSM needs a balanced panel, so no trials are dropped (see
  preprocessing.prepare_session1_controls_rl). Invalid trials (TooFast, NoResponse,
  rt <= 0.15 s) stay in the learning sequence with their real feedback but are coded
  response = -1 and contribute logp = 0. The built-in analytical "ddm" would score those
  rows as lower-boundary responses, hence the masked wrapper. p_outlier = 0 so they
  contribute exactly 0.

Model formulation (every parameter ~ 1 + (1 | subject)):
  rl_alpha, scaler, v0  (learning process)
  a, t                  (decision process)
  z = 0.5               (fixed)
  rl_alpha and a use logit links (HSSM's gen_logit over their bounds), so the hierarchy can
  never push a subject past a bound; with identity links ~30% of transitions diverged.
  Their Intercepts and subject sigmas are therefore on the logit scale in the trace;
  summary.csv adds natural-scale <param>_group rows.

Posterior predictive:
  RLSSM has no simulator for this custom model, so model.sample_posterior_predictive() is
  unavailable. Given cue and observed feedback, v is deterministic for each posterior draw,
  so the script recomputes v per trial, simulates the DDM with ssm-simulators, and stores the
  draws in the trace's posterior_predictive group in HSSM's own "rt,response" format. HSSM's
  plotting functions then use them; masked trials are excluded on both the observed and the
  predicted side.
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import jax.numpy as jnp
from jax.lax import scan
import hssm
from hssm.rl import get_rlssm_model_config, register_rlssm_model
from hssm.rl.registry import _SSM_REGISTRY, register_ssm
from hssm.likelihoods import logp_ddm_jax
from hssm.modelconfig import get_default_model_config
from hssm.utils import annotate_function
import arviz as az
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless Oscar runs
import matplotlib.pyplot as plt

# Precision: float32 by default; RL_DDM_FLOATX=float64 to test/fit in double precision
hssm.set_floatX(os.environ.get("RL_DDM_FLOATX", "float32"), update_jax=True)

MODEL_NAME = "mid_rw_ddm_analytical"
DECISION_PROCESS = "ddm_analytical_masked"
N_CUES = 4
CUE_LABELS = {0: "Neutral", 1: "Small", 2: "Medium", 3: "Large"}

LEARNING_BOUNDS = {"rl_alpha": (0.0, 1.0), "scaler": (-5.0, 20.0), "v0": (0.0, 20.0)}
LEARNING_DEFAULTS = [0.2, 1.0, 6.0]
DDM_PARAMS = ["v", "a", "z", "t"]
DDM_BOUNDS = {"a": (0.3, 5.0), "z": (0.0, 1.0), "t": (0.0, 0.6)}
DDM_DEFAULTS = [6.0, 1.5, 0.5, 0.25]


# ---------------------------------------------------------------------------
# Learning process: Rescorla-Wagner value per cue -> drift
# ---------------------------------------------------------------------------

def _rw_trial(q, row):
    rl_alpha, scaler, v0, cue_idx, feedback = row
    cue = jnp.astype(cue_idx, jnp.int32)
    v = v0 + scaler * q[cue]
    q = q.at[cue].set(q[cue] + rl_alpha * (feedback - q[cue]))
    return q, v


@annotate_function(
    inputs=["rl_alpha", "scaler", "v0", "cue_idx", "feedback"],
    outputs=["v"],
)
def compute_v(subject_trials):
    """One subject's trials (n_trials, 5) -> trial-wise drift (n_trials,)."""
    _, v = scan(_rw_trial, jnp.zeros(N_CUES), subject_trials)
    return v


def compute_v_numpy(rl_alpha, scaler, v0, cue_idx, feedback):
    """NumPy twin of compute_v for one subject (used for the PPC)."""
    q = np.zeros(N_CUES)
    v = np.empty(len(cue_idx))
    for i, (c, r) in enumerate(zip(cue_idx.astype(int), feedback)):
        v[i] = v0 + scaler * q[c]
        q[c] += rl_alpha * (r - q[c])
    return v


# ---------------------------------------------------------------------------
# Decision process: analytical DDM, masked (response = -1) trials contribute 0
# ---------------------------------------------------------------------------

def ddm_masked_logp(lan_matrix):
    """lan_matrix columns: v, a, z, t, rt, response. Rows with response <= 0 give logp 0."""
    lan_matrix = jnp.asarray(lan_matrix)
    v, a, z, t = (lan_matrix[:, i] for i in range(4))
    logp = logp_ddm_jax(lan_matrix[:, 4:6], v, a, z, t)
    return jnp.where(lan_matrix[:, 5] > 0, logp, 0.0)


def register_models():
    """Register the masked decision process and the RL model (safe to call repeatedly)."""
    if DECISION_PROCESS not in _SSM_REGISTRY:
        register_ssm(
            name=DECISION_PROCESS,
            ssm_base_logp_func=annotate_function(
                inputs=DDM_PARAMS + ["rt", "response"], outputs=["logp"]
            )(ddm_masked_logp),
            list_params_ssm=DDM_PARAMS,
            bounds_ssm=DDM_BOUNDS,
            params_default_ssm=DDM_DEFAULTS,
            response=["rt", "response"],
        )
    register_rlssm_model(
        name=MODEL_NAME,
        decision_process=DECISION_PROCESS,
        learning_process={"v": compute_v},
        learning_process_params=list(LEARNING_BOUNDS),
        learning_process_bounds=LEARNING_BOUNDS,
        learning_process_params_default=LEARNING_DEFAULTS,
        extra_fields=["cue_idx", "feedback"],
        choices=[-1, 1],
        description="Rescorla-Wagner value per cue -> analytical DDM drift, masked invalid trials",
        decision_process_loglik_kind="analytical",
    )


register_models()


# ---------------------------------------------------------------------------
# Model construction and fitting
# ---------------------------------------------------------------------------

def hierarchical_param(name, lower, upper, mu, sigma, subject_sigma, initval=None):
    """Group intercept (TruncatedNormal, starting at its mean unless initval is given) +
    per-subject Normal deviation."""
    intercept_kwargs = {"initval": mu if initval is None else initval}
    return hssm.Param(
        name,
        formula=f"{name} ~ 1 + (1|subject)",
        prior={
            "Intercept": hssm.Prior(
                "TruncatedNormal", lower=lower, upper=upper, mu=mu, sigma=sigma,
                **intercept_kwargs,
            ),
            "1|subject": {
                "name": "Normal", "mu": 0,
                "sigma": {"name": "HalfNormal", "sigma": subject_sigma},
            },
        },
    )


def logit_param(name, center, intercept_sigma=1.0, subject_sigma=0.5):
    """
    Group intercept + per-subject deviation on the LOGIT scale of the parameter's bounds
    (HSSM's gen_logit link): value = lower + (upper - lower) * sigmoid(intercept + offset).
    Every subject's value then stays strictly inside the bounds. With identity links the
    hierarchy could push a subject's rl_alpha or a past its bound, where the likelihood is
    -inf; those walls caused ~30% divergent transitions in smoke tests.
    Priors are on the logit scale, centered so the group value starts at `center`.
    """
    lower, upper = PARAM_BOUNDS[name]
    x = float(np.log((center - lower) / (upper - center)))
    return hssm.Param(
        name,
        formula=f"{name} ~ 1 + (1|subject)",
        link="log_logit",
        prior={
            "Intercept": {"name": "Normal", "mu": x, "sigma": intercept_sigma, "initval": x},
            "1|subject": {
                "name": "Normal", "mu": 0,
                "sigma": {"name": "HalfNormal", "sigma": subject_sigma},
            },
        },
    )


def build_model(df):
    """Builds the RL-DDM. evaluate_models.py imports this so the spec lives in one place."""
    # Prior centers follow the analytical prior_success fit (v ~ 6.9, t ~ 0.28).
    # rl_alpha and a use logit links (see logit_param); a is centered at 0.65, the estimate
    # from the 5-subject smoke fits (the RL-DDM's a is well below prior_success's ~1.7).
    return hssm.RLSSM(
        data=df,
        model_config=get_rlssm_model_config(MODEL_NAME),
        participant_col="subject",
        p_outlier=0,
        lapse=None,
        process_initvals=False,
        include=[
            logit_param("rl_alpha", center=0.2),
            hierarchical_param("scaler", -5.0, 20.0, 1.0, 3.0, 1.0),
            hierarchical_param("v0", 0.1, 20.0, 6.0, 3.0, 1.5),
            logit_param("a", center=0.65),
            hierarchical_param("t", 0.05, 0.6, 0.25, 0.1, 0.05, initval=0.15),  # start below every RT (min 0.20 s)
            hssm.Param("z", prior=0.5),  # fixed, not estimated
        ],
    )


def fit_rl_ddm_model(df, tune=1000, draws=1000, chains=4):
    print("\n" + "="*60)
    print(" Fitting analytical RL-DDM (Session 1, Controls)")
    print("="*60)

    model = build_model(df)
    model.sample(
        sampler="numpyro",
        tune=tune,
        draws=draws,
        chains=chains,
        cores=chains,
        target_accept=0.9,
    )
    print("Finished sampling RL-DDM.")
    return model


# ---------------------------------------------------------------------------
# Posterior: per-subject parameters
# ---------------------------------------------------------------------------

FREE_PARAMS = ["rl_alpha", "scaler", "v0", "a", "t"]
PARAM_BOUNDS = {**LEARNING_BOUNDS, **DDM_BOUNDS}
LOGIT_PARAMS = ["rl_alpha", "a"]  # sampled on the logit scale (see logit_param)


def to_natural(p, x):
    """Linear-predictor value -> parameter value (inverse gen_logit for LOGIT_PARAMS)."""
    if p in LOGIT_PARAMS:
        lower, upper = PARAM_BOUNDS[p]
        return lower + (upper - lower) / (1.0 + np.exp(-x))
    return x


def group_params_natural(posterior):
    """Group-level value of each parameter on its natural scale (for summaries and plots).
    For logit-linked parameters this is the value of a subject with offset 0."""
    return xr.Dataset({f"{p}_group": to_natural(p, posterior[f"{p}_Intercept"]) for p in FREE_PARAMS})


def subject_params(posterior, subjects):
    """
    Returns {param: array (chain, draw, n_subjects)} = Intercept + subject deviation,
    mapped to the natural scale (LOGIT_PARAMS) and clipped to the parameter bounds,
    with subjects ordered as in `subjects`.
    """
    out = {}
    for p in FREE_PARAMS:
        intercept = posterior[f"{p}_Intercept"]
        offsets = posterior[f"{p}_1|subject"]
        subj_dim = [d for d in offsets.dims if d not in ("chain", "draw")][0]
        offsets = offsets.sel({subj_dim: list(subjects)})
        values = (intercept + offsets).transpose("chain", "draw", subj_dim).values
        out[p] = np.clip(to_natural(p, values), *PARAM_BOUNDS[p])
    return out


# ---------------------------------------------------------------------------
# Posterior predictive (by hand; no ssm-simulators model exists for this RLSSM)
# ---------------------------------------------------------------------------

def simulate_ppc(posterior, df, draws_per_chain=50, seed=0):
    """
    Simulates one (rt, response) per trial (all rows, aligned with df) for a random subset
    of draws. response = -1 marks a lower-boundary hit, which this task cannot produce.
    Returns a DataArray in HSSM's posterior_predictive format:
    "rt,response" with dims (chain, draw, __obs__, rt,response_dim).
    """
    from ssms.basic_simulators.simulator import simulator

    rng = np.random.default_rng(seed)
    n_draws = posterior.sizes["draw"]
    draw_ids = np.sort(rng.choice(n_draws, size=min(draws_per_chain, n_draws), replace=False))
    sub = posterior.isel(draw=draw_ids)

    subjects = pd.unique(df["subject"])
    params = subject_params(sub, subjects)
    rows = [np.flatnonzero(df["subject"].values == s) for s in subjects]
    cue_idx, feedback = df["cue_idx"].values, df["feedback"].values

    n_chain, n_draw = params["t"].shape[:2]
    out = np.empty((n_chain, n_draw, len(df), 2), dtype=np.float32)
    for c in range(n_chain):
        for d in range(n_draw):
            theta = np.zeros((len(df), 4))
            for j, idx in enumerate(rows):
                theta[idx, 0] = compute_v_numpy(params["rl_alpha"][c, d, j], params["scaler"][c, d, j],
                                                params["v0"][c, d, j], cue_idx[idx], feedback[idx])
                theta[idx, 1] = params["a"][c, d, j]
                theta[idx, 2] = 0.5
                theta[idx, 3] = params["t"][c, d, j]
            sim = simulator(theta=theta, model="ddm", n_samples=1,
                            random_state=int(rng.integers(2**31 - 1)))
            out[c, d, :, 0] = sim["rts"].ravel()
            out[c, d, :, 1] = sim["choices"].ravel()

    return xr.DataArray(
        out,
        dims=("chain", "draw", "__obs__", "rt,response_dim"),
        coords={"chain": sub["chain"].values, "draw": sub["draw"].values,
                "__obs__": np.arange(len(df)), "rt,response_dim": [0, 1]},
        name="rt,response",
    )


def add_posterior_predictive(idata, pp):
    """Stores the simulated draws as the trace's posterior_predictive group."""
    idata["posterior_predictive"] = xr.DataTree(xr.Dataset({"rt,response": pp}))
    return idata


def valid_view(idata, df):
    """Valid trials only, re-indexed 0..n-1 on both sides (for HSSM's plotting functions)."""
    valid_idx = np.flatnonzero(df["valid"].values == 1)
    pp = idata["posterior_predictive"]["rt,response"].isel(__obs__=valid_idx)
    pp = pp.assign_coords(__obs__=np.arange(len(valid_idx)))
    dt = xr.DataTree.from_dict({"posterior_predictive": xr.Dataset({"rt,response": pp})})
    data = df.iloc[valid_idx].reset_index(drop=True)
    return dt, data


# ---------------------------------------------------------------------------
# Plotting: posterior predictive
# ---------------------------------------------------------------------------

def plot_ppc(model, idata, df, output_dir):
    """RT quintiles by cue, the RT learning curve by cue, and HSSM's predictive plot."""
    os.makedirs(output_dir, exist_ok=True)
    pp_dt, data = valid_view(idata, df)
    pp = pp_dt["posterior_predictive"]["rt,response"].values
    pp = pp.reshape(-1, pp.shape[-2], 2)  # (n_samples, n_valid_trials, 2)
    # Lower-boundary hits are responses this task cannot produce: drop them from RT summaries
    pred = np.where(pp[..., 1] > 0, pp[..., 0], np.nan)
    print(f"  Predicted lower-boundary (impossible) responses: {np.isnan(pred).mean():.2%} of trials")
    obs_rt = data["rt"].values
    cue = data["cue_idx"].values
    quantiles = [0.10, 0.30, 0.50, 0.70, 0.90]

    # 1. RT quintiles by cue
    fig, axes = plt.subplots(1, N_CUES, figsize=(16, 4), sharey=True)
    fig.suptitle("RL-DDM: Observed vs. Predicted RT Quintiles (by Cue)", fontsize=13)
    for ax, (c, name) in zip(axes, CUE_LABELS.items()):
        mask = cue == c
        obs_q = np.quantile(obs_rt[mask], quantiles)
        pred_qs = np.nanquantile(pred[:, mask], quantiles, axis=1).T
        mean, lo, hi = pred_qs.mean(0), np.percentile(pred_qs, 2.5, 0), np.percentile(pred_qs, 97.5, 0)
        ax.plot([0, 1.2], [0, 1.2], 'k--', alpha=0.4, label='Perfect fit')
        ax.errorbar(obs_q, mean, yerr=[mean - lo, hi - mean], fmt='o', capsize=4,
                    color='steelblue', label='Quintiles ± 95% CI')
        ax.set_title(f"{name} Reward")
        ax.set_xlabel("Observed RT Quintile (s)")
        if c == 0:
            ax.set_ylabel("Predicted RT Quintile (s)")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "RL_DDM_rt_quintiles.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Learning curve: mean RT across trial bins, by cue
    n_bins = 8
    bins = pd.cut(data["trial_idx"], bins=n_bins, labels=False).values
    fig, axes = plt.subplots(1, N_CUES, figsize=(16, 4), sharey=True)
    fig.suptitle("RL-DDM: Mean RT Across the Session (by Cue)", fontsize=13)
    for ax, (c, name) in zip(axes, CUE_LABELS.items()):
        obs_m, pred_m, pred_lo, pred_hi = [], [], [], []
        for b in range(n_bins):
            mask = (cue == c) & (bins == b)
            obs_m.append(obs_rt[mask].mean())
            draw_means = np.nanmean(pred[:, mask], axis=1)
            pred_m.append(draw_means.mean())
            pred_lo.append(np.percentile(draw_means, 2.5))
            pred_hi.append(np.percentile(draw_means, 97.5))
        x = np.arange(1, n_bins + 1)
        ax.fill_between(x, pred_lo, pred_hi, color='steelblue', alpha=0.3, label='Predicted 95% CI')
        ax.plot(x, pred_m, color='steelblue', label='Predicted mean')
        ax.plot(x, obs_m, 'ko-', label='Observed')
        ax.set_title(f"{name} Reward")
        ax.set_xlabel(f"Trial bin (session split into {n_bins})")
        if c == 0:
            ax.set_ylabel("Mean RT (s)")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "RL_DDM_learning_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. HSSM's own predictive plot, split by cue (dt and data passed explicitly: valid trials only)
    try:
        hssm.plotting.plot_predictive(model, dt=pp_dt, data=data, col="cue_type",
                                      title="RL-DDM: Posterior Predictive")
        plt.savefig(os.path.join(output_dir, "RL_DDM_hssm_predictive.png"),
                    dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"  WARNING: hssm.plotting.plot_predictive failed: {e}")
    plt.close('all')
    print("  Saved PPC quintile, learning-curve and HSSM predictive plots.")


# ---------------------------------------------------------------------------
# Plotting: trace, marginals, tradeoffs
# ---------------------------------------------------------------------------

GROUP_VARS = [f"{p}_Intercept" for p in FREE_PARAMS]


def plot_posterior_diagnostics(idata, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_prefix = os.path.join(output_dir, "RL_DDM")

    pc = az.plot_trace_dist(idata, var_names=GROUP_VARS)
    pc.savefig(f"{file_prefix}_trace.png", dpi=150, bbox_inches='tight')
    plt.close('all')

    # Marginals on the natural scale (rl_alpha and a are sampled on the logit scale)
    natural = xr.DataTree.from_dict({"posterior": group_params_natural(idata["posterior"].to_dataset())})
    pc = az.plot_dist(natural)
    pc.savefig(f"{file_prefix}_marginals.png", dpi=300, bbox_inches='tight')
    plt.close('all')

    # Main identifiability risks: learning rate vs value scaling vs baseline drift vs boundary.
    # Shown on the sampled scale (logit for rl_alpha and a), where the sampler's geometry lives.
    pc = az.plot_pair(idata, var_names=["rl_alpha_Intercept", "scaler_Intercept",
                                        "v0_Intercept", "a_Intercept", "t_Intercept"])
    pc.savefig(f"{file_prefix}_tradeoffs.png", dpi=300, bbox_inches='tight')
    plt.close('all')
    print("  Saved trace, marginal and tradeoff plots.")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Load the balanced Session 1 controls dataset
    data_path = "../data/cleaned_rl_model_controls.csv"
    print(f"Loading data from '{data_path}'...")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. "
            "Run: python utils/preprocessing.py  (which also calls prepare_session1_controls_rl())"
        )

    df = pd.read_csv(data_path)

    required = ["subject", "trial_idx", "cue_idx", "cue_type", "feedback", "valid", "rt", "response"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Data is missing columns: {missing}. "
            "Re-run utils/preprocessing.py to regenerate the cleaned CSV."
        )
    # RLSSM reads the learning sequence in row order
    df = df.sort_values(["subject", "trial_idx"], kind="stable").reset_index(drop=True)

    # Optional quick run: RL_DDM_SMOKE=<n_subjects> fits a few subjects with a short chain
    # (length adjustable with RL_DDM_TUNE / RL_DDM_DRAWS / RL_DDM_CHAINS)
    smoke = int(os.environ.get("RL_DDM_SMOKE", "0"))
    if smoke:
        df = df[df["subject"].isin(pd.unique(df["subject"])[:smoke])].reset_index(drop=True)

    print(f"Loaded {len(df)} trials from {df['subject'].nunique()} subjects "
          f"({(df['valid'] == 0).sum()} masked).")

    # 2. Fit the model
    if smoke:
        model = fit_rl_ddm_model(df,
                                 tune=int(os.environ.get("RL_DDM_TUNE", "300")),
                                 draws=int(os.environ.get("RL_DDM_DRAWS", "200")),
                                 chains=int(os.environ.get("RL_DDM_CHAINS", "1")))
    else:
        model = fit_rl_ddm_model(df)
    idata = model.traces

    # 3. Set up output directory
    date_str     = datetime.datetime.now().strftime("%Y-%m-%d")
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_dir     = os.path.join(project_root, "outputs",
                                f"rl_ddm_analytical{'_smoke' if smoke else ''}_{date_str}")

    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        output_dir = f"{base_dir}_Job{job_id}"
        os.makedirs(output_dir, exist_ok=True)
    else:
        counter = 1
        while True:
            output_dir = f"{base_dir}_v{counter}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                break
            counter += 1

    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # 4. Posterior predictive (by hand), stored in HSSM's format before saving
    print("\nSimulating posterior predictive...")
    pp = simulate_ppc(idata["posterior"].to_dataset(), df, draws_per_chain=50)
    add_posterior_predictive(idata, pp)

    # 5. Save trace
    trace_path = os.path.join(models_dir, "rl_ddm_analytical_model_trace.nc")
    idata.to_netcdf(trace_path)
    print(f"Saved trace to {trace_path}")

    # 6. Save free RVs and convergence summary
    with open(os.path.join(output_dir, "free_variables.txt"), "w") as f:
        f.write("=== Analytical RL-DDM Free RVs ===\n")
        for rv in model.pymc_model.free_RVs:
            f.write(f"  {rv}\n")
    # Sampled-scale rows (convergence: rl_alpha/a Intercepts and sigmas are on the logit scale),
    # plus natural-scale group values (<param>_group) for interpretation
    summary = pd.concat([
        az.summary(idata, var_names=GROUP_VARS + [f"{p}_1|subject_sigma" for p in FREE_PARAMS]),
        az.summary(xr.DataTree.from_dict({"posterior": group_params_natural(idata["posterior"].to_dataset())})),
    ])
    summary.to_csv(os.path.join(output_dir, "summary.csv"))
    print(summary)
    ss = idata["sample_stats"]
    print(f"Step size: {float(ss['step_size'].mean()):.4g}   "
          f"Divergent: {float(ss['diverging'].mean()):.1%}   "
          f"Mean tree depth: {float(ss['tree_depth'].mean()):.2f}")

    # 7. Plots
    plots_dir = os.path.join(output_dir, "plots")
    print("\nGenerating plots...")
    plot_posterior_diagnostics(idata, output_dir=plots_dir)
    plot_ppc(model, idata, df, output_dir=plots_dir)

    print(f"\nAll outputs saved to: {output_dir}")
