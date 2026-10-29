"""
rl_wald.py

A hierarchical reinforcement-learning shifted-Wald model (hssm.RLSSM) fit on Session 1,
Placebo-arm subjects.

Participants must learn each cue's reward by trial and error, so a Rescorla-Wagner learner
tracks a value Q for each of the 4 cues from the feedback actually received, and Q drives
the drift trial by trial:

  v_t       = max(v0 + scaler * Q_t[cue_t], V_MIN)
  Q_{t+1}   = Q_t[cue_t] + rl_alpha * (feedback_t - Q_t[cue_t])
  feedback  = hit * cue_value / 5    (reward received, scaled to [0, 1]; 0 on failures)
  Q_0       = 0 for every cue

Why a Wald and not a DDM:
  The task has a single possible response, so there is no second boundary. The Wald
  (inverse Gaussian) is the first-passage time of a diffusion with drift v and unit noise to
  ONE threshold, i.e. a one-sided DDM. A two-boundary DDM can only mimic it by pushing v high
  enough that the lower boundary is never reached (prior_success landed at v ~ 6.9), which is
  also outside the range of RLSSM's LAN "ddm" (v in [-3, 3]); see rl_ddm.py.

Decision process ("wald_masked", the single-accumulator racing diffusion model):
  Each trial the accumulator starts uniformly in [0, A] and drifts at rate v toward the
  threshold b = A + B; RT = t + first-passage time. A is trial-to-trial start-point
  variability (mostly shapes the fast end of the RT distribution; A -> 0 is the pure shifted
  Wald). B = b - A > 0 is the gap above the start range, which makes b > A automatic.
  The density is exact (closed form, no LAN). It is HSSM's single-accumulator RDM density
  (hssm.likelihoods.analytical._jax_rdm_tpdf) rewritten in log space with normal tail
  functions: the original formula cancels catastrophically for fast decision times (two
  normal CDFs ~1 subtracted), giving a noisy logp that made NUTS diverge.

Masking:
  RLSSM needs a balanced panel, so no trials are dropped (see
  preprocessing.prepare_session1_controls_rl). Invalid trials (TooFast, NoResponse,
  rt <= 0.15 s) stay in the learning sequence with their real feedback but are coded
  response = -1 and contribute logp = 0. p_outlier = 0 so they contribute exactly 0.

Model formulation (every parameter ~ 1 + (1 | subject)):
  rl_alpha, scaler, v0  (learning process)
  A, B, t               (decision process)

Posterior predictive:
  RLSSM looks up a simulator in ssm-simulators by model name, and none exists for this model,
  so model.sample_posterior_predictive() is unavailable. Given cue and observed feedback,
  v is deterministic for each posterior draw, so the script simulates by hand (start point
  x0 ~ U(0, A); first-passage time over d = b - x0 is exactly InverseGaussian(d/v, d^2)) and
  stores the draws in the trace's posterior_predictive group in HSSM's own "rt,response"
  format. HSSM's plotting functions then use them; masked trials are excluded on both the
  observed and the predicted side.
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import jax.numpy as jnp
from jax.lax import scan
from jax.scipy.stats import norm
from jax.scipy.special import log_ndtr, logsumexp
import hssm
from hssm.rl import get_rlssm_model_config, register_rlssm_model
from hssm.rl.registry import _SSM_REGISTRY, register_ssm
from hssm.likelihoods.analytical import LOGP_LB
from hssm.utils import annotate_function
import arviz as az
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless Oscar runs
import matplotlib.pyplot as plt

hssm.set_floatX("float32", update_jax=True)

MODEL_NAME = "mid_rw_wald"
DECISION_PROCESS = "wald_masked"
N_CUES = 4
V_MIN = 0.01  # the Wald needs positive drift to always respond
CUE_LABELS = {0: "Neutral", 1: "Small", 2: "Medium", 3: "Large"}

LEARNING_BOUNDS = {"rl_alpha": (0.0, 1.0), "scaler": (-5.0, 20.0), "v0": (0.0, 20.0)}
LEARNING_DEFAULTS = [0.2, 1.0, 6.0]
WALD_PARAMS = ["v", "A", "B", "t"]
WALD_BOUNDS = {"A": (0.01, 2.0), "B": (0.01, 3.0), "t": (0.0, 0.6)}
WALD_DEFAULTS = [6.0, 0.3, 0.6, 0.2]


# ---------------------------------------------------------------------------
# Learning process: Rescorla-Wagner value per cue -> drift
# ---------------------------------------------------------------------------

def _rw_trial(q, row):
    rl_alpha, scaler, v0, cue_idx, feedback = row
    cue = jnp.astype(cue_idx, jnp.int32)
    v = jnp.maximum(v0 + scaler * q[cue], V_MIN)
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
    """NumPy twin of compute_v for one subject (used for the PPC and the unit check)."""
    q = np.zeros(N_CUES)
    v = np.empty(len(cue_idx))
    for i, (c, r) in enumerate(zip(cue_idx.astype(int), feedback)):
        v[i] = max(v0 + scaler * q[c], V_MIN)
        q[c] += rl_alpha * (r - q[c])
    return v


# ---------------------------------------------------------------------------
# Decision process: single-accumulator racing diffusion (shifted Wald), masked
# ---------------------------------------------------------------------------

def wald_pdf(t, b, v, A, eps_t=1e-8):
    """
    First-passage-time density of one diffusion accumulator (unit noise) starting
    uniformly in [0, A] with threshold b. Copied from HSSM's private
    hssm.likelihoods.analytical._jax_rdm_tpdf. Kept only as a reference for the unit
    checks: for fast decision times it computes Phi(beta) - Phi(alpha) with both terms ~1,
    which cancels catastrophically (noisy logp -> NUTS divergences). Use wald_logpdf.
    """
    t_eff = jnp.maximum(t, eps_t)
    sqrt_t = jnp.sqrt(t_eff)

    alpha = (b - A - t_eff * v) / sqrt_t
    beta = (b - t_eff * v) / sqrt_t

    return (
        1.0
        / A
        * (
            -v * norm.cdf(alpha)
            + (1.0 / sqrt_t) * norm.pdf(alpha)
            + v * norm.cdf(beta)
            - (1.0 / sqrt_t) * norm.pdf(beta)
        )
    )


def wald_logpdf(t, b, v, A):
    """
    Log of the same density, computed stably:

      pdf = (1/A) * [ v * (Phi(beta) - Phi(alpha)) + (phi(alpha) - phi(beta)) / sqrt(t) ]

    The CDF difference is written with whichever tail does not cancel:
    Phi(-alpha) - Phi(-beta) when alpha > 0 (fast RTs, both CDFs ~1), otherwise
    Phi(beta) - Phi(alpha). All four terms are combined in log space with a signed
    logsumexp, so the far tails neither cancel nor underflow. Requires t > 0, v > 0, A > 0.
    """
    sqrt_t = jnp.sqrt(t)
    alpha = (b - A - t * v) / sqrt_t
    beta = (b - t * v) / sqrt_t

    upper = alpha > 0
    log_cdf_pos = jnp.where(upper, log_ndtr(-alpha), log_ndtr(beta))
    log_cdf_neg = jnp.where(upper, log_ndtr(-beta), log_ndtr(alpha))
    log_phi_alpha = -0.5 * alpha ** 2 - 0.5 * jnp.log(2 * jnp.pi) - 0.5 * jnp.log(t)
    log_phi_beta = -0.5 * beta ** 2 - 0.5 * jnp.log(2 * jnp.pi) - 0.5 * jnp.log(t)
    log_v = jnp.log(v)

    terms = jnp.stack([log_v + log_cdf_pos, log_v + log_cdf_neg, log_phi_alpha, log_phi_beta])
    signs = jnp.array([1.0, -1.0, 1.0, -1.0]).reshape(4, 1)
    log_bracket, sign = logsumexp(terms, axis=0, b=signs, return_sign=True)
    return jnp.where(sign > 0, log_bracket - jnp.log(A), LOGP_LB)


def wald_masked_logp(lan_matrix, epsilon=1e-10):
    """lan_matrix columns: v, A, B, t, rt, response. Rows with response <= 0 give logp 0."""
    lan_matrix = jnp.asarray(lan_matrix)
    v, A, B, t, rt, response = (lan_matrix[:, i] for i in range(6))
    dt = rt - t
    too_fast = dt <= 0.0
    dt_safe = jnp.where(too_fast, 1.0, dt)  # any benign value; replaced below

    logp = jnp.clip(wald_logpdf(dt_safe, A + B, v, A), LOGP_LB, -LOGP_LB)
    logp = jnp.where(too_fast, LOGP_LB, logp)
    return jnp.where(response > 0, logp, 0.0)


def register_models():
    """Register the masked decision process and the RL model (safe to call repeatedly)."""
    if DECISION_PROCESS not in _SSM_REGISTRY:
        register_ssm(
            name=DECISION_PROCESS,
            ssm_base_logp_func=annotate_function(
                inputs=WALD_PARAMS + ["rt", "response"], outputs=["logp"]
            )(wald_masked_logp),
            list_params_ssm=WALD_PARAMS,
            bounds_ssm=WALD_BOUNDS,
            params_default_ssm=WALD_DEFAULTS,
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
        description="Rescorla-Wagner value per cue -> shifted-Wald drift, masked invalid trials",
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


def build_model(df):
    """Builds the RL-Wald. evaluate_models.py imports this so the spec lives in one place."""
    return hssm.RLSSM(
        data=df,
        model_config=get_rlssm_model_config(MODEL_NAME),
        participant_col="subject",
        p_outlier=0,
        lapse=None,
        process_initvals=False,
        include=[
            hierarchical_param("rl_alpha", 0.01, 1.0, 0.2, 0.2, 0.2),
            hierarchical_param("scaler", -5.0, 20.0, 1.0, 3.0, 1.0),
            hierarchical_param("v0", 0.1, 20.0, 6.0, 3.0, 1.5),
            hierarchical_param("A", 0.01, 2.0, 0.3, 0.3, 0.1),
            hierarchical_param("B", 0.01, 3.0, 0.6, 0.5, 0.2),
            hierarchical_param("t", 0.05, 0.6, 0.2, 0.1, 0.05, initval=0.15),  # start below every RT (min 0.20 s)
        ],
    )


def fit_rl_wald_model(df, tune=1000, draws=1000, chains=4):
    print("\n" + "="*60)
    print(" Fitting RL-Wald (Session 1, Controls)")
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
    print("Finished sampling RL-Wald.")
    return model


# ---------------------------------------------------------------------------
# Posterior: per-subject parameters
# ---------------------------------------------------------------------------

FREE_PARAMS = ["rl_alpha", "scaler", "v0", "A", "B", "t"]
PARAM_BOUNDS = {**LEARNING_BOUNDS, **WALD_BOUNDS}


def subject_params(posterior, subjects):
    """
    Returns {param: array (chain, draw, n_subjects)} = Intercept + subject deviation,
    clipped to the parameter bounds, with subjects ordered as in `subjects`.
    """
    out = {}
    for p in FREE_PARAMS:
        intercept = posterior[f"{p}_Intercept"]
        offsets = posterior[f"{p}_1|subject"]
        subj_dim = [d for d in offsets.dims if d not in ("chain", "draw")][0]
        offsets = offsets.sel({subj_dim: list(subjects)})
        values = (intercept + offsets).transpose("chain", "draw", subj_dim).values
        out[p] = np.clip(values, *PARAM_BOUNDS[p])
    return out


# ---------------------------------------------------------------------------
# Posterior predictive (by hand; no ssm-simulators model exists for this RLSSM)
# ---------------------------------------------------------------------------

def simulate_wald(v, A, B, t, rng):
    """One RT per trial: x0 ~ U(0, A), first passage over d = A + B - x0 ~ IG(d/v, d^2)."""
    d = A + B - rng.uniform(0.0, A)
    return t + rng.wald(d / v, d ** 2)


def simulate_ppc(posterior, df, draws_per_chain=50, seed=0):
    """
    Simulates one RT per trial (all rows, aligned with df) for a random subset of draws.
    Returns a DataArray in HSSM's posterior_predictive format:
    "rt,response" with dims (chain, draw, __obs__, rt,response_dim).
    """
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
    out[..., 1] = 1.0
    for c in range(n_chain):
        for d in range(n_draw):
            v, A, B, t = (np.empty(len(df)) for _ in range(4))
            for j, idx in enumerate(rows):
                v[idx] = compute_v_numpy(params["rl_alpha"][c, d, j], params["scaler"][c, d, j],
                                         params["v0"][c, d, j], cue_idx[idx], feedback[idx])
                A[idx] = params["A"][c, d, j]
                B[idx] = params["B"][c, d, j]
                t[idx] = params["t"][c, d, j]
            out[c, d, :, 0] = simulate_wald(v, A, B, t, rng)

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
    pred = pp_dt["posterior_predictive"]["rt,response"].values[..., 0]
    pred = pred.reshape(-1, pred.shape[-1])  # (n_samples, n_valid_trials)
    obs_rt = data["rt"].values
    cue = data["cue_idx"].values
    quantiles = [0.10, 0.30, 0.50, 0.70, 0.90]

    # 1. RT quintiles by cue
    fig, axes = plt.subplots(1, N_CUES, figsize=(16, 4), sharey=True)
    fig.suptitle("RL-Wald: Observed vs. Predicted RT Quintiles (by Cue)", fontsize=13)
    for ax, (c, name) in zip(axes, CUE_LABELS.items()):
        mask = cue == c
        obs_q = np.quantile(obs_rt[mask], quantiles)
        pred_qs = np.quantile(pred[:, mask], quantiles, axis=1).T
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
    plt.savefig(os.path.join(output_dir, "RL_Wald_rt_quintiles.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Learning curve: mean RT across trial bins, by cue
    n_bins = 8
    bins = pd.cut(data["trial_idx"], bins=n_bins, labels=False).values
    fig, axes = plt.subplots(1, N_CUES, figsize=(16, 4), sharey=True)
    fig.suptitle("RL-Wald: Mean RT Across the Session (by Cue)", fontsize=13)
    for ax, (c, name) in zip(axes, CUE_LABELS.items()):
        obs_m, pred_m, pred_lo, pred_hi = [], [], [], []
        for b in range(n_bins):
            mask = (cue == c) & (bins == b)
            obs_m.append(obs_rt[mask].mean())
            draw_means = pred[:, mask].mean(axis=1)
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
    plt.savefig(os.path.join(output_dir, "RL_Wald_learning_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. HSSM's own predictive plot, split by cue (dt and data passed explicitly: valid trials only)
    try:
        hssm.plotting.plot_predictive(model, dt=pp_dt, data=data, col="cue_type",
                                      x_range=(0, 1), title="RL-Wald: Posterior Predictive")
        plt.savefig(os.path.join(output_dir, "RL_Wald_hssm_predictive.png"),
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
    file_prefix = os.path.join(output_dir, "RL_Wald")

    pc = az.plot_trace_dist(idata, var_names=GROUP_VARS)
    pc.savefig(f"{file_prefix}_trace.png", dpi=150, bbox_inches='tight')
    plt.close('all')

    pc = az.plot_dist(idata, var_names=GROUP_VARS)
    pc.savefig(f"{file_prefix}_marginals.png", dpi=300, bbox_inches='tight')
    plt.close('all')

    # Main identifiability risks: drift vs threshold, and start-point range vs t
    pc = az.plot_pair(idata, var_names=["rl_alpha_Intercept", "scaler_Intercept",
                                        "v0_Intercept", "B_Intercept",
                                        "A_Intercept", "t_Intercept"])
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

    # Optional quick run: RL_WALD_SMOKE=<n_subjects> fits a few subjects with a short chain
    smoke = int(os.environ.get("RL_WALD_SMOKE", "0"))
    if smoke:
        df = df[df["subject"].isin(pd.unique(df["subject"])[:smoke])].reset_index(drop=True)

    print(f"Loaded {len(df)} trials from {df['subject'].nunique()} subjects "
          f"({(df['valid'] == 0).sum()} masked).")

    # 2. Fit the model
    if smoke:
        model = fit_rl_wald_model(df, tune=300, draws=200, chains=1)
    else:
        model = fit_rl_wald_model(df)
    idata = model.traces

    # 3. Set up output directory
    date_str     = datetime.datetime.now().strftime("%Y-%m-%d")
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_dir     = os.path.join(project_root, "outputs",
                                f"rl_wald{'_smoke' if smoke else ''}_{date_str}")

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
    trace_path = os.path.join(models_dir, "rl_wald_model_trace.nc")
    idata.to_netcdf(trace_path)
    print(f"Saved trace to {trace_path}")

    # 6. Save free RVs and convergence summary
    with open(os.path.join(output_dir, "free_variables.txt"), "w") as f:
        f.write("=== RL-Wald Free RVs ===\n")
        for rv in model.pymc_model.free_RVs:
            f.write(f"  {rv}\n")
    summary = az.summary(idata, var_names=GROUP_VARS + [f"{p}_1|subject_sigma" for p in FREE_PARAMS])
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
