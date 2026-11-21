"""
rl_ddm_all.py

The analytical RL-DDM from rl_ddm.py (same Rescorla-Wagner learner, masked analytical DDM,
logit links), fit on EVERY subject and BOTH sessions at once, with session and drug as
covariates.

Data: ../data/cleaned_rl_model_all.csv, written by utils/preprocessing.py:prepare_rl_dataset_all.
  71 subjects, 133 subject x session units of 134 trials:
                 Placebo   Bupropion XL
      Session 1     62          9
      Session 2      6         56

Learning sequences:
  One RLSSM panel unit = one subject's session (participant_col="unit"). Q restarts at 0
  each session, because the cue -> reward mapping is reversed in session 2.
  Random effects of the learning/drift parameters (rl_alpha, scaler, v0) are grouped by
  SUBJECT: a subject's two sessions share one random intercept, so the session and drug
  coefficients describe within-person shifts. The boundary a and non-decision time t get
  one random intercept per subject x session UNIT (RE_GROUP): with t shared across
  sessions, a subject's single t must sit below the fastest RT of both sessions, and in
  smoke tests that wall gave 86-100% divergent transitions (2.5% with per-unit t).

Model formulation (session2 = 1 in session 2; treatment = 1 on Bupropion XL):
  rl_alpha ~ 1 + session2 + treatment + (1|subject)   logit link   learning rate
  scaler   ~ 1 + session2 + treatment + (1|subject)                value -> drift
  v0       ~ 1 + session2 + treatment + (1|subject)                baseline drift
  a        ~ 1 + (1|unit)                              logit link  boundary
  t        ~ 1 + (1|unit)                                          non-decision time
  z = 0.5
  Covariates go on the learning and drift parameters only: per the repo's modeling
  conventions, v and a (or v and t) should not both track the same covariate. Change
  COVARIATE_PARAMS to move them.
  The Intercept is Session 1 on placebo.

Identifiability:
  Session and drug are strongly correlated in this design (most subjects go placebo ->
  Bupropion). They are separated by the 15 units in the off-diagonal cells (6 placebo ->
  placebo, 6 Bupropion -> Bupropion, 3 single-session Bupropion). Expect wide, negatively
  correlated session and drug effects; RL_DDM_effect_tradeoffs.png shows it.

Natural scale:
  Every plot shows parameters on their natural scale (learning rate in [0, 1], boundary in
  its units), never the logit scale the sampler works on. For a logit-linked parameter, a
  condition value is the value of a typical subject (random offset 0) in that condition,
  and an effect is the difference between those values, averaged over the other covariate.
  summary.csv keeps the sampled-scale rows for convergence checks (R-hat, ESS).

Usage:
  cd models && python rl_ddm_all.py
  RL_DDM_SMOKE=8 python rl_ddm_all.py   # 8 subjects spread over all arm patterns, short chain
  RL_DDM_TARGET_ACCEPT=0.99 python rl_ddm_all.py   # override the sampler's target acceptance (default 0.95)

Known data quirk, left as is: a few subject x session units have one isolated very fast RT
(an anticipation, e.g. 0.202 s when the next fastest is 0.302 s). If the unit's t sits above
it, HSSM scores that trial at its LOGP_LB floor (-66.1), which in effect excludes it.
"""

import os
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import arviz as az
import hssm
from hssm.rl import get_rlssm_model_config
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import rl_ddm  # registers the RL model and sets floatX = float32
from rl_ddm import (MODEL_NAME, N_CUES, CUE_LABELS, FREE_PARAMS, PARAM_BOUNDS, LOGIT_PARAMS,
                    to_natural, compute_v_numpy, add_posterior_predictive, valid_view)
import plot_subject_ppc as sp

DATA_PATH = "../data/cleaned_rl_model_all.csv"
RUN_NAME = "rl_ddm_all"
TRACE_NAME = "rl_ddm_all_model_trace.nc"
GROUP_LABEL = "All subjects, both sessions"

COVARIATES = ["session2", "treatment"]
COVARIATE_PARAMS = ["rl_alpha", "scaler", "v0"]
UNIT_COL = "unit"
# 0.95, not 0.9: the 10-29 Session 1 fit diverged 7.9% of the time at 0.9 on the
# rl_alpha x scaler ridge (FAQ.md), which this model shares. Override: RL_DDM_TARGET_ACCEPT.
TARGET_ACCEPT = 0.95
# Random-effect grouping per parameter. a and t vary per subject x session: a single t per
# subject has to stay below the fastest RT of BOTH sessions, and in smoke tests sharing a, t
# (or everything) across sessions gave 86-100% divergences.
RE_GROUP = {"rl_alpha": "subject", "scaler": "subject", "v0": "subject", "a": UNIT_COL, "t": UNIT_COL}

# (session2, treatment, key, label); colors are validated categorical slots 1-4
CELLS = [(0, 0, "s1_placebo", "Session 1 · Placebo"),
         (0, 1, "s1_bupropion", "Session 1 · Bupropion"),
         (1, 0, "s2_placebo", "Session 2 · Placebo"),
         (1, 1, "s2_bupropion", "Session 2 · Bupropion")]
CELL_COLORS = sp.CUE_COLORS
CELL_SHORT = {"s1_placebo": "s1 Pla", "s1_bupropion": "s1 Bup",
              "s2_placebo": "s2 Pla", "s2_bupropion": "s2 Bup"}
EFFECTS = [("treatment", "Drug effect (Bupropion − placebo)"),
           ("session2", "Session effect (session 2 − session 1)")]
PARAM_LABELS = sp.PARAM_LABELS


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def covariate_param(name, covariates, intercept, slope_sigma, subject_sigma, link=None):
    """name ~ 1 + covariates + (1|<RE_GROUP[name]>). `intercept` is the Intercept prior dict.
    Slopes ~ Normal(0, slope_sigma) on the link scale, starting at 0."""
    group = RE_GROUP[name]
    rhs = " + ".join(["1"] + list(covariates) + [f"(1|{group})"])
    prior = {
        "Intercept": intercept,
        f"1|{group}": {"name": "Normal", "mu": 0,
                      "sigma": {"name": "HalfNormal", "sigma": subject_sigma}},
        **{c: {"name": "Normal", "mu": 0.0, "sigma": slope_sigma, "initval": 0.0}
           for c in covariates},
    }
    kwargs = {"link": link} if link else {}
    return hssm.Param(name, formula=f"{name} ~ {rhs}", prior=prior, **kwargs)


def logit_center(name, center):
    lower, upper = PARAM_BOUNDS[name]
    return float(np.log((center - lower) / (upper - center)))


def build_model(df):
    """The session x drug RL-DDM. evaluate_models.py imports this to reattach a trace."""
    covs = lambda p: COVARIATES if p in COVARIATE_PARAMS else []
    x_alpha, x_a = logit_center("rl_alpha", 0.2), logit_center("a", 0.65)
    # Priors follow rl_ddm.build_model (centers from the single-group fits). Slope sigmas:
    # rl_alpha 1.0 on the logit scale (odds x/÷ 2.7), scaler 2.0 and v0 1.5 (about one
    # between-subject SD in the single-group fits).
    return hssm.RLSSM(
        data=df,
        model_config=get_rlssm_model_config(MODEL_NAME),
        participant_col=UNIT_COL,
        p_outlier=0,
        lapse=None,
        process_initvals=False,
        include=[
            covariate_param("rl_alpha", covs("rl_alpha"),
                            {"name": "Normal", "mu": x_alpha, "sigma": 1.0, "initval": x_alpha},
                            slope_sigma=1.0, subject_sigma=0.5, link="log_logit"),
            covariate_param("scaler", covs("scaler"),
                            hssm.Prior("TruncatedNormal", lower=-5.0, upper=20.0, mu=1.0, sigma=3.0, initval=1.0),
                            slope_sigma=2.0, subject_sigma=1.0),
            covariate_param("v0", covs("v0"),
                            hssm.Prior("TruncatedNormal", lower=0.1, upper=20.0, mu=6.0, sigma=3.0, initval=6.0),
                            slope_sigma=1.5, subject_sigma=1.5),
            covariate_param("a", covs("a"),
                            {"name": "Normal", "mu": x_a, "sigma": 1.0, "initval": x_a},
                            slope_sigma=0.5, subject_sigma=0.5, link="log_logit"),
            covariate_param("t", covs("t"),
                            hssm.Prior("TruncatedNormal", lower=0.05, upper=0.6, mu=0.25, sigma=0.1, initval=0.15),
                            slope_sigma=0.05, subject_sigma=0.05),   # t starts below every RT (min 0.20 s)
            hssm.Param("z", prior=0.5),
        ],
    )


def fit(df, tune=1000, draws=1000, chains=4):
    print("\n" + "=" * 60)
    print(f" Fitting session x drug RL-DDM ({GROUP_LABEL})")
    print("=" * 60)
    model = build_model(df)
    model.sample(sampler="numpyro", tune=tune, draws=draws, chains=chains, cores=chains,
                 target_accept=float(os.environ.get("RL_DDM_TARGET_ACCEPT", TARGET_ACCEPT)),
                 chain_method=os.environ.get("RL_DDM_CHAIN_METHOD", "parallel"))
    print("Finished sampling.")
    return model


# ---------------------------------------------------------------------------
# Posterior on the natural scale
# ---------------------------------------------------------------------------

def _covs(p):
    return COVARIATES if p in COVARIATE_PARAMS else []


def linear_predictor(posterior, p, x):
    """Intercept + sum(beta_c * x[c]) for each draw. x: {covariate: scalar or 1-d array}.
    Returns an array (chain, draw, ...) on the link scale (no subject offset)."""
    eta = posterior[f"{p}_Intercept"].values
    out = eta[..., None] if any(np.ndim(v) for v in x.values()) else eta
    for c in _covs(p):
        b = posterior[f"{p}_{c}"].values
        xv = np.asarray(x[c], dtype=float)
        out = out + (b[..., None] * xv if xv.ndim else b * xv)
    return out


def natural_summary(posterior):
    """
    Natural-scale group quantities, each (chain, draw):
      <p>_<cell>            typical-subject value in each session x drug cell (covariate params)
      <p>_group             typical-subject value (params without covariates)
      <p>_drug_effect       Bupropion - placebo, averaged over sessions
      <p>_session_effect    session 2 - session 1, averaged over drug
      <p>_subject_sd        SD across subjects (or subject x session units, for a and t) of
                            their values in the Session 1 placebo cell
    """
    dims = ("chain", "draw")
    coords = {"chain": posterior["chain"].values, "draw": posterior["draw"].values}
    out = {}
    for p in FREE_PARAMS:
        if _covs(p):
            vals = {}
            for s2, trt, key, _ in CELLS:
                vals[(s2, trt)] = to_natural(p, linear_predictor(posterior, p, {"session2": s2, "treatment": trt}))
                out[f"{p}_{key}"] = vals[(s2, trt)]
            out[f"{p}_drug_effect"] = 0.5 * ((vals[(0, 1)] - vals[(0, 0)]) + (vals[(1, 1)] - vals[(1, 0)]))
            out[f"{p}_session_effect"] = 0.5 * ((vals[(1, 0)] - vals[(0, 0)]) + (vals[(1, 1)] - vals[(0, 1)]))
        else:
            out[f"{p}_group"] = to_natural(p, posterior[f"{p}_Intercept"].values)
        eta0 = linear_predictor(posterior, p, {c: 0 for c in COVARIATES})
        offsets = posterior[f"{p}_1|{RE_GROUP[p]}"]
        offsets = offsets.transpose("chain", "draw", ...).values
        out[f"{p}_subject_sd"] = to_natural(p, eta0[..., None] + offsets).std(axis=-1)
    return xr.Dataset({k: (dims, np.asarray(v)) for k, v in out.items()}, coords=coords)


def condition_vars(p):
    return [f"{p}_{key}" for *_, key, _ in CELLS] if _covs(p) else [f"{p}_group"]


def unit_table(df):
    """One row per unit, in data order: unit, subject, session, session2, treatment, cell key."""
    u = df.drop_duplicates(UNIT_COL)[[UNIT_COL, "subject", "session", "session2", "treatment"]]
    u = u.reset_index(drop=True)
    cell_key = {(s2, trt): key for s2, trt, key, _ in CELLS}
    u["cell"] = [cell_key[(a, b)] for a, b in zip(u["session2"], u["treatment"])]
    return u


def unit_params(posterior, units):
    """{param: (chain, draw, n_units)} natural-scale values for each unit (its subject's
    offset plus its session x drug shift), clipped to the parameter bounds."""
    out = {}
    for p in FREE_PARAMS:
        eta = linear_predictor(posterior, p, {c: units[c].values for c in COVARIATES})
        group = RE_GROUP[p]
        offsets = posterior[f"{p}_1|{group}"]
        subj_dim = [d for d in offsets.dims if d not in ("chain", "draw")][0]
        offsets = offsets.sel({subj_dim: units[group].values}).transpose("chain", "draw", subj_dim).values
        if eta.ndim == 2:
            eta = eta[..., None]
        out[p] = np.clip(to_natural(p, eta + offsets), *PARAM_BOUNDS[p])
    return out


# ---------------------------------------------------------------------------
# Posterior predictive (by hand, as in rl_ddm.simulate_ppc)
# ---------------------------------------------------------------------------

def simulate_ppc(posterior, df, draws_per_chain=50, seed=0):
    from ssms.basic_simulators.simulator import simulator

    rng = np.random.default_rng(seed)
    n_draws = posterior.sizes["draw"]
    draw_ids = np.sort(rng.choice(n_draws, size=min(draws_per_chain, n_draws), replace=False))
    sub = posterior.isel(draw=draw_ids)

    units = unit_table(df)
    params = unit_params(sub, units)
    rows = [np.flatnonzero(df[UNIT_COL].values == u) for u in units[UNIT_COL]]
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
        out, dims=("chain", "draw", "__obs__", "rt,response_dim"),
        coords={"chain": sub["chain"].values, "draw": sub["draw"].values,
                "__obs__": np.arange(len(df)), "rt,response_dim": [0, 1]},
        name="rt,response",
    )


def valid_predictions(idata, df):
    """Valid trials only: (data, pred) with pred (n_samples, n_valid) RTs, NaN where the
    simulation hit the lower boundary (a response this task cannot produce)."""
    pp_dt, data = valid_view(idata, df)
    pp = pp_dt["posterior_predictive"]["rt,response"].values
    pp = pp.reshape(-1, pp.shape[-2], 2)
    return pp_dt, data, np.where(pp[..., 1] > 0, pp[..., 0], np.nan)


# ---------------------------------------------------------------------------
# Plots: group level, natural scale
# ---------------------------------------------------------------------------

def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _interval(ax, y, v, color):
    lo90, lo50, med, hi50, hi90 = np.percentile(v, [5, 25, 50, 75, 95])
    ax.hlines(y, lo90, hi90, color=color, lw=1.4)
    ax.hlines(y, lo50, hi50, color=color, lw=4.5)
    ax.plot(med, y, "o", color=color, markersize=7, markeredgecolor="white", markeredgewidth=1.2, zorder=3)


def plot_conditions(nat, out):
    """Typical-subject parameter value in each session x drug cell (median, 50% and 90%)."""
    fig, axes = plt.subplots(1, len(FREE_PARAMS), figsize=(15, 3.4), sharey=True)
    ys = np.arange(len(CELLS))[::-1]
    for ax, p in zip(axes, FREE_PARAMS):
        if _covs(p):
            for y, (*_, key, _), color in zip(ys, CELLS, CELL_COLORS):
                _interval(ax, y, nat[f"{p}_{key}"].values.ravel(), color)
        else:
            _interval(ax, ys.mean(), nat[f"{p}_group"].values.ravel(), sp.INK_2)
            ax.text(0.5, 0.02, "same in all cells\n(no covariates)", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=8, color=sp.MUTED)
        ax.set_title(PARAM_LABELS[p], loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys, [label for *_, label in CELLS])
    axes[0].set_ylim(-0.6, len(CELLS) - 0.4)
    handles = [Line2D([], [], color=sp.INK_2, marker="o", lw=4.5, label="Median & 50% interval"),
               Line2D([], [], color=sp.INK_2, lw=1.4, label="90% interval")]
    sp.header(fig, "RL-DDM parameters by session and drug (natural scale, typical subject)", handles, ncol=2)
    _save(fig, os.path.join(out, "RL_DDM_conditions.png"))


def plot_effects(nat, out):
    """Posterior of each drug and session effect on the natural scale, with P(effect > 0)."""
    fig, axes = plt.subplots(len(EFFECTS), len(COVARIATE_PARAMS), figsize=(12, 5.6), squeeze=False)
    for i, (cov, title) in enumerate(EFFECTS):
        name = "drug_effect" if cov == "treatment" else "session_effect"
        for j, p in enumerate(COVARIATE_PARAMS):
            ax = axes[i, j]
            v = nat[f"{p}_{name}"].values.ravel()
            lo, med, hi = np.percentile(v, [5, 50, 95])
            ax.hist(v, bins=60, density=True, color=sp.PRED_BAND, edgecolor="white", linewidth=0.3)
            ax.axvspan(lo, hi, color=sp.PRED_BAND, alpha=0.25, lw=0)
            ax.axvline(med, color=sp.PRED, lw=2)
            ax.axvline(0, color=sp.INK, lw=1, ls="--")
            ax.text(0.02, 0.97, f"median {med:+.3g}\n90% [{lo:+.3g}, {hi:+.3g}]\nP(> 0) = {np.mean(v > 0):.2f}",
                    transform=ax.transAxes, va="top", fontsize=8, color=sp.INK)
            ax.set_yticks([])
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{PARAM_LABELS[p]}", loc="left")
            if i == len(EFFECTS) - 1:
                ax.set_xlabel("Difference (natural units)")
        axes[i, 0].set_ylabel(title.split(" (")[0], fontsize=10, color=sp.INK)
    handles = [Patch(color=sp.PRED_BAND, label="Posterior"),
               Line2D([], [], color=sp.PRED, lw=2, label="Median"),
               Line2D([], [], color=sp.INK, lw=1, ls="--", label="No effect")]
    sp.header(fig, "Drug effect (Bupropion − placebo, top) and session effect (2 − 1, bottom), natural scale",
              handles, ncol=3)
    _save(fig, os.path.join(out, "RL_DDM_effects.png"))


def plot_trace_and_pairs(nat, out):
    """Trace and pair plots of natural-scale group quantities."""
    dt = xr.DataTree.from_dict({"posterior": nat})
    baseline = [f"{p}_s1_placebo" if _covs(p) else f"{p}_group" for p in FREE_PARAMS]
    effects = [f"{p}_{e}" for p in COVARIATE_PARAMS for e in ("drug_effect", "session_effect")]

    pc = az.plot_trace_dist(dt, var_names=baseline + effects)
    pc.savefig(os.path.join(out, "RL_DDM_trace.png"), dpi=150, bbox_inches="tight")
    plt.close("all")

    pc = az.plot_pair(dt, var_names=baseline)
    pc.savefig(os.path.join(out, "RL_DDM_tradeoffs.png"), dpi=200, bbox_inches="tight")
    plt.close("all")

    # Drug vs session effect per parameter: the design makes them trade off
    fig, axes = plt.subplots(1, len(COVARIATE_PARAMS), figsize=(13, 4.2))
    for ax, p in zip(axes, COVARIATE_PARAMS):
        d, s = nat[f"{p}_drug_effect"].values.ravel(), nat[f"{p}_session_effect"].values.ravel()
        ax.plot(s, d, ".", color=sp.PRED, alpha=0.15, markersize=3)
        ax.axhline(0, color=sp.MUTED, lw=0.8, ls="--")
        ax.axvline(0, color=sp.MUTED, lw=0.8, ls="--")
        ax.text(0.02, 0.97, f"r = {np.corrcoef(s, d)[0, 1]:+.2f}", transform=ax.transAxes,
                va="top", fontsize=9, color=sp.INK)
        ax.set_xlabel("Session effect (2 − 1)")
        ax.set_ylabel("Drug effect (Bup − placebo)")
        ax.set_title(PARAM_LABELS[p], loc="left")
    sp.header(fig, "Drug vs session effect: posterior draws (natural scale)")
    _save(fig, os.path.join(out, "RL_DDM_effect_tradeoffs.png"))

    pc = az.plot_dist(dt, var_names=[v for p in FREE_PARAMS for v in condition_vars(p)]
                      + [f"{p}_subject_sd" for p in FREE_PARAMS])
    pc.savefig(os.path.join(out, "RL_DDM_marginals.png"), dpi=200, bbox_inches="tight")
    plt.close("all")


# ---------------------------------------------------------------------------
# Plots: posterior predictive by condition
# ---------------------------------------------------------------------------

def _cell_of(data):
    key = {(s2, trt): k for s2, trt, k, _ in CELLS}
    return np.array([key[(a, b)] for a, b in zip(data["session2"], data["treatment"])])


def plot_ppc(model, idata, df, out):
    pp_dt, data, pred = valid_predictions(idata, df)
    print(f"  Predicted lower-boundary (impossible) responses: {np.isnan(pred).mean():.2%} of trials")
    obs, cue, cell = data["rt"].values, data["cue_idx"].values, _cell_of(data)
    quantiles = [0.10, 0.30, 0.50, 0.70, 0.90]
    n_bins = 8
    bins = pd.cut(data["trial_idx"], bins=n_bins, labels=False).values

    # 1. RT quintiles, rows = session x drug cell, cols = cue
    fig, axes = plt.subplots(len(CELLS), N_CUES, figsize=(13, 12.5), sharex=True, sharey=True, squeeze=False)
    for i, ((*_, key, label), color) in enumerate(zip(CELLS, CELL_COLORS)):
        for c, name in CUE_LABELS.items():
            ax = axes[i, c]
            m = (cue == c) & (cell == key)
            ax.plot([0.2, 1.0], [0.2, 1.0], color=sp.MUTED, ls="--", lw=1)
            if m.sum() >= 10:
                oq = np.quantile(obs[m], quantiles)
                pq = np.nanquantile(pred[:, m], quantiles, axis=1).T
                mid, lo, hi = np.median(pq, 0), np.percentile(pq, 5, 0), np.percentile(pq, 95, 0)
                ax.errorbar(oq, mid, yerr=[mid - lo, hi - mid], fmt="o", color=color, ecolor=color,
                            markersize=6, markeredgecolor="white", capsize=3)
                ax.text(0.03, 0.97, f"n = {m.sum()}", transform=ax.transAxes, va="top", fontsize=8,
                        color=sp.INK_2)
            if i == 0:
                ax.set_title(f"{name} reward", loc="left")
            if c == 0:
                ax.set_ylabel(f"{label}\nPredicted RT quantile (s)")
            if i == len(CELLS) - 1:
                ax.set_xlabel("Observed RT quantile (s)")
    handles = [Line2D([], [], marker="o", ls="", color=col, label=lab) for (*_, lab), col in zip(CELLS, CELL_COLORS)]
    handles.append(Line2D([], [], color=sp.MUTED, ls="--", label="Perfect fit"))
    sp.header(fig, "Observed vs predicted RT quantiles (10/30/50/70/90%) with 90% interval", handles, ncol=5)
    _save(fig, os.path.join(out, "RL_DDM_rt_quintiles.png"))

    # 2. Learning curve: mean RT over 8 trial bins, rows = cell, cols = cue
    fig, axes = plt.subplots(len(CELLS), N_CUES, figsize=(13, 12.5), sharex=True, sharey=True, squeeze=False)
    x = np.arange(1, n_bins + 1)
    for i, ((*_, key, label), color) in enumerate(zip(CELLS, CELL_COLORS)):
        for c, name in CUE_LABELS.items():
            ax = axes[i, c]
            o, lo, mid, hi = [], [], [], []
            for b in range(n_bins):
                m = (cue == c) & (cell == key) & (bins == b)
                if m.sum() == 0:
                    o.append(np.nan); lo.append(np.nan); mid.append(np.nan); hi.append(np.nan)
                    continue
                o.append(obs[m].mean())
                dm = np.nanmean(pred[:, m], axis=1)
                lo.append(np.percentile(dm, 5)); mid.append(np.median(dm)); hi.append(np.percentile(dm, 95))
            ax.fill_between(x, lo, hi, color=color, alpha=0.25, lw=0)
            ax.plot(x, mid, color=color, lw=2)
            ax.plot(x, o, "o-", color=sp.INK, lw=1, markersize=4.5, markeredgecolor="white")
            if i == 0:
                ax.set_title(f"{name} reward", loc="left")
            if c == 0:
                ax.set_ylabel(f"{label}\nMean RT (s)")
            if i == len(CELLS) - 1:
                ax.set_xlabel(f"Trial bin (session in {n_bins})")
    handles = [Line2D([], [], color=sp.INK, marker="o", lw=1, label="Observed"),
               Line2D([], [], color=sp.PRED, lw=2, label="Predicted median (cell color)"),
               Patch(color=sp.PRED, alpha=0.25, label="Predicted 90% interval")]
    sp.header(fig, "Learning curves: mean RT across the session, by cue and session x drug cell", handles, ncol=3)
    _save(fig, os.path.join(out, "RL_DDM_learning_curve.png"))

    # 3. HSSM's own predictive plot by cue (valid trials only)
    try:
        hssm.plotting.plot_predictive(model, dt=pp_dt, data=data, col="cue_type",
                                      title="RL-DDM: Posterior Predictive")
        plt.savefig(os.path.join(out, "RL_DDM_hssm_predictive.png"), dpi=200, bbox_inches="tight")
    except Exception as e:
        print(f"  WARNING: hssm.plotting.plot_predictive failed: {e}")
    plt.close("all")
    print("  Saved PPC quintile, learning-curve and HSSM predictive plots.")


# ---------------------------------------------------------------------------
# Plots: subject level (per subject x session unit)
# ---------------------------------------------------------------------------

PAGE = 36  # units per page in the per-unit grids (6 x 6)


def unit_keys(units):
    """Display labels: subject number in data order plus session and arm, e.g. 'S12 s2 Bup'."""
    subj_num = {s: k + 1 for k, s in enumerate(pd.unique(units["subject"]))}
    return [f"S{subj_num[s]} {CELL_SHORT[c]}" for s, c in zip(units["subject"], units["cell"])]


def plot_unit_mean_rt(met, units, out):
    """Observed vs predicted mean RT per unit, colored by cell."""
    fig, ax = plt.subplots(figsize=(6, 6.4))
    m = met.loc[units[UNIT_COL]]
    lim = (min(m["obs_mean_rt"].min(), m["pred_mean_rt"].min()) - 0.02,
           max(m["obs_mean_rt"].max(), m["pred_mean_rt"].max()) + 0.02)
    ax.plot(lim, lim, color=sp.MUTED, ls="--", lw=1)
    for (*_, key, label), color in zip(CELLS, CELL_COLORS):
        sel = (units["cell"] == key).values
        ax.plot(m["obs_mean_rt"].values[sel], m["pred_mean_rt"].values[sel], "o", color=color,
                markersize=7, markeredgecolor="white", markeredgewidth=0.8, label=f"{label} (n={sel.sum()})")
    r = np.corrcoef(m["obs_mean_rt"], m["pred_mean_rt"])[0, 1]
    ax.text(0.03, 0.97, f"r = {r:.3f}\nRMSE = {1000 * np.sqrt(np.mean((m['obs_mean_rt'] - m['pred_mean_rt'])**2)):.0f} ms",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("Observed mean RT (s)"); ax.set_ylabel("Predicted mean RT (s)")
    ax.legend(loc="lower right", fontsize=8)
    sp.header(fig, "Mean RT per subject x session")
    _save(fig, os.path.join(out, "subject_mean_rt_by_condition.png"))


def plot_unit_params(posterior, units, keys, out):
    """Per-unit parameter estimates on the natural scale (median & 90%), colored by cell."""
    vals = unit_params(posterior, units)
    n = len(units)
    color = dict(zip([k for *_, k, _ in CELLS], CELL_COLORS))
    fig, axes = plt.subplots(1, len(FREE_PARAMS), figsize=(14, 0.16 * n + 1.9), sharey=True)
    y = np.arange(n)[::-1]
    cols = [color[c] for c in units["cell"]]
    for ax, p in zip(axes, FREE_PARAMS):
        v = vals[p].reshape(-1, n)
        lo, mid, hi = np.percentile(v, [5, 50, 95], axis=0)
        ax.hlines(y, lo, hi, colors=cols, lw=1.1)
        ax.scatter(mid, y, c=cols, s=12, zorder=3)
        ax.set_title(PARAM_LABELS[p], loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, keys, fontsize=5.5)
    axes[0].set_ylim(-0.7, n - 0.3)
    handles = [Line2D([], [], color=c, marker="o", lw=1.1, label=lab) for (*_, lab), c in zip(CELLS, CELL_COLORS)]
    sp.header(fig, "Parameters per subject x session (natural scale, median & 90% interval)", handles, ncol=4)
    _save(fig, os.path.join(out, "subject_params.png"))


def plot_subject_level(idata, df, out):
    """Subject-level fit: per-unit RT densities and cue means (paged), accuracy summary,
    per-unit mean RT, per-unit parameters, and a metrics CSV."""
    os.makedirs(out, exist_ok=True)
    _, data, pred = valid_predictions(idata, df)
    units = unit_table(df)
    keys = unit_keys(units)
    ulist = list(units[UNIT_COL])
    met, q, cdf = sp.per_subject_stats(data, pred, ulist, unit_col=UNIT_COL, keys=keys)
    met = met.join(units.set_index(UNIT_COL)[["subject", "session", "treatment", "cell"]])
    met.round(4).to_csv(os.path.join(out, "subject_ppc_metrics.csv"))
    label = GROUP_LABEL
    n_pages = int(np.ceil(len(ulist) / PAGE))
    for i in range(n_pages):
        sl = slice(i * PAGE, (i + 1) * PAGE)
        extra = f" (page {i + 1}/{n_pages})" if n_pages > 1 else ""
        sp.plot_rt_density(data, pred, ulist[sl], met, label, out, unit_col=UNIT_COL, keys=keys[sl],
                           fname=f"subject_rt_density_p{i + 1}.png", ncols=6, title_extra=extra)
        sp.plot_cue_means(cdf, ulist[sl], label, out, keys=keys[sl],
                          fname=f"subject_cue_means_p{i + 1}.png", ncols=6, title_extra=extra)
    sp.plot_accuracy_summary(q, cdf, label, out)
    plot_unit_mean_rt(met, units, out)
    plot_unit_params(idata["posterior"].to_dataset(), units, keys, out)
    print(f"  Subject level: quantiles inside 90% PI {q['inside'].mean():.0%}, "
          f"cue means inside 90% PI {cdf['inside'].mean():.0%}; saved to {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_data(path=DATA_PATH, smoke=0):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run: cd utils && python preprocessing.py")
    df = pd.read_csv(path)
    required = [UNIT_COL, "subject", "session", "session2", "treatment", "trial_idx", "cue_idx",
                "cue_type", "feedback", "valid", "rt", "response"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Data is missing columns {missing}: re-run utils/preprocessing.py")
    if smoke:
        # Spread the smoke subjects over the arm patterns so every cell is present
        pattern = df.groupby("subject").apply(
            lambda g: tuple(g.drop_duplicates("session").sort_values("session")["treatment"]),
            include_groups=False)
        groups = [list(v) for _, v in pattern.groupby(pattern).groups.items()]
        chosen = []
        while len(chosen) < smoke and any(groups):
            for g in groups:
                if g and len(chosen) < smoke:
                    chosen.append(g.pop(0))
        df = df[df["subject"].isin(chosen)]
    # RLSSM reads each unit's learning sequence in row order
    return df.sort_values(["subject", "session", "trial_idx"], kind="stable").reset_index(drop=True)


def output_dir_for(smoke):
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = os.path.join(root, "outputs", f"{RUN_NAME}{'_smoke' if smoke else ''}_{date_str}")
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        path = f"{base}_Job{job_id}"
    else:
        k = 1
        while os.path.exists(f"{base}_v{k}"):
            k += 1
        path = f"{base}_v{k}"
    os.makedirs(os.path.join(path, "models"), exist_ok=True)
    return path


def effects_table(nat):
    rows = []
    for p in COVARIATE_PARAMS:
        for e in ("drug_effect", "session_effect"):
            v = nat[f"{p}_{e}"].values.ravel()
            rows.append(dict(param=p, effect=e, median=np.median(v), mean=v.mean(),
                             eti90_lb=np.percentile(v, 5), eti90_ub=np.percentile(v, 95),
                             p_gt_0=np.mean(v > 0)))
    return pd.DataFrame(rows)


def main():
    smoke = int(os.environ.get("RL_DDM_SMOKE", "0"))
    df = load_data(smoke=smoke)
    units = unit_table(df)
    print(f"Loaded {len(df)} trials: {df['subject'].nunique()} subjects, {len(units)} units "
          f"({(df['valid'] == 0).sum()} masked trials).")
    print("Units per cell: " + ", ".join(f"{k} {n}" for k, n in units["cell"].value_counts().items()))

    if smoke:
        model = fit(df, tune=int(os.environ.get("RL_DDM_TUNE", "300")),
                    draws=int(os.environ.get("RL_DDM_DRAWS", "200")),
                    chains=int(os.environ.get("RL_DDM_CHAINS", "1")))
    else:
        model = fit(df)
    idata = model.traces
    out = output_dir_for(smoke)

    print("\nSimulating posterior predictive...")
    add_posterior_predictive(idata, simulate_ppc(idata["posterior"].to_dataset(), df, draws_per_chain=50))
    trace_path = os.path.join(out, "models", TRACE_NAME)
    idata.to_netcdf(trace_path)
    print(f"Saved trace to {trace_path}")

    with open(os.path.join(out, "free_variables.txt"), "w") as f:
        f.write(f"=== Session x drug RL-DDM free RVs ({GROUP_LABEL}) ===\n")
        for rv in model.pymc_model.free_RVs:
            f.write(f"  {rv}\n")

    posterior = idata["posterior"].to_dataset()
    nat = natural_summary(posterior)
    sampled = [v for v in posterior.data_vars
               if v.endswith("_Intercept") or v.endswith("_sigma") or any(v.endswith(f"_{c}") for c in COVARIATES)]
    summary = pd.concat([az.summary(idata, var_names=sampled),
                         az.summary(xr.DataTree.from_dict({"posterior": nat}))])
    summary.to_csv(os.path.join(out, "summary.csv"))
    eff = effects_table(nat)
    eff.to_csv(os.path.join(out, "effects.csv"), index=False)
    print(summary)
    print("\nNatural-scale effects (90% interval):\n" + eff.round(4).to_string(index=False))
    ss = idata["sample_stats"]
    print(f"Step size: {float(ss['step_size'].mean()):.4g}   "
          f"Divergent: {float(ss['diverging'].mean()):.1%}   "
          f"Mean tree depth: {float(ss['tree_depth'].mean()):.2f}   "
          f"Max sampled-scale R-hat: {summary.loc[sampled, 'r_hat'].max():.3f}")

    plots = os.path.join(out, "plots")
    os.makedirs(plots, exist_ok=True)
    print("\nGenerating plots...")
    plot_conditions(nat, plots)
    plot_effects(nat, plots)
    plot_trace_and_pairs(nat, plots)
    plot_ppc(model, idata, df, plots)
    plot_subject_level(idata, df, os.path.join(plots, "subject_ppc"))
    print(f"\nAll outputs saved to: {out}")


def replot(run_dir):
    """Regenerates every plot of a finished run from its saved trace (no refit):
    python rl_ddm_all.py --replot ../outputs/<run>"""
    idata = az.from_netcdf(os.path.join(run_dir, "models", TRACE_NAME))
    df = load_data()
    fitted = set(str(s) for s in idata["posterior"]["subject__factor_dim"].values)
    df = df[df["subject"].astype(str).isin(fitted)].reset_index(drop=True)
    plots = os.path.join(run_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    nat = natural_summary(idata["posterior"].to_dataset())
    plot_conditions(nat, plots)
    plot_effects(nat, plots)
    plot_trace_and_pairs(nat, plots)
    model = build_model(df)
    plot_ppc(model, idata, df, plots)
    plot_subject_level(idata, df, os.path.join(plots, "subject_ppc"))


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--replot":
        replot(sys.argv[2].rstrip("/"))
    else:
        main()
