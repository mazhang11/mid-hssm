"""
rl_cue_scaling.py

Shared machinery for the cue-scaling variants of the analytical RL-DDM (rl_ddm.py), fit on
Session 1 controls:

  rl_cl.py   log scaling:   value(cue) = log(1 + $) / log(6)              (fixed)
  rl_ce.py   power scaling: value(cue) = ($ / 5) ** gamma                 (gamma fitted)

Everything except the reward value a hit delivers to the learner is the rl_ddm model: same
Rescorla-Wagner learner, masked analytical DDM, priors, logit links and posterior-predictive
simulation. Each model file defines its learner, build_model and a Spec; this module
does the fitting, saving and plotting for any Spec.

Plots (all parameters on the natural scale), written to <run>/plots/:
  <PFX>_trace.png, <PFX>_marginals.png      group-level (typical subject) posteriors
  <PFX>_tradeoffs_natural.png               pair plot, styled like RL_DDM_tradeoffs_natural
  RL_DDM_rt_quintiles.png, RL_DDM_learning_curve.png, RL_DDM_hssm_predictive.png   (rl_ddm.plot_ppc)
  subject_ppc/subject_rt_density.png        per subject: observed RT histogram vs predicted density
  subject_ppc/subject_cue_means.png         per subject: observed vs predicted mean RT per cue
  param_split/ppc_rt_dist_<p>_split.png     median split on rl_alpha and on scaler: 2 x 4 panels
                                            (Low/High x cue), observed vs predicted RT distribution
  drift_trajectories/drift_*_subjects_p*.png per-subject implied drift for every cue, trial by trial
  (<PFX> = RL_CL or RL_CE)

Usage (from models/):
  python rl_cl.py                         fit, save, plot
  RL_DDM_SMOKE=5 python rl_cl.py          5 subjects, short chain
  python rl_cl.py ../outputs/<run>        re-make every plot from a saved trace (no refit)
"""

import os
import sys
import datetime
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import rl_ddm  # registers the masked analytical DDM decision process; sets floatX = float32
import plot_subject_ppc as sp
import plot_tradeoffs_natural as ptn
import plot_drift_trajectories as pdt
from plot_param_split_trials import frame_bins, style_row_labels

N_CUES = rl_ddm.N_CUES
CUES = rl_ddm.CUE_LABELS
# Monetary value of a hit on each cue, indexed by cue_idx (neutral, small "Quarters", medium
# "Dollar", large "FiveDollars"), as in preprocessing's cue_value
DOLLARS = np.array([0.0, 0.5, 1.0, 5.0])

DATA_PATH = "../data/cleaned_rl_model_controls.csv"
GROUP_LABEL = "Session 1, Controls"
SPLITS = [("rl_alpha", "Learning rate"), ("scaler", "Value-to-drift scaling")]
GROUPS = ["Low", "High"]
# rl_ddm's 10-29 fit diverged 7.9% of the time at 0.9 on the rl_alpha x scaler ridge (FAQ.md),
# so these runs default to 0.95 like rl_ddm_all. Override: RL_DDM_TARGET_ACCEPT.
TARGET_ACCEPT = 0.95


@dataclass
class Spec:
    run_name: str                  # output prefix and trace name: outputs/<run_name>_<date>_...
    title: str                     # short model description for plot titles
    free_params: list
    bounds: dict
    logit_params: list
    cue_values: Callable           # {p: (S,) natural-scale draws} -> (S, 4) value of a hit per cue
    build_model: Callable          # df -> hssm.RLSSM
    prepare: Callable = field(default=lambda df: df)   # adds the columns the learner reads

    @property
    def trace_name(self):
        return f"{self.run_name}_model_trace.nc"

    @property
    def prefix(self):
        return self.run_name.upper()

    @property
    def label(self):
        return f"{GROUP_LABEL} · {self.title}"


# ---------------------------------------------------------------------------
# Natural-scale parameters
# ---------------------------------------------------------------------------

def to_natural(spec, p, x):
    """Linear-predictor value -> parameter value (inverse gen_logit for logit-linked params)."""
    if p in spec.logit_params:
        lower, upper = spec.bounds[p]
        return lower + (upper - lower) / (1.0 + np.exp(-x))
    return x


def group_params_natural(spec, posterior):
    """Group-level value of each parameter (a subject with offset 0) on its natural scale."""
    return xr.Dataset({f"{p}_group": to_natural(spec, p, posterior[f"{p}_Intercept"])
                       for p in spec.free_params})


def subject_params(spec, posterior, subjects):
    """{param: (chain, draw, n_subjects)} natural-scale subject values, clipped to the bounds."""
    out = {}
    for p in spec.free_params:
        offsets = posterior[f"{p}_1|subject"]
        subj_dim = [d for d in offsets.dims if d not in ("chain", "draw")][0]
        offsets = offsets.sel({subj_dim: list(subjects)})
        values = (posterior[f"{p}_Intercept"] + offsets).transpose("chain", "draw", subj_dim).values
        out[p] = np.clip(to_natural(spec, p, values), *spec.bounds[p])
    return out


def flat_draws(params):
    """{p: (chain, draw, units)} -> {p: (samples, units)}."""
    return {p: v.reshape(-1, v.shape[-1]) for p, v in params.items()}


# ---------------------------------------------------------------------------
# The learner in NumPy (twin of the model file's JAX learner), vectorized over draws
# ---------------------------------------------------------------------------

def drift_all_cues(spec, params, cue, hit):
    """
    One subject. params: {p: (S,)} natural-scale draws; cue, hit: (n_trials,).
    Returns (S, n_trials, 4): the drift each cue would get on trial t, using Q before trial t's
    feedback. feedback_t = hit_t * value[cue_t], with value from spec.cue_values.
    """
    value = spec.cue_values(params)                     # (S, 4)
    alpha, scaler, v0 = params["rl_alpha"], params["scaler"], params["v0"]
    q = np.zeros((len(alpha), N_CUES))
    v = np.empty((len(alpha), len(cue), N_CUES))
    for i, (c, h) in enumerate(zip(cue, hit)):
        v[:, i, :] = v0[:, None] + scaler[:, None] * q
        q[:, c] += alpha * (h * value[:, c] - q[:, c])
    return v


def trial_drift(v_all, cue):
    """(S, n_trials, 4) -> (S, n_trials) drift of the cue actually shown."""
    return np.take_along_axis(v_all, cue[None, :, None], axis=2)[..., 0]


def subsample(posterior, draws_per_chain, seed):
    rng = np.random.default_rng(seed)
    n = posterior.sizes["draw"]
    return posterior.isel(draw=np.sort(rng.choice(n, min(draws_per_chain, n), replace=False)))


# ---------------------------------------------------------------------------
# Posterior predictive (by hand, as rl_ddm.simulate_ppc)
# ---------------------------------------------------------------------------

def simulate_ppc(spec, posterior, df, draws_per_chain=50, seed=0):
    """One simulated (rt, response) per trial and draw, in HSSM's posterior_predictive format."""
    from ssms.basic_simulators.simulator import simulator

    rng = np.random.default_rng(seed)
    sub = subsample(posterior, draws_per_chain, seed)
    subjects = pd.unique(df["subject"])
    params = flat_draws(subject_params(spec, sub, subjects))
    n_chain, n_draw = sub.sizes["chain"], sub.sizes["draw"]
    S = n_chain * n_draw
    cue, hit = df["cue_idx"].values.astype(int), df["hit"].values.astype(float)

    theta = np.empty((S, len(df), 4))
    for j, s in enumerate(subjects):
        idx = np.flatnonzero(df["subject"].values == s)
        pj = {p: v[:, j] for p, v in params.items()}
        theta[:, idx, 0] = trial_drift(drift_all_cues(spec, pj, cue[idx], hit[idx]), cue[idx])
        theta[:, idx, 1] = pj["a"][:, None]
        theta[:, idx, 2] = 0.5
        theta[:, idx, 3] = pj["t"][:, None]

    out = np.empty((S, len(df), 2), dtype=np.float32)
    for k in range(S):
        sim = simulator(theta=theta[k], model="ddm", n_samples=1,
                        random_state=int(rng.integers(2**31 - 1)))
        out[k, :, 0] = sim["rts"].ravel()
        out[k, :, 1] = sim["choices"].ravel()

    return xr.DataArray(
        out.reshape(n_chain, n_draw, len(df), 2),
        dims=("chain", "draw", "__obs__", "rt,response_dim"),
        coords={"chain": sub["chain"].values, "draw": sub["draw"].values,
                "__obs__": np.arange(len(df)), "rt,response_dim": [0, 1]},
        name="rt,response",
    )


def valid_predictions(idata, df):
    """Valid trials only: (df_valid, pred (samples, n_valid) RTs, NaN for lower-boundary hits)."""
    pp = idata["posterior_predictive"]["rt,response"].values
    pp = pp.reshape(-1, pp.shape[-2], 2)
    if pp.shape[1] != len(df):
        raise ValueError(f"Trace has {pp.shape[1]} trials but the data has {len(df)}")
    pred = np.where(pp[..., 1] > 0, pp[..., 0], np.nan)
    valid = df["valid"].values == 1
    return df.loc[valid].reset_index(drop=True), pred[:, valid]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_diagnostics(spec, idata, out):
    """Trace, marginals and tradeoff pair plot of the group-level parameters, natural scale."""
    os.makedirs(out, exist_ok=True)
    prefix = os.path.join(out, spec.prefix)
    nat = group_params_natural(spec, idata["posterior"].to_dataset())
    natural = xr.DataTree.from_dict({"posterior": nat})
    az.plot_trace_dist(natural).savefig(f"{prefix}_trace.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    az.plot_dist(natural).savefig(f"{prefix}_marginals.png", dpi=300, bbox_inches="tight")
    plt.close("all")
    draws = {p: nat[f"{p}_group"].values.ravel() for p in spec.free_params}
    ptn.pair_plot(draws, {p: ptn.padded_range(draws[p], spec.bounds[p]) for p in draws},
                  f"{spec.prefix}: parameter tradeoffs, group level (natural scale) — {spec.label}",
                  f"{prefix}_tradeoffs_natural.png")


def plot_subject_ppc(spec, dv, pred, out):
    """Per-subject observed RT histogram vs predicted density, and per-cue mean RTs."""
    os.makedirs(out, exist_ok=True)
    subjects = list(pd.unique(dv["subject"]))
    met, q, cdf = sp.per_subject_stats(dv, pred, subjects)
    met.round(4).to_csv(os.path.join(out, "subject_ppc_metrics.csv"))
    pd.DataFrame({"key": [f"S{k + 1}" for k in range(len(subjects))], "subject": subjects}) \
        .to_csv(os.path.join(out, "subject_key.csv"), index=False)
    sp.plot_rt_density(dv, pred, subjects, met, spec.label, out)
    sp.plot_cue_means(cdf, subjects, spec.label, out)
    print(f"  Subject PPC: quantiles inside 90% PI {q['inside'].mean():.0%}, "
          f"cue means inside 90% PI {cdf['inside'].mean():.0%}")


def split_table(spec, posterior, subjects):
    """Per-subject posterior medians (natural scale) and Low/High labels for each split."""
    params = flat_draws(subject_params(spec, posterior, subjects))
    med = pd.DataFrame({p: np.median(v, axis=0) for p, v in params.items()}, index=subjects)
    med.insert(0, "key", [f"S{k + 1}" for k in range(len(subjects))])
    for p, _ in SPLITS:
        med[f"{p}_group"] = np.where(med[p] > med[p].median(), "High", "Low")
    return med


def plot_split_ppc(spec, dv, pred, med, p, pname, out):
    """2 x 4: Low/High group on `p` x cue. Observed RT histogram (valid trials pooled over the
    group's subjects, one bin per 60 Hz frame) vs the model's predicted density (median, 90%)."""
    rt_ms, pred_ms = 1000 * dv["rt"].values, 1000 * pred
    bins = frame_bins(rt_ms)
    widths, centers = np.diff(bins), 0.5 * (bins[1:] + bins[:-1])
    cue = dv["cue_idx"].values
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.4), sharex=True, sharey=True)
    for r, g in enumerate(GROUPS):
        in_group = dv["subject"].isin(med.index[med[f"{p}_group"] == g]).values
        for c, name in CUES.items():
            ax = axes[r, c]
            m = in_group & (cue == c)
            obs = rt_ms[m]
            ax.bar(bins[:-1], np.histogram(obs, bins)[0] / (len(obs) * widths), width=widths,
                   align="edge", color=sp.OBS_FILL, edgecolor="white", linewidth=0.6)
            # Density per draw over that draw's upper-boundary RTs, normalized like the observed
            dens = []
            for x in pred_ms[:, m]:
                x = x[~np.isnan(x)]
                dens.append(np.histogram(x, bins)[0] / (len(x) * widths))
            lo, mid, hi = np.percentile(dens, [sp.CI[0], 50, sp.CI[1]], axis=0)
            ax.fill_between(centers, lo, hi, color=sp.PRED_BAND, alpha=0.6, lw=0)
            ax.plot(centers, mid, color=sp.PRED, lw=1.6)
            pred_med = np.nanmedian(pred_ms[:, m], axis=1)
            ax.axvline(np.median(obs), color=sp.INK, ls="--", lw=1.2)
            ax.axvline(np.median(pred_med), color=sp.PRED, ls=":", lw=1.4)
            ax.text(0.97, 0.95,
                    f"median obs {np.median(obs):.0f} ms\n"
                    f"median pred {np.median(pred_med):.0f} ms\n"
                    f"  90% [{np.percentile(pred_med, sp.CI[0]):.0f}, {np.percentile(pred_med, sp.CI[1]):.0f}]\n"
                    f"{len(obs)} trials, {np.mean(obs > bins[-1]) * 100:.1f}% > {bins[-1]:.0f} ms",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=sp.INK_2)
            if r == 0:
                ax.set_title(f"{name} cue", loc="left", color=sp.CUE_COLORS[c], fontweight="bold")
            else:
                ax.set_xlabel("RT (ms)")
            if c == 0:
                ax.set_ylabel("Density")
            ax.set_xlim(bins[0], bins[-1])
    style_row_labels(axes, p, pname, med)
    handles = [Patch(color=sp.OBS_FILL, label="Observed RTs (valid trials, group pooled; 1 bin per 60 Hz frame)"),
               Line2D([], [], color=sp.PRED, lw=1.6, label="Model prediction (median)"),
               Patch(color=sp.PRED_BAND, alpha=0.6, label="90% predictive interval"),
               Line2D([], [], color=sp.INK, ls="--", lw=1.2, label="Observed median"),
               Line2D([], [], color=sp.PRED, ls=":", lw=1.4, label="Predicted median")]
    sp.header(fig, f"Posterior predictive RT distribution by cue, median split on {pname.lower()} "
                   f"— {spec.label}", handles, ncol=5)
    fig.subplots_adjust(left=0.08)
    path = os.path.join(out, f"ppc_rt_dist_{p}_split.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_splits(spec, idata, dv, pred, out):
    os.makedirs(out, exist_ok=True)
    med = split_table(spec, idata["posterior"].to_dataset(), list(pd.unique(dv["subject"])))
    med.round(4).to_csv(os.path.join(out, "param_split_subjects.csv"))
    for p, pname in SPLITS:
        plot_split_ppc(spec, dv, pred, med, p, pname, out)
        print(f"  Split on {p}: cut at {med[p].median():.3g} "
              f"(range {med[p].min():.3g}–{med[p].max():.3g})")


def plot_drift(spec, idata, df, out, draws_per_chain=50, seed=0):
    """Per-subject implied drift for every cue at every trial (plot_drift_trajectories layout)."""
    os.makedirs(out, exist_ok=True)
    subjects = list(pd.unique(df["subject"]))
    params = flat_draws(subject_params(spec, subsample(idata["posterior"].to_dataset(),
                                                       draws_per_chain, seed), subjects))
    d = df.assign(unit=df["subject"])
    v = np.stack([drift_all_cues(spec, {p: x[:, j] for p, x in params.items()},
                                 d.loc[d["unit"] == s, "cue_idx"].values.astype(int),
                                 d.loc[d["unit"] == s, "hit"].values.astype(float))
                  for j, s in enumerate(subjects)])            # (subjects, samples, trials, cues)
    units = pd.DataFrame({"unit": subjects, "subject": subjects,
                          "key": [f"S{k + 1}" for k in range(len(subjects))]})
    pdt.plot_group(v, units, d, spec.label, False, out)
    pdt.plot_subjects(v, units, d, spec.label, out)
    pdt.plot_subjects(v, units, d, spec.label, out, advantage=True)
    med = np.median(v, axis=1)
    n_tr = v.shape[2]
    pd.DataFrame({
        "key": np.repeat(units["key"].values, n_tr * N_CUES),
        "subject": np.repeat(subjects, n_tr * N_CUES),
        "trial": np.tile(np.repeat(np.arange(1, n_tr + 1), N_CUES), len(subjects)),
        "cue": np.tile([CUES[c] for c in CUES], len(subjects) * n_tr),
        "drift_median": med.reshape(-1),
    }).round(4).to_csv(os.path.join(out, "drift_trajectories_median.csv"), index=False)


def make_plots(spec, idata, df, plots_dir, model=None):
    print("\nGenerating plots...")
    plot_diagnostics(spec, idata, plots_dir)
    rl_ddm.plot_ppc(model, idata, df, plots_dir)        # quintiles, learning curve (+ HSSM's plot if model)
    dv, pred = valid_predictions(idata, df)
    plot_subject_ppc(spec, dv, pred, os.path.join(plots_dir, "subject_ppc"))
    plot_splits(spec, idata, dv, pred, os.path.join(plots_dir, "param_split"))
    plot_drift(spec, idata, df, os.path.join(plots_dir, "drift_trajectories"))
    print(f"Plots saved to {plots_dir}")


# ---------------------------------------------------------------------------
# Data, fitting, outputs
# ---------------------------------------------------------------------------

def load_data(spec, path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at '{path}'. Run: cd utils && python preprocessing.py")
    df = pd.read_csv(path)
    required = ["subject", "trial_idx", "cue_idx", "cue_type", "hit", "valid", "rt", "response"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Data is missing columns {missing}: re-run utils/preprocessing.py")
    # RLSSM reads the learning sequence in row order
    df = df.sort_values(["subject", "trial_idx"], kind="stable").reset_index(drop=True)
    return spec.prepare(df)


def output_dir_for(spec, smoke):
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = os.path.join(root, "outputs", f"{spec.run_name}{'_smoke' if smoke else ''}_{date_str}")
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        out = f"{base}_Job{job_id}"
    else:
        k = 1
        while os.path.exists(f"{base}_v{k}"):
            k += 1
        out = f"{base}_v{k}"
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    return out


def write_summary(spec, model, idata, output_dir):
    with open(os.path.join(output_dir, "free_variables.txt"), "w") as f:
        f.write(f"=== {spec.prefix} free RVs ({spec.label}) ===\n")
        for rv in model.pymc_model.free_RVs:
            f.write(f"  {rv}\n")
    post = idata["posterior"].to_dataset()
    # Sampled-scale rows for convergence, plus natural-scale <param>_group rows
    summary = pd.concat([
        az.summary(idata, var_names=[f"{p}_Intercept" for p in spec.free_params]
                   + [f"{p}_1|subject_sigma" for p in spec.free_params]),
        az.summary(xr.DataTree.from_dict({"posterior": group_params_natural(spec, post)})),
    ])
    summary.to_csv(os.path.join(output_dir, "summary.csv"))
    print(summary)
    ss = idata["sample_stats"]
    print(f"Step size: {float(ss['step_size'].mean()):.4g}   "
          f"Divergent: {float(ss['diverging'].mean()):.1%}   "
          f"Mean tree depth: {float(ss['tree_depth'].mean()):.2f}")


def run(spec):
    df = load_data(spec)
    smoke = int(os.environ.get("RL_DDM_SMOKE", "0"))
    if smoke:
        df = df[df["subject"].isin(pd.unique(df["subject"])[:smoke])].reset_index(drop=True)
    print(f"Loaded {len(df)} trials from {df['subject'].nunique()} subjects "
          f"({(df['valid'] == 0).sum()} masked).")
    print("Cue values at the group start (neutral, small, medium, large): "
          + ", ".join(f"{x:.3f}" for x in spec.cue_values(
              {p: np.array([1.0]) for p in spec.free_params})[0]))

    tune = int(os.environ.get("RL_DDM_TUNE", "300" if smoke else "1000"))
    draws = int(os.environ.get("RL_DDM_DRAWS", "200" if smoke else "1000"))
    chains = int(os.environ.get("RL_DDM_CHAINS", "1" if smoke else "4"))
    target_accept = float(os.environ.get("RL_DDM_TARGET_ACCEPT", TARGET_ACCEPT))
    print("\n" + "=" * 60)
    print(f" Fitting {spec.prefix}: RL-DDM, {spec.title} ({GROUP_LABEL})")
    print(f" {chains} chains x {tune} tune / {draws} draws, target_accept {target_accept}")
    print("=" * 60)
    model = spec.build_model(df)
    model.sample(sampler="numpyro", tune=tune, draws=draws, chains=chains, cores=chains,
                 target_accept=target_accept)
    idata = model.traces

    output_dir = output_dir_for(spec, smoke)
    print("\nSimulating posterior predictive...")
    rl_ddm.add_posterior_predictive(idata, simulate_ppc(spec, idata["posterior"].to_dataset(), df))
    trace_path = os.path.join(output_dir, "models", spec.trace_name)
    idata.to_netcdf(trace_path)
    print(f"Saved trace to {trace_path}")
    write_summary(spec, model, idata, output_dir)
    make_plots(spec, idata, df, os.path.join(output_dir, "plots"), model=model)
    print(f"\nAll outputs saved to: {output_dir}")


def replot(spec, run_dir):
    """Re-make every plot from a saved run (HSSM's own predictive plot is skipped: no model)."""
    idata = az.from_netcdf(os.path.join(run_dir, "models", spec.trace_name))
    df = load_data(spec)
    fitted = set(str(s) for s in idata["posterior"]["subject__factor_dim"].values)
    df = df[df["subject"].astype(str).isin(fitted)].reset_index(drop=True)
    make_plots(spec, idata, df, os.path.join(run_dir, "plots"))


def main(spec):
    if len(sys.argv) > 1:
        replot(spec, sys.argv[1].rstrip("/"))
    else:
        run(spec)


# ---------------------------------------------------------------------------
# The linear baseline (rl_ddm.py) as a Spec, so its saved runs get the same plots
# ---------------------------------------------------------------------------

RL_DDM_SPEC = Spec(
    run_name="rl_ddm_analytical",          # trace: rl_ddm_analytical_model_trace.nc
    title="linear cue scaling (rl_ddm)",
    free_params=rl_ddm.FREE_PARAMS,
    bounds=rl_ddm.PARAM_BOUNDS,
    logit_params=rl_ddm.LOGIT_PARAMS,
    cue_values=lambda params: np.broadcast_to(DOLLARS / DOLLARS.max(), (len(params["rl_alpha"]), N_CUES)),
    build_model=rl_ddm.build_model,
)


if __name__ == "__main__":
    # python rl_cue_scaling.py ../outputs/<rl_ddm_analytical run>
    # Adds the median-split posterior predictive plots (param_split/ppc_rt_dist_<p>_split.png)
    # to a Session 1 controls rl_ddm run; its other plots come from rl_ddm.py and the plot_* scripts.
    if len(sys.argv) != 2 or "rl_ddm_analytical" not in sys.argv[1]:
        sys.exit("Usage: python rl_cue_scaling.py ../outputs/<rl_ddm_analytical run>")
    run_dir = sys.argv[1].rstrip("/")
    idata = az.from_netcdf(os.path.join(run_dir, "models", RL_DDM_SPEC.trace_name))
    df = load_data(RL_DDM_SPEC)
    fitted = set(str(s) for s in idata["posterior"]["subject__factor_dim"].values)
    df = df[df["subject"].astype(str).isin(fitted)].reset_index(drop=True)
    dv, pred = valid_predictions(idata, df)
    out = os.path.join(run_dir, "plots", "param_split")
    os.makedirs(out, exist_ok=True)
    med = split_table(RL_DDM_SPEC, idata["posterior"].to_dataset(), list(pd.unique(dv["subject"])))
    for p, pname in SPLITS:
        print("Saved", plot_split_ppc(RL_DDM_SPEC, dv, pred, med, p, pname, out))
