"""
plot_subject_ppc.py

Subject-wise posterior predictive checks for an analytical RL-DDM run (rl_ddm.py family),
made for presenting how well the model fits each subject. Reads the saved trace (which
holds the hand-simulated posterior_predictive group) and writes to <run>/plots/subject_ppc/:

  subject_rt_density.png       per subject: observed RT histogram vs predicted RT density
  subject_cue_means.png        per subject: observed vs predicted mean RT for each cue
  subject_accuracy_summary.png all subjects: observed vs predicted RT quantiles and cue means
  subject_params.png           per-subject parameter estimates on the NATURAL scale
  subject_ppc_metrics.csv      per-subject fit numbers; subject_key.csv maps S1.. to IDs

Only valid trials are used on both sides (masked trials have no observed RT), and simulated
lower-boundary hits, which this one-response task cannot produce, are dropped.

Usage: cd models && python evaluation/plot_subject_ppc.py ../outputs/<run_folder>
"""

import os
import sys
# models/ holds rl_ddm, rl_cl, rl_ce and rl_cue_scaling; make them importable from evaluation/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import rl_ddm

QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]
CI = (5, 95)  # 90% predictive interval

INK, INK_2, MUTED, GRID = "#1f1f1e", "#555552", "#8a8984", "#e4e3de"
OBS_FILL = "#c9c8c1"
PRED = "#2a78d6"
PRED_BAND = "#9ec5f4"
CUE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # categorical slots 1-4
Q_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # ordinal blue ramp
PARAM_LABELS = {"rl_alpha": "Learning rate (α)", "scaler": "Value → drift scaling",
                "v0": "Baseline drift (v0)", "a": "Boundary separation (a)",
                "t": "Non-decision time (t, s)", "gamma": "Reward exponent (γ)"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.facecolor": "white", "savefig.facecolor": "white",
})


def data_path_for(run_dir):
    here = os.path.dirname(os.path.abspath(__file__))
    for key in ["session2_bupropion", "session2_placebo", "session1_bupropion"]:
        if key in run_dir:
            return os.path.join(here, f"../../data/cleaned_rl_model_{key}.csv")
    return os.path.join(here, "../../data/cleaned_rl_model_controls.csv")


def group_label_for(run_dir):
    for key, label in [("session2_bupropion", "Session 2, Bupropion XL"),
                       ("session2_placebo", "Session 2, Placebo"),
                       ("session1_bupropion", "Session 1, Bupropion XL")]:
        if key in run_dir:
            return label
    return "Session 1, Controls (placebo)"


def load(run_dir):
    trace = os.path.join(run_dir, "models", "rl_ddm_analytical_model_trace.nc")
    dt = az.from_netcdf(trace)
    df = pd.read_csv(data_path_for(run_dir))
    df = df.sort_values(["subject", "trial_idx"], kind="stable").reset_index(drop=True)
    # Smoke runs fit only the first n subjects: keep the subjects the trace actually has
    fitted = [str(s) for s in dt["posterior"]["subject__factor_dim"].values]
    df = df[df["subject"].astype(str).isin(fitted)].reset_index(drop=True)
    pp = dt["posterior_predictive"]["rt,response"].values
    pp = pp.reshape(-1, pp.shape[-2], 2)          # (samples, trials, [rt, response])
    if pp.shape[1] != len(df):
        raise ValueError(f"Trace has {pp.shape[1]} trials but the data has {len(df)}: "
                         "is this a smoke run? Pass the matching data or subset.")
    pred = np.where(pp[..., 1] > 0, pp[..., 0], np.nan)   # drop lower-boundary hits
    valid = df["valid"].values == 1
    return dt, df.loc[valid].reset_index(drop=True), pred[:, valid]


def per_subject_stats(df, pred, subjects, unit_col="subject", keys=None):
    """Per-unit fit numbers. A unit is a subject (default) or, for rl_ddm_all, one subject's
    session (unit_col="unit"); `keys` are the display labels (default S1, S2, ...)."""
    keys = keys or [f"S{k+1}" for k in range(len(subjects))]
    rows, qrows, crows = [], [], []
    for k, s in enumerate(subjects):
        m = (df[unit_col] == s).values
        obs = df["rt"].values[m]
        p = pred[:, m]
        oq = np.quantile(obs, QUANTILES)
        pq = np.nanquantile(p, QUANTILES, axis=1)            # (5, samples)
        for i, q in enumerate(QUANTILES):
            lo, hi = np.percentile(pq[i], CI)
            qrows.append(dict(subject=s, key=keys[k], q=q, obs=oq[i], pred=np.median(pq[i]),
                              lo=lo, hi=hi, inside=lo <= oq[i] <= hi))
        cue = df["cue_idx"].values[m]
        for c in range(rl_ddm.N_CUES):
            cm = cue == c
            dm = np.nanmean(p[:, cm], axis=1)
            lo, hi = np.percentile(dm, CI)
            crows.append(dict(subject=s, key=keys[k], cue=c, obs=obs[cm].mean(),
                              pred=np.median(dm), lo=lo, hi=hi, n=int(cm.sum())))
        rows.append(dict(subject=s, key=keys[k], n_valid=int(m.sum()),
                         obs_mean_rt=obs.mean(), pred_mean_rt=np.nanmean(p),
                         quantile_rmse_ms=1000 * np.sqrt(np.mean((np.median(pq, 1) - oq) ** 2)),
                         lower_boundary_share=np.isnan(p).mean()))
    q = pd.DataFrame(qrows)
    cdf = pd.DataFrame(crows)
    cdf["inside"] = (cdf["obs"] >= cdf["lo"]) & (cdf["obs"] <= cdf["hi"])
    met = pd.DataFrame(rows).set_index("subject")
    met["quantiles_inside_90pi"] = q.groupby("subject")["inside"].sum().astype(int).astype(str) + "/5"
    met["cue_means_inside_90pi"] = cdf.groupby("subject")["inside"].sum().astype(int).astype(str) + "/4"
    return met, q, cdf


def header(fig, title, handles=None, ncol=3):
    """Title on the first line, legend on the second, both left-aligned; reserves ~0.85 in."""
    h = fig.get_figheight()
    fig.suptitle(title, x=0.01, y=1 - 0.12 / h, ha="left", va="top", fontsize=12, fontweight="bold")
    if handles:
        fig.legend(handles=handles, loc="upper left", ncol=ncol, bbox_to_anchor=(0.005, 1 - 0.42 / h))
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.85 if handles else 0.45) / h))


def grid(n, ncols, w=2.3, h=1.8):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows + 1.0),
                             sharex=True, squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    return fig, axes


def plot_rt_density(df, pred, subjects, met, label, out, unit_col="subject", keys=None,
                    fname="subject_rt_density.png", ncols=None, title_extra=""):
    lo_x = max(0.15, np.floor(np.quantile(df["rt"], 0.001) * 20) / 20)
    # Largest per-subject 99th percentile: slow subjects stay visible, a lone stray RT can't
    # stretch every panel (RTs beyond it are still counted in each density)
    hi_x = np.ceil(df.groupby("subject")["rt"].quantile(0.99).max() * 20) / 20
    bins = np.linspace(lo_x, hi_x, int(round((hi_x - lo_x) / 0.025)) + 1)   # 25 ms bins
    centers = 0.5 * (bins[1:] + bins[:-1])
    keys = keys or [f"S{k+1}" for k in range(len(subjects))]
    ncols = ncols or (6 if len(subjects) > 12 else 4)
    fig, axes = grid(len(subjects), ncols)
    for k, s in enumerate(subjects):
        ax = axes.ravel()[k]
        m = (df[unit_col] == s).values
        ax.hist(df["rt"].values[m], bins=bins, density=True, color=OBS_FILL, edgecolor="white",
                linewidth=0.6)
        dens = np.array([np.histogram(r[~np.isnan(r)], bins=bins, density=True)[0]
                         for r in pred[:, m]])
        lo, mid, hi = np.percentile(dens, [CI[0], 50, CI[1]], axis=0)
        ax.fill_between(centers, lo, hi, color=PRED_BAND, alpha=0.6, linewidth=0, step=None)
        ax.plot(centers, mid, color=PRED, linewidth=1.6)
        r = met.loc[s]
        ax.set_title(f"{keys[k]}   n={r.n_valid}   q in PI {r.quantiles_inside_90pi}",
                     loc="left", fontsize=7.5)
        ax.set_yticks([])
        ax.set_xlim(lo_x, hi_x)
        ax.grid(axis="y", visible=False)
    for ax in axes[-1]:
        ax.set_xlabel("RT (s)")
    for ax in axes.ravel()[len(subjects) - ncols:len(subjects)]:
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_xlabel("RT (s)")
    handles = [Patch(color=OBS_FILL, label="Observed RTs"),
               Line2D([], [], color=PRED, lw=1.6, label="Model prediction (median)"),
               Patch(color=PRED_BAND, alpha=0.6, label="90% predictive interval")]
    header(fig, f"RL-DDM fit per subject: RT distributions — {label}{title_extra}", handles)
    fig.savefig(os.path.join(out, fname), dpi=200)
    plt.close(fig)


def plot_cue_means(cdf, subjects, label, out, keys=None, fname="subject_cue_means.png",
                   ncols=None, title_extra=""):
    keys = keys or [f"S{k+1}" for k in range(len(subjects))]
    ncols = ncols or (6 if len(subjects) > 12 else 4)
    fig, axes = grid(len(subjects), ncols, h=1.9)
    x = np.arange(rl_ddm.N_CUES)
    for k, s in enumerate(subjects):
        ax = axes.ravel()[k]
        d = cdf[cdf["subject"] == s].sort_values("cue")
        ax.fill_between(x, d["lo"], d["hi"], color=PRED_BAND, alpha=0.6, linewidth=0)
        ax.plot(x, d["pred"], color=PRED, lw=1.6)
        ax.plot(x, d["obs"], "o", color=INK, markersize=4.5, markeredgecolor="white",
                markeredgewidth=0.8, zorder=3)
        ax.set_title(f"{keys[k]}   cues in PI {int(d['inside'].sum())}/4", loc="left", fontsize=7.5)
        ax.set_xticks(x, ["Neu", "Small", "Med", "Large"])
        ax.set_xlim(-0.3, rl_ddm.N_CUES - 0.7)
    for i in range(axes.shape[0]):
        axes[i, 0].set_ylabel("Mean RT (s)")
    for ax in axes.ravel()[max(0, len(subjects) - ncols):len(subjects)]:
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_xlabel("Cue (reward)")
    handles = [Line2D([], [], ls="", marker="o", color=INK, label="Observed mean RT"),
               Line2D([], [], color=PRED, lw=1.6, label="Model prediction (median)"),
               Patch(color=PRED_BAND, alpha=0.6, label="90% predictive interval")]
    header(fig, f"RL-DDM fit per subject: mean RT by cue — {label}{title_extra}", handles)
    fig.savefig(os.path.join(out, fname), dpi=200)
    plt.close(fig)


def plot_accuracy_summary(q, cdf, label, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.4))
    panels = [
        (axes[0], q, "q", QUANTILES, Q_COLORS, lambda v: f"{int(v*100)}th pct",
         "RT quantiles (one dot per subject × quantile)"),
        (axes[1], cdf, "cue", list(range(rl_ddm.N_CUES)), CUE_COLORS,
         lambda v: rl_ddm.CUE_LABELS[v], "Mean RT by cue (one dot per subject × cue)"),
    ]
    for ax, d, col, levels, colors, name, title in panels:
        lim = (min(d["obs"].min(), d["lo"].min()) - 0.02, max(d["obs"].max(), d["hi"].max()) + 0.02)
        ax.plot(lim, lim, color=MUTED, lw=1, ls="--", zorder=1)
        for v, c in zip(levels, colors):
            e = d[d[col] == v]
            ax.errorbar(e["obs"], e["pred"], yerr=[e["pred"] - e["lo"], e["hi"] - e["pred"]],
                        fmt="o", color=c, ecolor=c, elinewidth=0.8, alpha=0.9, markersize=5,
                        markeredgecolor="white", markeredgewidth=0.6, label=name(v), zorder=2)
        r = np.corrcoef(d["obs"], d["pred"])[0, 1]
        rmse = 1000 * np.sqrt(np.mean((d["obs"] - d["pred"]) ** 2))
        ax.text(0.03, 0.97, f"r = {r:.3f}\nRMSE = {rmse:.0f} ms\n"
                f"observed inside 90% PI: {d['inside'].mean():.0%}",
                transform=ax.transAxes, va="top", fontsize=9, color=INK)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel("Observed RT (s)")
        ax.set_ylabel("Predicted RT (s)")
        ax.set_title(title, loc="left")
        ax.legend(loc="lower right", fontsize=8)
    header(fig, f"RL-DDM accuracy across subjects — {label}")
    fig.savefig(os.path.join(out, "subject_accuracy_summary.png"), dpi=200)
    plt.close(fig)


def plot_params(dt, subjects, label, out):
    post = dt["posterior"].to_dataset()
    vals = rl_ddm.subject_params(post, subjects)              # natural scale (inverse logit)
    group = rl_ddm.group_params_natural(post)
    n = len(subjects)
    fig, axes = plt.subplots(1, len(rl_ddm.FREE_PARAMS), figsize=(13, 0.28 * n + 1.9), sharey=True)
    y = np.arange(n)[::-1]
    for ax, p in zip(axes, rl_ddm.FREE_PARAMS):
        v = vals[p].reshape(-1, n)
        lo, mid, hi = np.percentile(v, [5, 50, 95], axis=0)
        g = group[f"{p}_group"].values.ravel()
        ax.axvspan(*np.percentile(g, [5, 95]), color=GRID, alpha=0.8, lw=0)
        ax.axvline(np.median(g), color=MUTED, lw=1, ls="--")
        ax.hlines(y, lo, hi, color=PRED, lw=1.4)
        ax.plot(mid, y, "o", color=PRED, markersize=4)
        ax.set_title(PARAM_LABELS[p], loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [f"S{k+1}" for k in range(n)])
    axes[0].set_ylim(-0.7, n - 0.3)
    handles = [Line2D([], [], color=PRED, marker="o", lw=1.4, label="Subject: median & 90% interval"),
               Line2D([], [], color=MUTED, ls="--", label="Group (typical subject)")]
    header(fig, f"RL-DDM subject parameters (natural scale) — {label}", handles, ncol=2)
    fig.savefig(os.path.join(out, "subject_params.png"), dpi=200)
    plt.close(fig)


def main(run_dir):
    if "rl_ddm_all" in run_dir:
        # Session x drug model: per subject x session units, paged grids (rl_ddm_all.py)
        import arviz as _az
        import rl_ddm_all
        idata = _az.from_netcdf(os.path.join(run_dir, "models", rl_ddm_all.TRACE_NAME))
        df = rl_ddm_all.load_data(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data",
                                               "cleaned_rl_model_all.csv"))
        fitted = set(str(s) for s in idata["posterior"]["subject__factor_dim"].values)
        df = df[df["subject"].astype(str).isin(fitted)].reset_index(drop=True)
        rl_ddm_all.plot_subject_level(idata, df, os.path.join(run_dir, "plots", "subject_ppc"))
        return
    label = group_label_for(run_dir)
    out = os.path.join(run_dir, "plots", "subject_ppc")
    os.makedirs(out, exist_ok=True)
    dt, df, pred = load(run_dir)
    subjects = list(pd.unique(df["subject"]))
    print(f"{label}: {len(subjects)} subjects, {len(df)} valid trials, "
          f"{pred.shape[0]} predictive draws")
    met, q, cdf = per_subject_stats(df, pred, subjects)
    met.round(4).to_csv(os.path.join(out, "subject_ppc_metrics.csv"))
    pd.DataFrame({"key": [f"S{k+1}" for k in range(len(subjects))], "subject": subjects}) \
        .to_csv(os.path.join(out, "subject_key.csv"), index=False)
    plot_rt_density(df, pred, subjects, met, label, out)
    plot_cue_means(cdf, subjects, label, out)
    plot_accuracy_summary(q, cdf, label, out)
    plot_params(dt, subjects, label, out)
    print(met[["n_valid", "obs_mean_rt", "pred_mean_rt", "quantile_rmse_ms",
               "quantiles_inside_90pi", "cue_means_inside_90pi"]].round(3).to_string())
    print(f"Quantiles inside 90% PI: {q['inside'].mean():.0%}   "
          f"cue means inside 90% PI: {cdf['inside'].mean():.0%}")
    print("Saved to", out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python evaluation/plot_subject_ppc.py ../outputs/<run_folder>")
    main(sys.argv[1].rstrip("/"))
