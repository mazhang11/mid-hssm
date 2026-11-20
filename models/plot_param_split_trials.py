"""
plot_param_split_trials.py

Trial-level companion to plot_param_split.py. Subjects are split at the median of their fitted
RL-DDM parameter (posterior median, natural scale): learning rate (rl_alpha) and value-to-drift
scaling (scaler). Split rule and data are identical to plot_param_split.py. Observed data only.

Every subject in this dataset sees the same 134-trial cue sequence, so trial number lines up
across subjects and each cue panel has a point only at the trials where that cue appeared.

Per split, two figures (rows: Low / High group; columns: Neutral, Small, Medium, Large cue):
  rt_by_trial_<param>_split.png   x = trial number in session, y = mean RT across the group's
                                  subjects at that trial (valid trials only), +/- 1 SE band,
                                  with a 5-presentation centered running mean. Dotted lines
                                  mark block boundaries.
  rt_dist_<param>_split.png       RT distribution per cue: density histogram of all valid trials
                                  pooled over the group's subjects, dashed line at the median.
                                  RTs are quantized to 60 Hz frames (~16.7 ms), so there is one
                                  bin per frame, centered on the frame; trials beyond the plotted
                                  range are counted in the panel text, not piled into an edge bin.
  rt_by_trial_<param>_overlay.png Same as rt_by_trial, but Low and High groups overlaid in one
  rt_dist_<param>_overlay.png     row of 4 cue panels (Low = gray, High = black).

Writes to <run>/plots/param_split/.

Usage: cd models && python plot_param_split_trials.py [../outputs/<rl_run>]
"""

import os
import sys
import numpy as np
import pandas as pd
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import plot_subject_ppc as sp
import rl_ddm
from plot_param_split import RL_RUN, SPLITS, GROUP_STYLE

CUES = rl_ddm.CUE_LABELS
GROUPS = ["Low", "High"]
SMOOTH = 5                       # running-mean window, in presentations of the cue
FRAME_MS = 1000 / 60             # RTs are quantized to 60 Hz display frames
RT_RANGE = (150, 800)            # ms shown in the distribution plots


def load(run_dir):
    """Observed trials + per-subject posterior medians and Low/High labels for each split."""
    idata = az.from_netcdf(os.path.join(run_dir, "models", "rl_ddm_analytical_model_trace.nc"))
    df = pd.read_csv(sp.data_path_for(run_dir))
    df = df.sort_values(["subject", "trial_idx"], kind="stable").reset_index(drop=True)
    df["block"] = df["run"].str.extract(r"(\d+)$")[0].astype(int)
    df["rt_ms"] = np.where(df["valid"] == 1, 1000 * df["rt"], np.nan)
    subjects = list(pd.unique(df["subject"]))

    params = rl_ddm.subject_params(idata["posterior"].to_dataset(), subjects)
    med = pd.DataFrame({p: np.median(params[p].reshape(-1, len(subjects)), axis=0)
                        for p, _ in SPLITS}, index=subjects)
    for p, _ in SPLITS:
        med[f"{p}_group"] = np.where(med[p] > med[p].median(), "High", "Low")
    return df, med


def frame_bins(rt_ms):
    """Bin edges one display frame wide, centered on the frame times the RTs fall on."""
    phase = np.angle(np.mean(np.exp(2j * np.pi * rt_ms / FRAME_MS))) * FRAME_MS / (2 * np.pi)
    k0 = np.floor((RT_RANGE[0] - phase) / FRAME_MS)
    k1 = np.ceil((RT_RANGE[1] - phase) / FRAME_MS)
    return phase + FRAME_MS * (np.arange(k0, k1 + 1) - 0.5)


def block_starts(df):
    first = df.groupby("block")["trial_idx"].min().sort_index()
    return first.values, df["trial_idx"].max()


def trial_means(df, subjects):
    """Per (trial, cue): mean RT, SE and n across the given subjects."""
    d = df[df["subject"].isin(subjects)]
    g = d.groupby(["trial_idx", "cue_idx"])["rt_ms"]
    out = g.agg(mean="mean", sd="std", n="count").reset_index()
    out["se"] = out["sd"] / np.sqrt(out["n"])
    return out


def style_row_labels(axes, p, pname, med):
    for r, g in enumerate(GROUPS):
        vals = med.loc[med[f"{p}_group"] == g, p]
        axes[r, 0].annotate(f"{g} {pname.lower()}\n(n={len(vals)}, {vals.min():.3g}–{vals.max():.3g})",
                            xy=(-0.30, 0.5), xycoords="axes fraction", rotation=90,
                            ha="center", va="center", fontsize=9.5, fontweight="bold", color=sp.INK)


def plot_trials(df, med, p, pname, label, out):
    starts, last = block_starts(df)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.6), sharex=True, sharey=True)
    for r, g in enumerate(GROUPS):
        subs = med.index[med[f"{p}_group"] == g]
        tm = trial_means(df, subs)
        for c, name in CUES.items():
            ax = axes[r, c]
            d = tm[tm["cue_idx"] == c].sort_values("trial_idx")
            x, m, se = d["trial_idx"].values, d["mean"].values, d["se"].values
            ax.fill_between(x, m - se, m + se, color=sp.CUE_COLORS[c], alpha=0.15, lw=0)
            ax.plot(x, m, "o", color=sp.CUE_COLORS[c], markersize=4, alpha=0.6,
                    markeredgecolor="white", markeredgewidth=0.5)
            run = pd.Series(m).rolling(SMOOTH, center=True, min_periods=2).mean()
            ax.plot(x, run, "-", color=sp.CUE_COLORS[c], lw=2)
            for s in starts[1:]:
                ax.axvline(s - 0.5, color=sp.MUTED, ls=":", lw=1, zorder=0)
            if r == 0:
                ax.set_title(f"{name} cue  ({len(x)} trials)", loc="left")
            if r == 1:
                ax.set_xlabel("Trial number (session)")
            if c == 0:
                ax.set_ylabel("Mean RT (ms)")
    for ax in axes.ravel():
        ax.set_xlim(0, last + 1)
    # Block labels along the top of the first row
    edges = list(starts) + [last + 1]
    for ax in axes[0]:
        for b in range(len(starts)):
            ax.text((edges[b] + edges[b + 1]) / 2, 0.985, f"Block {b + 1}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7.5, color=sp.MUTED)
    style_row_labels(axes, p, pname, med)
    handles = [Line2D([], [], color=sp.MUTED, marker="o", ls="", markersize=4, label="Group mean at trial"),
               Line2D([], [], color=sp.MUTED, lw=2, label=f"Running mean ({SMOOTH} presentations)"),
               Patch(color=sp.MUTED, alpha=0.3, label="±1 SE across subjects"),
               Line2D([], [], color=sp.MUTED, ls=":", lw=1, label="Block boundary")]
    sp.header(fig, f"Observed RT by trial, split on {pname.lower()} — {label}", handles, ncol=4)
    fig.subplots_adjust(left=0.08)
    path = os.path.join(out, f"rt_by_trial_{p}_split.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_dists(df, med, p, pname, label, out):
    bins = frame_bins(df["rt_ms"].dropna().values)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.0), sharex=True, sharey=True)
    for r, g in enumerate(GROUPS):
        subs = med.index[med[f"{p}_group"] == g]
        d = df[df["subject"].isin(subs)]
        for c, name in CUES.items():
            ax = axes[r, c]
            rt = d.loc[d["cue_idx"] == c, "rt_ms"].dropna().values
            # density over all of the group's trials, so off-range trials still count in the total
            h, _ = np.histogram(rt, bins=bins)
            ax.bar(bins[:-1], h / (len(rt) * np.diff(bins)), width=np.diff(bins), align="edge",
                   color=sp.CUE_COLORS[c], alpha=0.75, edgecolor="white", linewidth=0.6)
            beyond = np.mean(rt > bins[-1]) * 100
            mdn = np.median(rt)
            ax.axvline(mdn, color=sp.INK, ls="--", lw=1.4)
            ax.text(0.97, 0.95, f"median {mdn:.0f} ms\nmean {rt.mean():.0f} ms\n{len(rt)} trials\n{beyond:.1f}% > {bins[-1]:.0f} ms",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8, color=sp.INK_2)
            if r == 0:
                ax.set_title(f"{name} cue", loc="left")
            if r == 1:
                ax.set_xlabel("RT (ms)")
            if c == 0:
                ax.set_ylabel("Density")
    style_row_labels(axes, p, pname, med)
    handles = [Patch(color=sp.MUTED, alpha=0.75,
                     label="Valid trials, pooled over the group's subjects (one bin per 60 Hz frame)"),
               Line2D([], [], color=sp.INK, ls="--", lw=1.4, label="Median")]
    sp.header(fig, f"Observed RT distribution by cue, split on {pname.lower()} — {label}", handles, ncol=2)
    fig.subplots_adjust(left=0.08)
    path = os.path.join(out, f"rt_dist_{p}_split.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def group_handles(med, p):
    out = []
    for g in GROUPS:
        vals = med.loc[med[f"{p}_group"] == g, p]
        out.append(Line2D([], [], color=GROUP_STYLE[g], lw=2,
                          label=f"{g} group (n={len(vals)}, {p} {vals.min():.3g}–{vals.max():.3g})"))
    return out


def plot_trials_overlay(df, med, p, pname, label, out):
    """Low and High groups overlaid in one panel per cue."""
    starts, last = block_starts(df)
    edges = list(starts) + [last + 1]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.9), sharex=True, sharey=True)
    for g in GROUPS:
        tm = trial_means(df, med.index[med[f"{p}_group"] == g])
        for c, name in CUES.items():
            ax = axes[c]
            d = tm[tm["cue_idx"] == c].sort_values("trial_idx")
            x, m, se = d["trial_idx"].values, d["mean"].values, d["se"].values
            ax.fill_between(x, m - se, m + se, color=GROUP_STYLE[g], alpha=0.14, lw=0)
            ax.plot(x, m, "o", color=GROUP_STYLE[g], markersize=3.5, alpha=0.5,
                    markeredgecolor="white", markeredgewidth=0.5)
            ax.plot(x, pd.Series(m).rolling(SMOOTH, center=True, min_periods=2).mean(), "-",
                    color=GROUP_STYLE[g], lw=2)
    for c, name in CUES.items():
        ax = axes[c]
        n_trials = (df.drop_duplicates("trial_idx")["cue_idx"] == c).sum()
        ax.set_title(f"{name} cue  ({n_trials} trials)", loc="left", color=sp.CUE_COLORS[c],
                     fontweight="bold")
        for s in starts[1:]:
            ax.axvline(s - 0.5, color=sp.MUTED, ls=":", lw=1, zorder=0)
        for b in range(len(starts)):
            ax.text((edges[b] + edges[b + 1]) / 2, 0.985, f"Block {b + 1}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7.5, color=sp.MUTED)
        ax.set_xlim(0, last + 1)
        ax.set_xlabel("Trial number (session)")
    axes[0].set_ylabel("Mean RT (ms)")
    handles = group_handles(med, p) + [
        Patch(color="none", label=f"lines: running mean over {SMOOTH} presentations · dots: group mean "
                                  f"at each trial · bands: ±1 SE across subjects")]
    sp.header(fig, f"Observed RT by trial, low vs high {pname.lower()} — {label}", handles, ncol=3)
    path = os.path.join(out, f"rt_by_trial_{p}_overlay.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_dists_overlay(df, med, p, pname, label, out):
    """Low (filled) and High (outline) RT histograms overlaid in one panel per cue."""
    bins = frame_bins(df["rt_ms"].dropna().values)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), sharex=True, sharey=True)
    for c, name in CUES.items():
        ax = axes[c]
        lines, stats = [], {}
        for g in GROUPS:
            d = df[df["subject"].isin(med.index[med[f"{p}_group"] == g])]
            rt = d.loc[d["cue_idx"] == c, "rt_ms"].dropna().values
            dens = np.histogram(rt, bins=bins)[0] / (len(rt) * np.diff(bins))
            if g == "Low":
                ax.bar(bins[:-1], dens, width=np.diff(bins), align="edge", color=GROUP_STYLE[g],
                       alpha=0.35, edgecolor="white", linewidth=0.6)
            ax.stairs(dens, bins, color=GROUP_STYLE[g], lw=1.6 if g == "High" else 1.0)
            ax.axvline(np.median(rt), color=GROUP_STYLE[g], ls="--", lw=1.4)
            stats[g] = (np.median(rt), rt.mean(), np.mean(rt > bins[-1]) * 100)
        txt = "median / mean (ms)\n" + "\n".join(f"{g:<4}  {s[0]:.0f} / {s[1]:.0f}   ({s[2]:.1f}% > {bins[-1]:.0f})"
                                                for g, s in stats.items())
        ax.text(0.97, 0.62, txt, transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                color=sp.INK_2, family="monospace")
        ax.set_title(f"{name} cue", loc="left", color=sp.CUE_COLORS[c], fontweight="bold")
        ax.set_xlabel("RT (ms)")
    axes[0].set_ylabel("Density")
    handles = [Patch(facecolor=GROUP_STYLE["Low"], alpha=0.35, edgecolor=GROUP_STYLE["Low"],
                     label=group_handles(med, p)[0].get_label()),
               Line2D([], [], color=GROUP_STYLE["High"], lw=1.6, label=group_handles(med, p)[1].get_label()),
               Line2D([], [], color=sp.MUTED, ls="--", lw=1.4, label="Group median")]
    handles.append(Patch(color="none", label="Valid trials pooled over subjects; one bin per 60 Hz frame"))
    sp.header(fig, f"Observed RT distribution by cue, low vs high {pname.lower()} — {label}", handles, ncol=4)
    path = os.path.join(out, f"rt_dist_{p}_overlay.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main(run_dir=RL_RUN):
    df, med = load(run_dir)
    label = sp.group_label_for(run_dir)
    out = os.path.join(run_dir, "plots", "param_split")
    os.makedirs(out, exist_ok=True)
    for p, pname in SPLITS:
        print("Saved", plot_trials(df, med, p, pname, label, out))
        print("Saved", plot_dists(df, med, p, pname, label, out))
        print("Saved", plot_trials_overlay(df, med, p, pname, label, out))
        print("Saved", plot_dists_overlay(df, med, p, pname, label, out))


if __name__ == "__main__":
    main(*[a.rstrip("/") for a in sys.argv[1:2]])
