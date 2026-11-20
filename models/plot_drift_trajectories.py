"""
plot_drift_trajectories.py

Implied learning trajectories of the drift rate from a fitted RL-DDM run (rl_ddm.py family or
rl_ddm_all.py). For each subject (or subject x session unit) and each of 200 posterior draws,
the Rescorla-Wagner learner is replayed over the observed cue/feedback sequence (masked trials
included: the participant still saw the feedback), and the drift for EACH cue is recorded at
every trial:

    v_t[cue] = v0 + scaler * Q_t[cue]     (Q_t = value before trial t's feedback)

i.e. the drift the model would use if that cue appeared on trial t. The drift actually used on
trial t is v_t[cue_t]. Everything is on the natural scale.

Writes to <run>/plots/drift_trajectories/:
  drift_trajectories_group.png      top: mean drift across subjects per cue (median, 90%);
                                    bottom: learned advantage over neutral, v[cue] - v[neutral]
                                    (rl_ddm_all: one column per session x drug cell)
  drift_trajectories_subjects_p*.png one panel per subject/unit: median drift per cue (90% band
                                    on the large cue); same y-axis for every panel and page
  drift_trajectories_subjects_zoom_p*.png  same, y-axis fixed at 2.5-10 (higher values cut off)
  drift_advantage_subjects_p*.png   same, as v[cue] - v[neutral] with 90% bands (shared y-axis)
  drift_trajectories_median.csv     per unit x trial x cue posterior median drift

Usage: cd models && python plot_drift_trajectories.py ../outputs/<run_folder>
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

N_DRAWS_PER_CHAIN = 50
PAGE = 36
ZOOM_YLIM = (2.5, 10.0)   # drift_trajectories_subjects_zoom_p*.png
CUES = rl_ddm.CUE_LABELS          # {0: "Neutral", ...}
CUE_COLORS = sp.CUE_COLORS


def load_run(run_dir, seed=0):
    """Returns (df, units, params, label, cell_col): df sorted in learning order, units = one
    row per learning sequence with a display key, params = {p: (samples, n_units)}."""
    is_all = "rl_ddm_all" in run_dir
    if is_all:
        import rl_ddm_all as m
        idata = az.from_netcdf(os.path.join(run_dir, "models", m.TRACE_NAME))
        df = m.load_data(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                                      "cleaned_rl_model_all.csv"))
    else:
        idata = az.from_netcdf(os.path.join(run_dir, "models", "rl_ddm_analytical_model_trace.nc"))
        df = pd.read_csv(sp.data_path_for(run_dir))
        df = df.sort_values(["subject", "trial_idx"], kind="stable").reset_index(drop=True)
    post = idata["posterior"].to_dataset()
    fitted = set(str(s) for s in post["subject__factor_dim"].values)
    df = df[df["subject"].astype(str).isin(fitted)].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    n = post.sizes["draw"]
    post = post.isel(draw=np.sort(rng.choice(n, min(N_DRAWS_PER_CHAIN, n), replace=False)))

    if is_all:
        units = m.unit_table(df)
        units["key"] = m.unit_keys(units)
        params = m.unit_params(post, units)
        label, unit_col = m.GROUP_LABEL, m.UNIT_COL
    else:
        subjects = list(pd.unique(df["subject"]))
        units = pd.DataFrame({"unit": subjects, "subject": subjects,
                              "key": [f"S{k+1}" for k in range(len(subjects))]})
        params = rl_ddm.subject_params(post, subjects)
        label, unit_col = sp.group_label_for(run_dir), "subject"
        df = df.assign(unit=df["subject"])
    params = {p: v.reshape(-1, v.shape[-1]) for p, v in params.items()}   # (samples, units)
    return df, units, params, label, is_all


def drift_trajectories(df, units, params):
    """(n_units, samples, n_trials, 4) drift for every cue at every trial."""
    out = []
    for j, u in enumerate(units["unit"]):
        d = df[df["unit"] == u]
        cue, fb = d["cue_idx"].values.astype(int), d["feedback"].values
        alpha, scaler, v0 = params["rl_alpha"][:, j], params["scaler"][:, j], params["v0"][:, j]
        q = np.zeros((len(alpha), rl_ddm.N_CUES))
        v = np.empty((len(alpha), len(d), rl_ddm.N_CUES))
        for i, (c, r) in enumerate(zip(cue, fb)):
            v[:, i, :] = v0[:, None] + scaler[:, None] * q
            q[:, c] += alpha * (r - q[:, c])
        out.append(v)
    return np.stack(out)


def block_starts(df, unit):
    run = df.loc[df["unit"] == unit, "run"].values
    return [i + 1 for i in range(1, len(run)) if run[i] != run[i - 1]]   # trial numbers (1-based)


def plot_group(v, units, df, label, is_all, out):
    if is_all:   # one column per session x drug cell
        import rl_ddm_all as m
        groups = [(lab, (units["cell"] == key).values) for *_, key, lab in m.CELLS]
    else:
        groups = [(label, np.ones(len(units), bool))]
    groups = [g for g in groups if g[1].any()]
    trials = np.arange(1, v.shape[2] + 1)
    fig, axes = plt.subplots(2, len(groups), figsize=(max(10.0, 4.6 * len(groups) + 0.8), 7.2),
                             sharex=True, sharey="row", squeeze=False)
    blocks = block_starts(df, units["unit"].iloc[0])
    for g, (glabel, sel) in enumerate(groups):
        mean_v = v[sel].mean(axis=0)                     # (samples, trials, cues): mean over units
        adv = mean_v - mean_v[..., :1]                   # minus neutral
        for row, arr in enumerate([mean_v, adv]):
            ax = axes[row, g]
            for b in blocks:
                ax.axvline(b - 0.5, color=sp.GRID, lw=1.2, zorder=0)
            ax.grid(axis="x", visible=False)
            for c, name in CUES.items():
                if row == 1 and c == 0:
                    continue
                lo, mid, hi = np.percentile(arr[..., c], [sp.CI[0], 50, sp.CI[1]], axis=0)
                ax.fill_between(trials, lo, hi, color=CUE_COLORS[c], alpha=0.18, lw=0)
                ax.plot(trials, mid, color=CUE_COLORS[c], lw=2)
            if row == 0:
                ax.set_title(f"{glabel}  (n={sel.sum()})", loc="left")
            else:
                ax.axhline(0, color=sp.MUTED, lw=1, ls="--")
                ax.set_xlabel("Trial (session; gray lines = block starts)")
        axes[0, 0].set_ylabel("Implied drift rate v")
        axes[1, 0].set_ylabel("Learned advantage over neutral\nv[cue] − v[neutral]")
    handles = [Line2D([], [], color=CUE_COLORS[c], lw=2, label=f"{name} cue") for c, name in CUES.items()]
    handles.append(Patch(color=sp.MUTED, alpha=0.3, label="90% posterior interval"))
    sp.header(fig, f"Implied drift-rate learning trajectories, mean across subjects — {label}", handles, ncol=5)
    fig.savefig(os.path.join(out, "drift_trajectories_group.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def shared_ylim(pct, advantage, pad=0.05):
    """One y-range for every subject panel (and page), covering everything drawn: the median
    lines of all cues plus the 90% bands (large cue only, or all non-neutral cues for the
    advantage plot, whose zero reference line is always included)."""
    banded = [1, 2, 3] if advantage else [3]
    lines = [1, 2, 3] if advantage else [0, 1, 2, 3]
    lo = min(pct[1][..., lines].min(), pct[0][..., banded].min())
    hi = max(pct[1][..., lines].max(), pct[2][..., banded].max())
    if advantage:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    w = hi - lo
    return lo - pad * w, hi + pad * w


def plot_subjects(v, units, df, label, out, advantage=False, ylim=None, suffix=""):
    """advantage=False: drift per cue (median lines; 90% band on the large cue only, since the
    baseline-drift uncertainty shared by all cues would otherwise stack four identical bands).
    advantage=True: v[cue] - v[neutral] with 90% bands (baseline uncertainty cancels)."""
    if advantage:
        v = v - v[..., :1]
    pct = np.percentile(v, [sp.CI[0], 50, sp.CI[1]], axis=1)      # (3, units, trials, cues)
    if ylim is None:
        ylim = shared_ylim(pct, advantage)
    n = len(units)
    pages = int(np.ceil(n / PAGE))
    trials = np.arange(1, v.shape[2] + 1)
    for p in range(pages):
        idx = list(range(p * PAGE, min(n, (p + 1) * PAGE)))
        fig, axes = sp.grid(len(idx), 6, w=2.4, h=1.8)
        for k, j in enumerate(idx):
            ax = axes.ravel()[k]
            u = units["unit"].iloc[j]
            for b in block_starts(df, u):
                ax.axvline(b - 0.5, color=sp.GRID, lw=1, zorder=0)
            d = df[df["unit"] == u]
            for c in CUES:
                if advantage and c == 0:
                    ax.axhline(0, color=CUE_COLORS[0], lw=1.2, ls="--")
                    continue
                lo, mid, hi = pct[:, j, :, c]
                if advantage or c == 3:
                    ax.fill_between(trials, lo, hi, color=CUE_COLORS[c], alpha=0.18, lw=0)
                ax.plot(trials, mid, color=CUE_COLORS[c], lw=1.4)
            ax.set_ylim(*ylim)
            ax.grid(axis="x", visible=False)   # vertical lines = block starts only
            # ticks along the bottom: trials where the large cue was rewarded
            big = np.flatnonzero((d["cue_idx"].values == 3) & (d["hit"].values == 1)) + 1
            ax.plot(big, np.full(len(big), ax.get_ylim()[0]), "|", color=CUE_COLORS[3],
                    markersize=4, alpha=0.8, clip_on=False)
            ax.set_title(units["key"].iloc[j], loc="left", fontsize=8)
        for ax in axes.ravel()[max(0, len(idx) - 6):len(idx)]:
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.set_xlabel("Trial")
        for r in range(axes.shape[0]):
            axes[r, 0].set_ylabel("v[cue] − v[neutral]" if advantage else "Drift v")
        handles = [Line2D([], [], color=CUE_COLORS[c], lw=1.4, ls="--" if advantage and c == 0 else "-",
                          label=f"{name}" + (" (reference)" if advantage and c == 0 else ""))
                   for c, name in CUES.items()]
        handles += [Patch(color=CUE_COLORS[3] if not advantage else sp.MUTED, alpha=0.3,
                          label="90% interval" + ("" if advantage else " (large cue)")),
                    Line2D([], [], ls="", marker="|", color=CUE_COLORS[3], label="Large reward received")]
        extra = f" (page {p + 1}/{pages})" if pages > 1 else ""
        what = "Learned drift advantage over neutral" if advantage else "Implied drift per cue across the session"
        sp.header(fig, f"{what}, per subject — {label}{extra}", handles, ncol=6)
        stem = "drift_advantage_subjects" if advantage else "drift_trajectories_subjects"
        fig.savefig(os.path.join(out, f"{stem}{suffix}_p{p + 1}.png"), dpi=200)
        plt.close(fig)


def main(run_dir):
    out = os.path.join(run_dir, "plots", "drift_trajectories")
    os.makedirs(out, exist_ok=True)
    df, units, params, label, is_all = load_run(run_dir)
    v = drift_trajectories(df, units, params)
    print(f"{label}: {len(units)} learning sequences x {v.shape[2]} trials, {v.shape[1]} posterior draws")
    med = np.median(v, axis=1)                                   # (units, trials, cues)
    long = pd.DataFrame({
        "unit": np.repeat(units["unit"].values, v.shape[2] * 4),
        "key": np.repeat(units["key"].values, v.shape[2] * 4),
        "trial": np.tile(np.repeat(np.arange(1, v.shape[2] + 1), 4), len(units)),
        "cue": np.tile([CUES[c] for c in CUES], len(units) * v.shape[2]),
        "drift_median": med.reshape(-1),
    })
    long.round(4).to_csv(os.path.join(out, "drift_trajectories_median.csv"), index=False)
    plot_group(v, units, df, label, is_all, out)
    plot_subjects(v, units, df, label, out)
    plot_subjects(v, units, df, label, out, advantage=True)
    # Zoomed version: fixed 2.5-10 drift axis (lines/bands above 10 are cut off)
    plot_subjects(v, units, df, label, out, ylim=ZOOM_YLIM, suffix="_zoom")
    end = v[:, :, -1, :].mean(axis=(0, 1))
    print("Mean implied drift at the last trial: " +
          ", ".join(f"{CUES[c]} {end[c]:.2f}" for c in CUES))
    print("Saved to", out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python plot_drift_trajectories.py ../outputs/<run_folder>")
    main(sys.argv[1].rstrip("/"))
