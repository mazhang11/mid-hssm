"""
plot_param_split.py

Observed behavior of subjects split at the median of a fitted RL-DDM parameter (rl_ddm.py
family): learning rate (rl_alpha) and value-to-drift scaling (scaler). Each subject's value is
its posterior median on the natural scale; the split is 28 / 28 for the 56 Session 1 controls.

Behavior = mean RT on valid trials, per cue and block (4 blocks; large-cue trials per block rise
5/8/11/14). Group curves average subject means; bands are +/- 1 SE across subjects. The
right-hand column shows the large-cue speed-up, RT(neutral) - RT(large), with the model's
posterior predictive (same trials, same averaging; lower-boundary hits dropped) dashed.

Caveat: the parameters were fit to these same RTs, so a group difference is partly built in by
the split. The informative parts are its size and shape, and whether the model reproduces it.

Writes to <run>/plots/param_split/:
  behavior_by_param_split.png   rows: learning-rate split, scaling split
  param_split_scatter.png       subject posterior medians, rl_alpha vs scaler, with both cut lines
  param_split_subjects.csv      per-subject medians and group labels (S-keys as elsewhere)
  param_split_behavior.csv      group x block x cue observed means and SEs

Usage: cd models && python plot_param_split.py [../outputs/<rl_run>]
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

HERE = os.path.dirname(os.path.abspath(__file__))
RL_RUN = os.path.join(HERE, "..", "outputs", "rl_ddm_analytical_2026-10-29_Job8086695")
SPLITS = [("rl_alpha", "Learning rate"), ("scaler", "Value-to-drift scaling")]
GROUP_STYLE = {"Low": sp.MUTED, "High": sp.INK}      # ordered split: light ink -> dark ink
CUES = rl_ddm.CUE_LABELS


def load(run_dir):
    idata = az.from_netcdf(os.path.join(run_dir, "models", "rl_ddm_analytical_model_trace.nc"))
    df = pd.read_csv(sp.data_path_for(run_dir))
    df = df.sort_values(["subject", "trial_idx"], kind="stable").reset_index(drop=True)
    df["block"] = df["run"].str.extract(r"(\d+)$")[0].astype(int)
    subjects = list(pd.unique(df["subject"]))

    params = rl_ddm.subject_params(idata["posterior"].to_dataset(), subjects)
    med = pd.DataFrame({p: np.median(params[p].reshape(-1, len(subjects)), axis=0)
                        for p in rl_ddm.FREE_PARAMS})
    med.insert(0, "subject", subjects)
    med.insert(0, "key", [f"S{k + 1}" for k in range(len(subjects))])

    pp = idata["posterior_predictive"]["rt,response"].values           # (chain, draw, obs, 2)
    pp = pp.reshape(-1, pp.shape[-2], 2)
    if pp.shape[1] != len(df):
        raise ValueError(f"trace has {pp.shape[1]} trials, data has {len(df)}")
    pred = np.where(pp[..., 1] > 0, pp[..., 0], np.nan)                  # (samples, obs)
    return df, med, pred


def cell_means(rt, df):
    """Per-subject mean RT per (block, cue) on valid trials -> array (subjects, 4 blocks, 4 cues).
    rt: (obs,) or (samples, obs)."""
    single = np.ndim(rt) == 1
    rt = np.atleast_2d(rt)
    valid = df["valid"].values == 1
    subj_codes, subjects = pd.factorize(df["subject"])
    key = (subj_codes * 16 + (df["block"].values - 1) * 4 + df["cue_idx"].values.astype(int))[valid]
    rt = rt[:, valid]
    out = np.full((rt.shape[0], len(subjects) * 16), np.nan)
    with np.errstate(all="ignore"):
        for k in np.unique(key):
            out[:, k] = np.nanmean(rt[:, key == k], axis=1)
    out = out.reshape(rt.shape[0], len(subjects), 4, 4)
    return out[0] if single else out


def summarize(cells):
    """(subjects, blocks, cues) -> mean, SE across subjects (ms)."""
    n = np.sum(~np.isnan(cells), axis=0)
    return 1000 * np.nanmean(cells, axis=0), 1000 * np.nanstd(cells, axis=0, ddof=1) / np.sqrt(n)


def main(run_dir=RL_RUN):
    df, med, pred = load(run_dir)
    label = sp.group_label_for(run_dir)
    out = os.path.join(run_dir, "plots", "param_split")
    os.makedirs(out, exist_ok=True)

    obs = cell_means(df["rt"].values, df)                     # (subjects, 4, 4)
    sim = cell_means(pred, df)                                # (samples, subjects, 4, 4)
    blocks = np.arange(1, 5)
    for p, _ in SPLITS:
        cut = med[p].median()
        med[f"{p}_group"] = np.where(med[p] > cut, "High", "Low")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.6), sharex=True)
    rows = []
    rt_lim = [np.inf, -np.inf]
    for r, (p, pname) in enumerate(SPLITS):
        groups = med[f"{p}_group"].values
        for c, g in enumerate(["Low", "High"]):
            ax = axes[r, c]
            sel = groups == g
            m, se = summarize(obs[sel])
            for cue, name in CUES.items():
                ax.fill_between(blocks, m[:, cue] - se[:, cue], m[:, cue] + se[:, cue],
                                color=sp.CUE_COLORS[cue], alpha=0.15, lw=0)
                ax.plot(blocks, m[:, cue], "o-", color=sp.CUE_COLORS[cue], lw=2, markersize=6,
                        markeredgecolor="white")
                rows += [{"split": p, "group": g, "block": b, "cue": name, "mean_rt_ms": m[b - 1, cue],
                          "se_ms": se[b - 1, cue], "n_subjects": int(sel.sum())} for b in blocks]
            rt_lim = [min(rt_lim[0], np.nanmin(m - se)), max(rt_lim[1], np.nanmax(m + se))]
            vals = med.loc[sel, p]
            ax.set_title(f"{g} {pname.lower()}  (n={sel.sum()}, {p} {vals.min():.3g}–{vals.max():.3g})",
                         loc="left", fontsize=9.5)
            if c == 0:
                ax.set_ylabel("Mean RT (ms)")

        # Large-cue speed-up: RT(neutral) - RT(large), observed vs model
        ax = axes[r, 2]
        for g in ["Low", "High"]:
            sel = groups == g
            gain_obs = 1000 * (obs[sel][..., 0] - obs[sel][..., 3])          # (subjects, blocks)
            m = np.nanmean(gain_obs, axis=0)
            se = np.nanstd(gain_obs, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(gain_obs), axis=0))
            gain_sim = 1000 * np.nanmean(sim[:, sel, :, 0] - sim[:, sel, :, 3], axis=1)   # (samples, blocks)
            ax.fill_between(blocks, m - se, m + se, color=GROUP_STYLE[g], alpha=0.18, lw=0)
            ax.plot(blocks, m, "o-", color=GROUP_STYLE[g], lw=2, markersize=6, markeredgecolor="white")
            ax.plot(blocks, np.nanmedian(gain_sim, axis=0), "--", color=GROUP_STYLE[g], lw=1.6)
        ax.axhline(0, color=sp.GRID, lw=1.2, zorder=0)
        ax.set_title(f"Large-cue speed-up by {pname.lower()} group", loc="left", fontsize=9.5)
        ax.set_ylabel("RT(neutral) − RT(large) (ms)")
    pad = 0.05 * (rt_lim[1] - rt_lim[0])
    for ax in axes[:, :2].ravel():
        ax.set_ylim(rt_lim[0] - pad, rt_lim[1] + pad)
    for ax in axes[1]:
        ax.set_xlabel("Block")
        ax.set_xticks(blocks)
    handles = [Line2D([], [], color=sp.CUE_COLORS[c], marker="o", lw=2, label=f"{n} cue")
               for c, n in CUES.items()]
    handles += [Line2D([], [], color=GROUP_STYLE[g], marker="o", lw=2, label=f"{g} group (observed)")
                for g in GROUP_STYLE]
    handles += [Line2D([], [], color=sp.MUTED, ls="--", lw=1.6, label="Model prediction"),
                Patch(color=sp.MUTED, alpha=0.25, label="±1 SE across subjects")]
    sp.header(fig, f"Observed RT by median split on fitted parameters — {label}", handles, ncol=4)
    fig.savefig(os.path.join(out, "behavior_by_param_split.png"), dpi=200)
    plt.close(fig)

    # How the two splits overlap
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    both = med["rl_alpha_group"] + "/" + med["scaler_group"]
    ax.axvline(med["rl_alpha"].median(), color=sp.MUTED, ls="--", lw=1)
    ax.axhline(med["scaler"].median(), color=sp.MUTED, ls="--", lw=1)
    ax.axhline(0, color=sp.GRID, lw=1.2, zorder=0)
    ax.plot(med["rl_alpha"], med["scaler"], "o", color=sp.CUE_COLORS[0], markersize=8,
            markeredgecolor="white")
    r = np.corrcoef(med["rl_alpha"], med["scaler"])[0, 1]
    tab = pd.crosstab(med["rl_alpha_group"], med["scaler_group"])
    ax.text(0.97, 0.97, f"r = {r:+.2f}\n" + "\n".join(
        f"{a} α / {s} scaling: {tab.loc[a, s]}" for a in ["Low", "High"] for s in ["Low", "High"]),
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=sp.INK_2)
    ax.set_xlabel(sp.PARAM_LABELS["rl_alpha"] + " (subject posterior median)")
    ax.set_ylabel(sp.PARAM_LABELS["scaler"] + " (subject posterior median)")
    sp.header(fig, "Learning rate vs scaling per subject (dashed = median cuts)")
    fig.savefig(os.path.join(out, "param_split_scatter.png"), dpi=200)
    plt.close(fig)

    med.round(4).to_csv(os.path.join(out, "param_split_subjects.csv"), index=False)
    pd.DataFrame(rows).round(2).to_csv(os.path.join(out, "param_split_behavior.csv"), index=False)
    print(f"rl_alpha subject medians {med['rl_alpha'].min():.3f}–{med['rl_alpha'].max():.3f} "
          f"(cut {med['rl_alpha'].median():.3f}); scaler {med['scaler'].min():.2f}–{med['scaler'].max():.2f} "
          f"(cut {med['scaler'].median():.2f}); r = {r:+.2f}")
    print(tab)
    print("Saved to", out)


if __name__ == "__main__":
    main(*[a.rstrip("/") for a in sys.argv[1:2]])
