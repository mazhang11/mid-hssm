"""
plot_tradeoffs_natural.py

Parameter-tradeoff pair plot for an RL-DDM run (rl_ddm.py family) on the NATURAL scale:
learning rate in [0, 1] and boundary separation in its own units, not the logit scale the
sampler works on. Group-level values are those of a typical subject (random offset 0), the
<param>_group rows of summary.csv.

Styled like prior_success's tradeoff plot (filled 2-D posterior density contours, marginal
densities on the diagonal, wide axes): each axis spans the full range of the posterior draws
plus 15% padding on each side, clipped to the parameter's bounds. Each off-diagonal panel
reports the posterior correlation r.

Writes to <run>/plots/:
  RL_DDM_tradeoffs_natural.png              per-parameter axis ranges
  RL_DDM_tradeoffs_natural_shared_axes.png  v0 and a on the same axis ranges as prior_success's
                                            v_Intercept and a_Intercept (union of both
                                            posteriors), for side-by-side reading

Usage: cd models && python plot_tradeoffs_natural.py ../outputs/<rl_run> [../outputs/<prior_success_run>]
"""

import os
import sys
import numpy as np
import arviz as az
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import plot_subject_ppc as sp
import rl_ddm

HERE = os.path.dirname(os.path.abspath(__file__))
PS_RUN = os.path.join(HERE, "..", "outputs", "prior_success_2026-05-12_v1")
PAD = 0.15
HDI_LEVELS = [0.95, 0.80, 0.50]                 # outer -> inner
FILLS = ["#c6dcf7", "#5598e7", "#104281"]       # one blue hue, light -> dark (sequential)
LINE = "#2a78d6"
SHARED = {"v0": "v_Intercept", "a": "a_Intercept"}   # RL-DDM param -> prior_success counterpart


def padded_range(x, bounds=None, pad=PAD):
    lo, hi = float(np.min(x)), float(np.max(x))
    w = hi - lo
    lo, hi = lo - pad * w, hi + pad * w
    if bounds is not None:
        lo, hi = max(lo, bounds[0]), min(hi, bounds[1])
    return lo, hi


def hdi_thresholds(kde, x, y, levels):
    """Density values whose upper set holds `levels` of the posterior mass (from the draws)."""
    dens = kde(np.vstack([x, y]))
    return [np.quantile(dens, 1 - p) for p in levels]


def pair_plot(draws, ranges, title, path):
    params = list(draws)
    n = len(params)
    fig, axes = plt.subplots(n, n, figsize=(2.75 * n, 2.6 * n + 0.9))
    for i, pi in enumerate(params):
        for j, pj in enumerate(params):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            if i == j:
                x = draws[pi]
                grid = np.linspace(*ranges[pi], 300)
                dens = gaussian_kde(x)(grid)
                ax.fill_between(grid, dens, color=FILLS[0], lw=0)
                ax.plot(grid, dens, color=LINE, lw=1.6)
                med = np.median(x)
                ax.axvline(med, color=FILLS[2], lw=1, ls="--")
                ax.set_title(f"median {med:.3g}", loc="left", fontsize=8, color=sp.INK_2)
                ax.set_yticks([])
                ax.grid(axis="y", visible=False)
                ax.set_ylim(bottom=0)
            else:
                x, y = draws[pj], draws[pi]
                kde = gaussian_kde(np.vstack([x, y]))
                gx, gy = np.meshgrid(np.linspace(*ranges[pj], 120), np.linspace(*ranges[pi], 120))
                z = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
                levels = hdi_thresholds(kde, x, y, HDI_LEVELS) + [z.max() * 1.01]
                ax.contourf(gx, gy, z, levels=levels, colors=FILLS)
                ax.contour(gx, gy, z, levels=levels[:-1], colors="white", linewidths=0.6)
                ax.text(0.03, 0.96, f"r = {np.corrcoef(x, y)[0, 1]:+.2f}", transform=ax.transAxes,
                        va="top", fontsize=8.5, color=sp.INK)
                ax.set_ylim(*ranges[pi])
            ax.set_xlim(*ranges[pj])
            if i == n - 1:
                ax.set_xlabel(sp.PARAM_LABELS[pj])
            else:
                ax.tick_params(labelbottom=False)
            if j == 0 and i > 0:
                ax.set_ylabel(sp.PARAM_LABELS[pi])
            elif j > 0 and i != j:
                ax.tick_params(labelleft=False)
    handles = [Patch(color=c, label=f"{int(p * 100)}% posterior region") for p, c in zip(HDI_LEVELS, FILLS)]
    handles.append(Line2D([], [], color=LINE, lw=1.6, label="Marginal density"))
    sp.header(fig, title, handles, ncol=4)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("Saved", path)


def main(run_dir, ps_run=PS_RUN):
    idata = az.from_netcdf(os.path.join(run_dir, "models", "rl_ddm_analytical_model_trace.nc"))
    nat = rl_ddm.group_params_natural(idata["posterior"].to_dataset())
    draws = {p: nat[f"{p}_group"].values.ravel() for p in rl_ddm.FREE_PARAMS}
    bounds = rl_ddm.PARAM_BOUNDS
    ranges = {p: padded_range(draws[p], bounds[p]) for p in draws}
    label = sp.group_label_for(run_dir)
    out = os.path.join(run_dir, "plots")

    pair_plot(draws, ranges, f"RL-DDM: parameter tradeoffs, group level (natural scale) — {label}",
              os.path.join(out, "RL_DDM_tradeoffs_natural.png"))

    if os.path.isdir(ps_run):
        ps = az.from_netcdf(os.path.join(ps_run, "models", "prior_success_model_trace.nc"))["posterior"]
        shared = dict(ranges)
        for p, q in SHARED.items():
            lo1, hi1 = padded_range(draws[p], bounds[p])
            lo2, hi2 = padded_range(ps[q].values.ravel())
            shared[p] = (min(lo1, lo2), max(hi1, hi2))
            print(f"  {p}: RL-DDM [{lo1:.3g}, {hi1:.3g}], prior_success {q} [{lo2:.3g}, {hi2:.3g}] "
                  f"-> shared [{shared[p][0]:.3g}, {shared[p][1]:.3g}]")
        pair_plot(draws, shared,
                  f"RL-DDM tradeoffs (natural scale), v0 and a on prior_success's axis ranges — {label}",
                  os.path.join(out, "RL_DDM_tradeoffs_natural_shared_axes.png"))


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(*[a.rstrip("/") for a in sys.argv[1:]])
