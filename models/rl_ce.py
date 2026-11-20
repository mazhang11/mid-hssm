"""
rl_ce.py

RL-DDM with POWER (fitted-exponent) cue scaling, Session 1 controls. Identical to rl_ddm.py
(Rescorla-Wagner value per cue -> drift, masked analytical DDM, same priors and links) except
for the reward value a hit delivers to the learner, which gets a free exponent gamma:

  value = ($ / 5) ** gamma      $ = 0, 0.50, 1, 5  ->  neutral 0, small 0.1**gamma,
                                                       medium 0.2**gamma, large 1

  gamma = 1   the linear rl_ddm model (0, 0.1, 0.2, 1)
  gamma < 1   concave: small and medium rewards worth relatively more (e.g. 0.5 -> 0, .32, .45, 1)
  gamma > 1   convex: only the large reward matters much  (e.g. 2 -> 0, .01, .04, 1)

Neutral stays 0 and large stays 1 for every gamma, so `scaler` keeps its meaning and is
comparable across the scaling models; gamma is identified by how fast small/medium cues
speed up relative to the large cue.

  v_t      = v0 + scaler * Q_t[cue_t]
  Q_{t+1}  = Q_t[cue_t] + rl_alpha * (feedback_t - Q_t[cue_t])
  feedback = hit * ($[cue] / 5) ** gamma          (0 on failures)

Free parameters, all ~ 1 + (1|subject):
  rl_alpha (logit), scaler, v0, a (logit), t      as in rl_ddm
  gamma    logit link over (0, 3); group prior Normal(logit-scale value of gamma = 1, 1),
           i.e. ~95% prior mass on gamma in (0.2, 2.3); starts at gamma = 1; subject
           sigma ~ HalfNormal(0.5) on the logit scale, like rl_alpha and a
  z = 0.5

Outputs: outputs/rl_ce_<date>_v<N> (or _Job<id>), trace models/rl_ce_model_trace.nc.
Fitting, PPC and plots: rl_cue_scaling.py.

Usage (from models/):
  python rl_ce.py                      fit
  RL_DDM_SMOKE=5 python rl_ce.py       smoke run
  python rl_ce.py ../outputs/<run>     re-make plots from a saved trace
"""

import os
import numpy as np
import jax.numpy as jnp
from jax.lax import scan
import hssm
from hssm.rl import get_rlssm_model_config, register_rlssm_model
from hssm.utils import annotate_function

import rl_ddm
from rl_ddm import (N_CUES, DECISION_PROCESS, LEARNING_BOUNDS, LEARNING_DEFAULTS, DDM_BOUNDS,
                    hierarchical_param, logit_param)
import rl_cue_scaling as rcs

MODEL_NAME = "mid_rw_ddm_powcue"
CUE_FRAC = rcs.DOLLARS / rcs.DOLLARS.max()                 # by cue_idx: 0, 0.1, 0.2, 1

GAMMA_BOUNDS = (0.0, 3.0)
GAMMA_CENTER = 1.0
CE_LEARNING_BOUNDS = {**LEARNING_BOUNDS, "gamma": GAMMA_BOUNDS}
CE_LEARNING_DEFAULTS = LEARNING_DEFAULTS + [GAMMA_CENTER]
FREE_PARAMS = ["rl_alpha", "scaler", "v0", "gamma", "a", "t"]
PARAM_BOUNDS = {**CE_LEARNING_BOUNDS, **DDM_BOUNDS}
LOGIT_PARAMS = ["rl_alpha", "a", "gamma"]

# value = exp(gamma * log(frac)) for the rewarded cues; the neutral cue (frac 0) is zeroed with
# jnp.where instead of computing 0 ** gamma, whose gradient in gamma is 0 * log(0) = NaN. Its log
# uses a dummy frac of 1 so the unused where-branch has a finite gradient (NaN * 0 is still NaN).
# This is a numerical detail only; it is unrelated to masked (response = -1) trials.
_LOG_FRAC = jnp.asarray(np.log(np.where(CUE_FRAC > 0, CUE_FRAC, 1.0)), dtype=jnp.float32)
_REWARDED = jnp.asarray(CUE_FRAC > 0)


# ---------------------------------------------------------------------------
# Learning process: same Rescorla-Wagner trial update as rl_ddm, power-scaled feedback
# ---------------------------------------------------------------------------

@annotate_function(
    inputs=["rl_alpha", "scaler", "v0", "gamma", "cue_idx", "hit"],
    outputs=["v"],
)
def compute_v(subject_trials):
    """One subject's trials (n_trials, 6) -> trial-wise drift (n_trials,)."""
    rl_alpha, scaler, v0, gamma, cue_idx, hit = (subject_trials[:, i] for i in range(6))
    cue = jnp.astype(cue_idx, jnp.int32)
    value = jnp.where(_REWARDED[cue], jnp.exp(gamma * _LOG_FRAC[cue]), 0.0)
    rows = jnp.stack([rl_alpha, scaler, v0, cue_idx, hit * value], axis=1)
    _, v = scan(rl_ddm._rw_trial, jnp.zeros(N_CUES, dtype=subject_trials.dtype), rows)
    return v


register_rlssm_model(
    name=MODEL_NAME,
    decision_process=DECISION_PROCESS,
    learning_process={"v": compute_v},
    learning_process_params=list(CE_LEARNING_BOUNDS),
    learning_process_bounds=CE_LEARNING_BOUNDS,
    learning_process_params_default=CE_LEARNING_DEFAULTS,
    extra_fields=["cue_idx", "hit"],
    choices=[-1, 1],
    description="Rescorla-Wagner value per cue (reward ** gamma) -> analytical DDM drift, masked",
    decision_process_loglik_kind="analytical",
)


def gamma_param(intercept_sigma=1.0, subject_sigma=0.5):
    """gamma ~ 1 + (1|subject) on the logit scale of GAMMA_BOUNDS, centered at gamma = 1
    (rl_ddm.logit_param with this model's bounds)."""
    lower, upper = GAMMA_BOUNDS
    x = float(np.log((GAMMA_CENTER - lower) / (upper - GAMMA_CENTER)))
    return hssm.Param(
        "gamma",
        formula="gamma ~ 1 + (1|subject)",
        link="log_logit",
        prior={
            "Intercept": {"name": "Normal", "mu": x, "sigma": intercept_sigma, "initval": x},
            "1|subject": {"name": "Normal", "mu": 0,
                          "sigma": {"name": "HalfNormal", "sigma": subject_sigma}},
        },
    )


# Where sampling starts (priors are unchanged). The first runs started every chain at the prior
# centers with t = 0.15, and most chains never left: at that start the log density is about
# -5,600 (healthy posterior about +12,400), and the stuck chains ended with t 0.215-0.248
# (healthy 0.261) and alpha still at its starting value, with step sizes ~1e-6.
# t = 0.19 alone lifts the start to about +400 while staying 12 ms below the fastest valid RT
# (0.202 s), so no trial is floored at the start. alpha, scaler, v0 and a start at rl_ddm's
# 10-29 Session 1 posterior. RL_DDM_PRIOR_START=1 restores the old start (Jobs 9372776, 9393765).
START = {"rl_alpha": 0.067, "scaler": 1.3, "v0": 6.7, "a": 0.73, "t": 0.19}


def logit_param_start(name, center, start, intercept_sigma=1.0, subject_sigma=0.5):
    """rl_ddm.logit_param (prior centered at `center`), but sampling starts at `start`."""
    p = logit_param(name, center, intercept_sigma, subject_sigma)
    lower, upper = PARAM_BOUNDS[name]
    p.prior["Intercept"]["initval"] = float(np.log((start - lower) / (upper - start)))
    return p


def build_model(df):
    """rl_ddm.build_model's parameters and priors + gamma. evaluate_models.py imports this.
    Starting values: START (see above), unless RL_DDM_PRIOR_START=1."""
    if os.environ.get("RL_DDM_PRIOR_START") == "1":
        start = {"rl_alpha": 0.2, "scaler": 1.0, "v0": 6.0, "a": 0.65, "t": 0.15}
    else:
        start = START
    return hssm.RLSSM(
        data=df,
        model_config=get_rlssm_model_config(MODEL_NAME),
        participant_col="subject",
        p_outlier=0,
        lapse=None,
        process_initvals=False,
        include=[
            logit_param_start("rl_alpha", center=0.2, start=start["rl_alpha"]),
            hierarchical_param("scaler", -5.0, 20.0, 1.0, 3.0, 1.0, initval=start["scaler"]),
            hierarchical_param("v0", 0.1, 20.0, 6.0, 3.0, 1.5, initval=start["v0"]),
            gamma_param(),
            logit_param_start("a", center=0.65, start=start["a"]),
            hierarchical_param("t", 0.05, 0.6, 0.25, 0.1, 0.05, initval=start["t"]),
            hssm.Param("z", prior=0.5),
        ],
    )


def cue_values(params):
    """Value of a hit per cue for each draw: (S, 4) = frac ** gamma, neutral 0."""
    g = np.asarray(params["gamma"], dtype=float)[:, None]
    safe = np.where(CUE_FRAC > 0, CUE_FRAC, 1.0)[None, :]
    return np.where(CUE_FRAC[None, :] > 0, safe ** g, 0.0)


SPEC = rcs.Spec(
    run_name="rl_ce",
    title="power cue scaling, fitted exponent (rl_ce)",
    free_params=FREE_PARAMS,
    bounds=PARAM_BOUNDS,
    logit_params=LOGIT_PARAMS,
    cue_values=cue_values,
    build_model=build_model,
)

if __name__ == "__main__":
    rcs.main(SPEC)
