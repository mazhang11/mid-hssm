"""
rl_cl.py

RL-DDM with LOG cue scaling, Session 1 controls. Identical to rl_ddm.py (Rescorla-Wagner
value per cue -> drift, masked analytical DDM, same priors and links) except for the reward
value a hit delivers to the learner:

  rl_ddm (linear):  value = $ / 5                    neutral 0, small 0.100, medium 0.200, large 1
  rl_cl  (log):     value = log(1 + $) / log(6)      neutral 0, small 0.226, medium 0.387, large 1

with $ = 0, 0.50, 1, 5 for the neutral/small/medium/large cue. Normalizing by log(6) keeps
neutral at 0 and large at 1, so `scaler` keeps its meaning (drift gained per unit of learned
large-cue value) and is comparable across the scaling models. The log only redistributes value
among the cues: small and medium are worth relatively more than under linear scaling.

  v_t      = v0 + scaler * Q_t[cue_t]
  Q_{t+1}  = Q_t[cue_t] + rl_alpha * (feedback_t - Q_t[cue_t])
  feedback = hit * log(1 + $[cue]) / log(6)     (0 on failures; column feedback_log)

Free parameters, all ~ 1 + (1|subject): rl_alpha (logit), scaler, v0, a (logit), t; z = 0.5.

Outputs: outputs/rl_cl_<date>_v<N> (or _Job<id>), trace models/rl_cl_model_trace.nc.
Fitting, PPC and plots: rl_cue_scaling.py.

Usage (from models/):
  python rl_cl.py                      fit
  RL_DDM_SMOKE=5 python rl_cl.py       smoke run
  python rl_cl.py ../outputs/<run>     re-make plots from a saved trace
"""

import numpy as np
import jax.numpy as jnp
from jax.lax import scan
import hssm
from hssm.rl import get_rlssm_model_config, register_rlssm_model
from hssm.utils import annotate_function

import rl_ddm
from rl_ddm import (N_CUES, DECISION_PROCESS, LEARNING_BOUNDS, LEARNING_DEFAULTS, FREE_PARAMS,
                    PARAM_BOUNDS, LOGIT_PARAMS, hierarchical_param, logit_param)
import rl_cue_scaling as rcs

MODEL_NAME = "mid_rw_ddm_logcue"
CUE_VALUE = np.log1p(rcs.DOLLARS) / np.log1p(rcs.DOLLARS.max())   # by cue_idx: 0, .226, .387, 1


def prepare(df):
    """Adds feedback_log = hit * CUE_VALUE[cue_idx] (the reward the learner sees)."""
    return df.assign(feedback_log=df["hit"].values * CUE_VALUE[df["cue_idx"].values.astype(int)])


# ---------------------------------------------------------------------------
# Learning process: same Rescorla-Wagner trial update as rl_ddm, log-scaled feedback
# ---------------------------------------------------------------------------

@annotate_function(
    inputs=["rl_alpha", "scaler", "v0", "cue_idx", "feedback_log"],
    outputs=["v"],
)
def compute_v(subject_trials):
    """One subject's trials (n_trials, 5) -> trial-wise drift (n_trials,)."""
    _, v = scan(rl_ddm._rw_trial, jnp.zeros(N_CUES), subject_trials)
    return v


register_rlssm_model(
    name=MODEL_NAME,
    decision_process=DECISION_PROCESS,
    learning_process={"v": compute_v},
    learning_process_params=list(LEARNING_BOUNDS),
    learning_process_bounds=LEARNING_BOUNDS,
    learning_process_params_default=LEARNING_DEFAULTS,
    extra_fields=["cue_idx", "feedback_log"],
    choices=[-1, 1],
    description="Rescorla-Wagner value per cue (log-scaled reward) -> analytical DDM drift, masked",
    decision_process_loglik_kind="analytical",
)


def build_model(df):
    """Same parameters and priors as rl_ddm.build_model. evaluate_models.py imports this."""
    return hssm.RLSSM(
        data=prepare(df),
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
            hierarchical_param("t", 0.05, 0.6, 0.25, 0.1, 0.05, initval=0.15),
            hssm.Param("z", prior=0.5),
        ],
    )


def cue_values(params):
    """Value of a hit per cue for each draw: fixed, (S, 4)."""
    return np.broadcast_to(CUE_VALUE, (len(params["rl_alpha"]), N_CUES))


SPEC = rcs.Spec(
    run_name="rl_cl",
    title="log cue scaling (rl_cl)",
    free_params=FREE_PARAMS,
    bounds=PARAM_BOUNDS,
    logit_params=LOGIT_PARAMS,
    cue_values=cue_values,
    build_model=build_model,
    prepare=prepare,
)

if __name__ == "__main__":
    rcs.main(SPEC)
