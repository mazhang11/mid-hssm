"""
rl_ddm_session2_placebo.py

The analytical RL-DDM from rl_ddm.py (same learner, priors, masking and plots), fit on
Session 2, Placebo-arm subjects (Phase 2 Randomization = Placebo).

Data: ../data/cleaned_rl_model_session2_placebo.csv, written by
utils/preprocessing.py:prepare_rl_dataset(session=2, treatment=0).

Notes:
  - Small group (6 subjects), so the subject-level sigmas will be weakly identified and
    lean on their priors.
  - The cue->reward mapping is reversed between sessions. The learner indexes values by
    reward magnitude and restarts at Q = 0, i.e. it treats Session 2 as fresh learning and
    ignores any carry-over from the Session 1 mapping.
"""

# Not in use (small group). Uncomment the lines below to fit this group.
# from rl_ddm import run

# if __name__ == "__main__":
#     run(data_path="../data/cleaned_rl_model_session2_placebo.csv",
#         run_name="rl_ddm_analytical_session2_placebo",
#         group_label="Session 2, Placebo")
