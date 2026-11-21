"""
rl_ddm_session1_bupropion.py

The analytical RL-DDM from rl_ddm.py (same learner, priors, masking and plots), fit on
Session 1, Bupropion XL-arm subjects (Phase 1 Randomization = Bupropion XL).

Data: ../data/cleaned_rl_model_session1_bupropion.csv, written by
utils/preprocessing.py:prepare_rl_dataset(session=1, treatment=1).

Notes:
  - Small group (9 subjects), so the subject-level sigmas will be weakly identified and
    lean on their priors.
  - Same session as rl_ddm.py's controls, so the two fits are directly comparable.
"""

# Not in use (small group). Uncomment the lines below to fit this group.
# from rl_ddm import run

# if __name__ == "__main__":
#     run(data_path="../data/cleaned_rl_model_session1_bupropion.csv",
#         run_name="rl_ddm_analytical_session1_bupropion",
#         group_label="Session 1, Bupropion XL")
