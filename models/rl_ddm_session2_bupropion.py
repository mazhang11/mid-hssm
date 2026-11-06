"""
rl_ddm_session2_bupropion.py

The analytical RL-DDM from rl_ddm.py (same learner, priors, masking and plots), fit on
Session 2, Bupropion XL-arm subjects (Phase 2 Randomization = Bupropion XL).

Data: ../data/cleaned_rl_model_session2_bupropion.csv, written by
utils/preprocessing.py:prepare_rl_dataset(session=2, treatment=1).

Notes:
  - The group mixes subjects who switched from Placebo in Session 1 (most) with subjects
    on Bupropion in both sessions; the CSV's per-subject Session 1 arm is not stored, so
    use mid_patient_data_summary.csv to split them if needed.
  - The cue->reward mapping is reversed between sessions. The learner indexes values by
    reward magnitude and restarts at Q = 0, i.e. it treats Session 2 as fresh learning and
    ignores any carry-over from the Session 1 mapping.
"""

from rl_ddm import run

if __name__ == "__main__":
    run(data_path="../data/cleaned_rl_model_session2_bupropion.csv",
        run_name="rl_ddm_analytical_session2_bupropion",
        group_label="Session 2, Bupropion XL")
