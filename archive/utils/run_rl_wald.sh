#!/bin/bash
#SBATCH --job-name=mid_hssm_rl_wald
#SBATCH -p l40s-gcondo            # Use the L40s GPU condo partition
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --time=24:00:00           # 24h; check sampling time on the first run
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4         # 4 CPUs for 4 numpyro chains
#SBATCH --output=logs/rl_wald_out_%j.txt
#SBATCH --error=logs/rl_wald_err_%j.txt

# 1. Load Anaconda
module load anaconda3/2023.09-0

# 2. Activate your environment
eval "$(conda shell.bash hook)"
conda activate hssm_env

# Assumes cleaned_rl_model_controls.csv already exists in data/.
# Run utils/preprocessing.py locally first if it doesn't exist.

# Step up to the project root, then into models
cd ../models

# Run the RL-Wald model (hssm >= 0.5.0 RLSSM)
python rl_wald.py
