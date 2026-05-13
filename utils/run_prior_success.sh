#!/bin/bash
#SBATCH --job-name=mid_hssm_prior_success
#SBATCH -p l40s-gcondo            # Use the L40s GPU condo partition
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --time=24:00:00           # 24h should be sufficient for this simpler model
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4         # 4 CPUs for 4 numpyro chains
#SBATCH --output=logs/prior_success_out_%j.txt
#SBATCH --error=logs/prior_success_err_%j.txt

# 1. Load Anaconda
module load anaconda3/2023.09-0

# 2. Activate your environment
eval "$(conda shell.bash hook)"
conda activate hssm_env

# Assumes mid_session1_controls_hssm.csv already exists in data/.
# Run utils/preprocessing.py locally first if it doesn't exist.

# Step up to the project root, then into models
cd ../models

# Run the prior-success pooled model
python prior_success.py
