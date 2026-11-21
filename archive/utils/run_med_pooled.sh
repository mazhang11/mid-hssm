#!/bin/bash
#SBATCH --job-name=mid_hssm_med_pooled
#SBATCH -p l40s-gcondo            # Use the L40s GPU condo partition
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --time=36:00:00           # More covariates = longer sampling; give it 36 hours
#SBATCH --mem=48G                 # Larger memory for the augmented model
#SBATCH --cpus-per-task=4         # 4 CPUs for 4 numpyro chains
#SBATCH --output=logs/med_pooled_out_%j.txt
#SBATCH --error=logs/med_pooled_err_%j.txt

# 1. Load Anaconda
module load anaconda3/2023.09-0

# 2. Activate your environment
# CORRECTED: Using the proper conda hook so the script doesn't fail
eval "$(conda shell.bash hook)"
conda activate hssm_env

# We assume the cleaned CSV (with treatment/is_responder columns) already exists in data/.
# Make sure to run utils/preprocessing.py locally BEFORE submitting this job.

# Step back up to the main repo folder, then down into models
cd ../models

# Run the medication-augmented pooled model
python med_pooled.py
