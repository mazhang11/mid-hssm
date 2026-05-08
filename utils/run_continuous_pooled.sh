#!/bin/bash
#SBATCH --job-name=mid_hssm_pooled
#SBATCH -p l40s-gcondo            # Use the L40s GPU partition
#SBATCH --gres=gpu:1              # Requests 1 GPU
#SBATCH --time=24:00:00           # Pooled models take much longer to sample
#SBATCH --mem=32G                 # Increased memory to hold traces for all subjects
#SBATCH --cpus-per-task=4         # Requests 4 CPUs for 4 PyMC chains
#SBATCH --output=logs/pooled_out_%j.txt 
#SBATCH --error=logs/pooled_err_%j.txt  

# 1. Load Anaconda 
module load anaconda3/2023.09-0

# 2. Activate your environment 
# CORRECTED: Using the proper conda hook so the script doesn't fail
eval "$(conda shell.bash hook)"
conda activate hssm_env

# We assume the cleaned CSV with covariates already exists in data/
# Preprocessing is run manually before submitting the job.

# Step back up to the main repo folder, then down into models
cd ../models

# Run the pooled script
python continuous_pooled.py