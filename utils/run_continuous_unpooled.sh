#!/bin/bash
#SBATCH --job-name=mid_hssm_array
#SBATCH -p l40s-gcondo            # Use the L40s GPU partition
#SBATCH --gres=gpu:1              # Request 1 GPU 
#SBATCH --time=12:00:00           # Gives it 12 hours to run 
#SBATCH --mem=8G                  # 8GB per subject
#SBATCH --cpus-per-task=4         # 4 CPUs for numpyro chains
#SBATCH --array=0-49              # Creates 50 simultaneous jobs (IDs 0 through 49)
#SBATCH --output=logs/hssm_out_%A_%a.txt # %A is the Job ID, %a is the Array ID
#SBATCH --error=logs/hssm_err_%A_%a.txt  

# 1. Load Anaconda 
module load anaconda3/2023.09-0

# 2. Activate your environment 
# CORRECTED: Using the proper conda hook so the script doesn't fail
eval "$(conda shell.bash hook)"
conda activate hssm_env

# 3. Run the scripts
# (Assuming your terminal is in the utils folder when you submit the job)

# We assume the cleaned CSV with covariates already exists in data/
# Preprocessing is run manually before submitting the job.

# Step back up to the main repo folder, then down into models
cd ../models

# Run the script and pass the unique array ID as an argument to Python
python continuous_unpooled.py $SLURM_ARRAY_TASK_ID