#!/bin/bash
#SBATCH --job-name=mid_hssm_array
#SBATCH --time=12:00:00           # Gives it 12 hours to run 
#SBATCH --mem=8G                  # Reduced to 8GB since it is only 1 subject
#SBATCH --cpus-per-task=4         # Requests 4 CPUs (matches cores=4 in python)
#SBATCH --array=0-49              # Creates 50 simultaneous jobs (IDs 0 through 49)
#SBATCH --output=logs/hssm_out_%A_%a.txt # %A is the Job ID, %a is the Array ID
#SBATCH --error=logs/hssm_err_%A_%a.txt  

# 1. Load Anaconda 
module load anaconda3/2023.09-0

# 2. Activate your environment 
source activate hssm_env

# 3. Run the scripts
# (Assuming your terminal is in the utils folder when you submit the job)

# Run data cleaning ONLY on the very first array task to prevent file read/write collisions
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    echo "Task 0: Running preprocessing..."
    python preprocessing.py
else
    # Give Task 0 a 30-second head start to finish creating the cleaned CSV
    echo "Task $SLURM_ARRAY_TASK_ID: Waiting 30 seconds for Task 0 to finish preprocessing..."
    sleep 30
fi

# Step back up to the main repo folder, then down into models
cd ../models

# Run the script and pass the unique array ID as an argument to Python
python secondpass.py $SLURM_ARRAY_TASK_ID