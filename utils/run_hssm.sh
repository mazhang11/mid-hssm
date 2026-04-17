#!/bin/bash
#SBATCH --job-name=mid_hssm
#SBATCH --time=1:00:00          # Gives it 1 hour to run 
#SBATCH --mem=32G                # Requests 32 GB of RAM
#SBATCH --cpus-per-task=4        # Requests 4 CPUs (matches cores=4 in python)
#SBATCH --output=hssm_out_%j.txt # Saves your print() statements here
#SBATCH --error=hssm_err_%j.txt  # Saves python errors here

# 1. Load Anaconda 
module load anaconda3/2023.09-0

# 2. Activate your environment 
source activate hssm_env

# 3. Run the scripts
# (Assuming your terminal is in the utils folder when you submit the job)

# Run data cleaning (located right here in the utils folder)
python preprocessing.py

# Step back up to the main repo folder, then down into models
cd ../models
python secondpass.py