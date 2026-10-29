#!/bin/bash
#SBATCH --job-name=mid_hssm_rl_ddm
#SBATCH -p l40s-gcondo            # Use the L40s GPU condo partition
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --time=24:00:00           # 24h; check sampling time on the first run
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4         # 4 CPUs for 4 numpyro chains
#SBATCH --output=logs/rl_ddm_out_%j.txt
#SBATCH --error=logs/rl_ddm_err_%j.txt

# 1. Load conda (the anaconda3 module is no longer usable on Oscar; CCV recommends miniforge3)
module load miniforge3/25.3.0-3

# 2. Activate your environment
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh   # per CCV docs (not conda init)
conda activate hssm_env

# Assumes cleaned_rl_model_controls.csv already exists in data/.
# Run utils/preprocessing.py locally first if it doesn't exist.

# Step up to the project root, then into models
cd ../models

# Record the node, the GPU, and whether JAX can see it (should list a CudaDevice, not CpuDevice)
echo "Node: $(hostname)   Start: $(date)"
nvidia-smi
python -c "import hssm, jax; print('hssm', hssm.__version__, '| jax devices:', jax.devices())"

# Run the analytical RL-DDM (needs hssm >= 0.5.1: install from GitHub, see below)
# pip install "hssm @ git+https://github.com/lnccbrown/HSSM.git@fb9c4bd6"
# -u: unbuffered, so print() output reaches the log while the job runs, not only at the end
python -u rl_ddm.py
echo "End: $(date)"
