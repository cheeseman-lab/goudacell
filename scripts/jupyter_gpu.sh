#!/bin/bash
#SBATCH --job-name=goudacell_jupyter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --partition=nvidia-A4000-20
#SBATCH --gres=gpu:1
#SBATCH --output=goudacell_jupyter-%j.out

# GoudaCell Jupyter Lab SLURM Script
#
# Usage:
#   cd /path/to/goudacell
#   sbatch scripts/jupyter_gpu.sh
#
# The notebook will open in the directory where you ran sbatch.

# Activate conda environment
source ~/.bashrc
conda activate goudacell

# Workaround for jupyter bug
unset XDG_RUNTIME_DIR

# Get the directory where sbatch was run from
NOTEBOOK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

jupyter-lab \
    --no-browser \
    --port-retries=0 \
    --ip=0.0.0.0 \
    --port=$(shuf -i 8900-10000 -n 1) \
    --notebook-dir="${NOTEBOOK_DIR}"
