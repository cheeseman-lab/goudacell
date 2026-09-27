#!/bin/bash
#SBATCH --job-name=goudacell_jupyter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --output=out/logs/goudacell_jupyter-%j.out

# GoudaCell Jupyter Lab SLURM Script
#
# Usage:
#   cd /path/to/goudacell
#   sbatch --partition=<gpu-partition> scripts/jupyter_gpu.sh
#
# No partition is set here: pass your cluster's GPU partition with --partition, or
# export SBATCH_PARTITION=<gpu-partition> (list partitions with: sinfo -o "%P %G").
# GOUDACELL_ENV picks the conda env (default: goudacell).
#
# The notebook will open in the directory where you ran sbatch.

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate "${GOUDACELL_ENV:-goudacell}"

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
