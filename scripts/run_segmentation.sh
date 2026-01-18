#!/bin/bash
#SBATCH --job-name=goudacell
#SBATCH --output=goudacell_%j.out
#SBATCH --error=goudacell_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# GoudaCell Batch Segmentation SLURM Script
#
# Usage:
#   sbatch run_segmentation.sh /path/to/config.yaml
#
# Make sure to activate your conda environment before submitting,
# or modify the conda activation below.

# Fail on errors
set -e

# Check for config argument
if [ -z "$1" ]; then
    echo "Usage: sbatch run_segmentation.sh /path/to/config.yaml"
    exit 1
fi

CONFIG_PATH="$1"

# Activate conda environment
# Modify this to match your environment name
source ~/.bashrc
conda activate goudacell

# Print environment info
echo "================================================"
echo "GoudaCell Batch Segmentation"
echo "================================================"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Config: ${CONFIG_PATH}"
echo "Python: $(which python)"
echo ""

# Check GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"
echo ""

# Run segmentation
echo "Starting segmentation..."
goudacell segment "${CONFIG_PATH}"

echo ""
echo "================================================"
echo "Completed: $(date)"
echo "================================================"
