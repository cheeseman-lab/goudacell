#!/bin/bash
#SBATCH --job-name=goudacell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --partition=nvidia-A4000-20
#SBATCH --gres=gpu:1
#SBATCH --output=goudacell-%j.out

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

CONFIG_PATH="$(realpath "$1")"
CONFIG_DIR="$(dirname "$CONFIG_PATH")"

# Change to config directory (so relative paths in config work)
cd "$CONFIG_DIR"

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
