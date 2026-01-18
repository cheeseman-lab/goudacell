#!/bin/bash
#SBATCH --job-name=jupyter_gpu
#SBATCH --output=jupyter_gpu_%j.out
#SBATCH --error=jupyter_gpu_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Jupyter Lab on GPU Node
#
# Usage:
#   sbatch jupyter_gpu.sh
#
# After submitting, check the output file for connection instructions.
# You'll need to set up SSH tunneling to access the notebook.

# Fail on errors
set -e

# Activate conda environment
source ~/.bashrc
conda activate goudacell

# Get node and port info
NODE=$(hostname)
PORT=$(shuf -i 8888-9999 -n 1)

echo "================================================"
echo "Jupyter Lab on GPU Node"
echo "================================================"
echo "Date: $(date)"
echo "Node: ${NODE}"
echo "Port: ${PORT}"
echo ""

# Check GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"
echo ""

echo "================================================"
echo "CONNECTION INSTRUCTIONS"
echo "================================================"
echo ""
echo "1. Open a NEW terminal on your local machine"
echo ""
echo "2. Run this SSH tunnel command:"
echo "   ssh -N -L ${PORT}:${NODE}:${PORT} YOUR_USERNAME@CLUSTER_LOGIN_NODE"
echo ""
echo "3. Open your browser to:"
echo "   http://localhost:${PORT}"
echo ""
echo "4. Use the token shown below to log in"
echo ""
echo "================================================"
echo ""

# Start Jupyter Lab
jupyter lab \
    --no-browser \
    --ip=0.0.0.0 \
    --port=${PORT} \
    --NotebookApp.token='' \
    --NotebookApp.password=''
