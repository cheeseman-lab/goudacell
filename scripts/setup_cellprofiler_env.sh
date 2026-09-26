#!/bin/bash
# Create the goudacell_cp conda env that goudacell's `cellprofiler` backend finds by itself.
#
# Usage (from the repo root; the solve is heavy, so on a compute node via srun on a cluster):
#   bash scripts/setup_cellprofiler_env.sh
#
# Idempotent: an existing goudacell_cp env is only checked, not rebuilt. The libmamba solver
# finishes in minutes where the classic solver can hang.

set -eo pipefail

ENV_NAME=goudacell_cp
SPEC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/envs/cellprofiler.yml"

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda env $ENV_NAME already exists"
else
    conda env create --solver=libmamba -f "$SPEC"
fi

conda activate "$ENV_NAME"
echo "CellProfiler: $(command -v cellprofiler)"
python -c "import cellprofiler; print('version', cellprofiler.__version__)"
