#!/bin/bash
# Create the goudacell_cp conda env that goudacell's `cellprofiler` backend finds by itself.
#
# Usage (from anywhere; on a cluster, run it on a compute node since the solve is heavy):
#   bash scripts/setup_cellprofiler_env.sh
#
# Idempotent: an existing goudacell_cp env is only checked, not rebuilt. Uses the libmamba
# solver when conda has it (the classic solver can take very long on this recipe).

set -eo pipefail

ENV_NAME=goudacell_cp
SUPPORTED=4.2.
SPEC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/envs/cellprofiler.yml"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

if ! command -v conda >/dev/null 2>&1; then
    fail "conda not found on PATH. Install Miniforge (https://github.com/conda-forge/miniforge)" \
        "or load your site's conda, then rerun."
fi
[ -f "$SPEC" ] || fail "env spec not found: $SPEC"
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda env $ENV_NAME already exists; checking it (to rebuild it:" \
        "conda env remove -n $ENV_NAME, then rerun)"
else
    SOLVER=()
    if "${CONDA_PYTHON_EXE:-python}" -c "import conda_libmamba_solver" >/dev/null 2>&1; then
        SOLVER=(--solver=libmamba)
    else
        echo "NOTE: conda-libmamba-solver not found, using the classic solver (can take very" \
            "long). To speed it up: conda install -n base -c conda-forge conda-libmamba-solver"
    fi
    echo "Creating $ENV_NAME from $SPEC (a few minutes)..."
    conda env create "${SOLVER[@]}" -f "$SPEC" \
        || fail "conda env create failed; see Troubleshooting in the README's CellProfiler section."
fi

conda activate "$ENV_NAME"
CELLPROFILER="$(command -v cellprofiler || true)"
[ -n "$CELLPROFILER" ] || fail "$ENV_NAME has no cellprofiler executable. Rebuild it:" \
    "conda env remove -n $ENV_NAME, then rerun this script."
# CellProfiler prints import warnings on stderr; the version is its last stdout line
VERSION="$(cellprofiler --version 2>/dev/null | tail -n 1 || true)"
case "$VERSION" in
    "$SUPPORTED"*) ;;
    *) fail "$CELLPROFILER reports version '${VERSION:-none}', goudacell needs ${SUPPORTED}x." \
        "Rebuild the env: conda env remove -n $ENV_NAME, then rerun this script." ;;
esac
command -v java >/dev/null 2>&1 || fail "Java not found in $ENV_NAME (CellProfiler needs it" \
    "to read images). Install it: conda install -n $ENV_NAME -c conda-forge openjdk"

echo "OK: CellProfiler $VERSION at $CELLPROFILER (Java: $(command -v java))"
echo "goudacell finds this env by itself; select the 'cellprofiler' feature method."
