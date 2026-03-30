# CLAUDE.md

## Overview

GoudaCell is an HPC-compatible cell segmentation toolkit using Cellpose. Produces segmentation masks and morphological/intensity features from microscopy images.

Part of the **fry-python-tools** ecosystem — single-purpose GPU tools for the Whitehead HPC. See also: [emmentalembed](https://github.com/cheeseman-lab/emmentalembed) (protein embeddings + structure prediction).

## Project Structure

```
goudacell/
├── src/goudacell/              # Main package
│   ├── io.py                   # Image I/O (ND2, TIFF, DV, OME-Zarr)
│   ├── segment.py              # Cellpose segmentation
│   ├── config.py               # YAML config handling
│   ├── cli.py                  # CLI entry point
│   ├── features.py             # Feature extraction dispatcher
│   ├── cp_emulator.py          # Built-in CP feature reimplementation
│   ├── features_cp_measure.py  # cp_measure backend
│   ├── features_cellprofiler.py # CellProfiler headless backend
│   ├── feature_table_utils.py  # Region property utilities
│   └── viz.py                  # Visualization utilities
├── data/                       # Put test images here
├── notebooks/                  # Interactive notebook
└── scripts/                    # SLURM submission scripts
```

## Development Setup

```bash
conda create -n goudacell -c conda-forge python=3.11 uv pip -y
conda activate goudacell
uv pip install -e ".[cellpose3]"
```

## Key Design Decisions

1. **Cellpose Version Detection**: Auto-detects version and validates model compatibility
2. **Notebook generates configs**: No manual YAML editing needed
3. **File Format Support**: ND2 (`nd2`), TIFF (`tifffile`), DV (`mrc`), OME-Zarr (`zarr`/`ome-zarr`)
4. **Three extraction backends**: `cp_emulator` (built-in), `cp_measure` (lightweight), `cellprofiler` (headless CP-core)
5. **Zarr v3 / OME-NGFF v0.5**: Follows brieflow zarr3 patterns with pyramid generation

## CLI Commands

```bash
goudacell segment config.yaml      # Batch segmentation
goudacell single in.tif out.tif    # Single file
goudacell version                  # Check versions
```

## TODOs

- [x] Swap to zarr — OME-Zarr v3 read/write in io.py (zarr, ome-zarr, dask deps)
- [x] CellProfiler headless — cellprofiler-core backend in features_cellprofiler.py
- [x] CPmeasure — cp_measure backend in features_cp_measure.py
- [ ] Subcellular embeddings — Extract embeddings from subcellular compartments (integrate last)

## Running Tests

```bash
pytest tests/ -v                   # Full test suite
pytest tests/test_install.py -v    # Smoke test (import, CLI, deps)
pytest tests/test_config.py -v     # Config loading/roundtrip
```
