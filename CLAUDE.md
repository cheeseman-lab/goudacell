# CLAUDE.md

## Overview

GoudaCell is an HPC-compatible cell segmentation toolkit using Cellpose. Produces segmentation masks and morphological/intensity features from microscopy images.

Part of the **fry-python-tools** ecosystem — single-purpose GPU tools for the Whitehead HPC. See also: [emmentalembed](https://github.com/cheeseman-lab/emmentalembed) (protein embeddings + structure prediction).

## Project Structure

```
goudacell/
├── src/goudacell/              # Main package
│   ├── io.py                   # Image I/O (ND2, TIFF, DV, OME-Zarr)
│   ├── segment.py              # Cellpose segmentation (+ reconcile, cytoplasm, diameters)
│   ├── secondary_objects.py    # Secondary-object detection (vendored from brieflow)
│   ├── config.py               # YAML config handling
│   ├── cli.py                  # CLI entry point
│   ├── features.py             # Feature extraction dispatcher
│   ├── cp_emulator.py          # Built-in CP feature reimplementation
│   ├── features_second_objs.py # Secondary-object features (vendored from brieflow)
│   ├── custom_features.py      # User-registered per-cell features (vendored from brieflow)
│   ├── feature_extraction.py   # extract_features(_bare) helpers (vendored from brieflow)
│   ├── constants.py            # Column-order metadata (vendored from brieflow)
│   ├── features_cp_measure.py  # cp_measure backend
│   ├── features_cellprofiler.py # CellProfiler headless backend
│   ├── feature_table_utils.py  # Region property utilities
│   ├── gpu.py                  # GPU detection / diagnostics
│   ├── notebook.py             # ipywidgets ParameterUI (notebook front-end)
│   └── viz.py                  # Visualization utilities
├── data/                       # Put test images here
├── configs/                    # Generated configs (segmentation_config.yaml)
├── out/                        # Batch masks + feature tables
│   └── logs/                   # SLURM .out logs
├── notebooks/                  # Interactive notebook (thin: ParameterUI)
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

## Brieflow parity

GoudaCell's phenotype path must give the same masks and features as brieflow's for the same
inputs and parameters. It is in parity with **brieflow `zarr3` @ `6beb71a`**
(`6beb71a531022e117064e814e80998dd8f465b5f`); the operator surface and defaults follow
brieflow-analysis `marimo` `analysis/3_phenotype.py` @ `e2a386b`.

| brieflow (`workflow/`) | goudacell (`src/goudacell/`) |
|---|---|
| `lib/shared/segment_cellpose.py` (`create_cellpose_model`, `prepare_cellpose`, `segment_cellpose_rgb`, `segment_cellpose_nuclei_rgb`, `estimate_diameters`) | `segment.py` (`create_cellpose_model`, `prepare_cellpose`, `segment_nuclei_and_cells`, `segment_nuclei`, `estimate_diameters`) |
| `lib/shared/segmentation_utils.py` (`image_log_scale`, `reconcile_nuclei_cells`, `count_nuclei_per_cell`) | `segment.py` (same names) |
| `lib/phenotype/identify_cytoplasm_cellpose.py` | `segment.py` `identify_cytoplasm` (vectorized, same result) |
| `lib/external/cp_emulator.py`, `lib/shared/log_filter.py` | `cp_emulator.py` |
| `lib/shared/feature_table_utils.py`, `lib/shared/feature_extraction.py` | `feature_table_utils.py`, `feature_extraction.py` |
| `lib/phenotype/extract_phenotype_cp_emulator.py`, `constants.py` | `features.py` `extract_features` (cp_emulator path), `constants.py` |
| `lib/phenotype/extract_phenotype_cp_measure.py` | `features_cp_measure.py` |
| `lib/phenotype/custom_features.py` | `custom_features.py` (verbatim) |
| `lib/phenotype/segment_secondary_object.py` | `secondary_objects.py` (verbatim minus microfilm plotting) |
| `lib/phenotype/extract_phenotype_second_objs.py` | `features_second_objs.py` (verbatim) |
| `scripts/phenotype/extract_phenotype.py` (`num_nuclei`), `merge_second_objs_phenotype_cp.py` | `features.py` `add_num_nuclei`, `merge_second_obj_summary`; `cli.py` |
| `scripts/phenotype/identify_second_objs.py` | `cli.py` `segment_second_objects` |

Intentional differences: no StarDist/watershed primary segmentation (TensorFlow dependency);
the nuclei model is configurable (brieflow fixes `nuclei`/`cpsam`); the cp_measure backend
tolerates a failing measurement instead of dropping the rest of its group; secondary-object
nucleus distances key centroids by nucleus label; the dataclass default `reconcile` stays
`consensus` (the UI defaults to brieflow's `contained_in_cells`). goudacell-only extras (cells
mode, sweeps, channel/compartment subsets, CellProfiler backend) default to brieflow behaviour.

`tests/test_brieflow_parity.py` runs both implementations on the same synthetic inputs
(masks equal, feature columns equal, values equal) and pins a hash of every mapped brieflow
file, so any upstream change to them fails the test until it is reviewed:

```bash
git -C /path/to/brieflow fetch origin
git -C /path/to/brieflow worktree add --detach /path/to/brieflow-parity origin/zarr3
BRIEFLOW_LIB=/path/to/brieflow-parity pytest tests/test_brieflow_parity.py -v
```

Run it on a compute node (it runs Cellpose on CPU). The cp_measure test needs `cp_measure`
(`.[cp_measure]`, or run in a brieflow env with goudacell on `PYTHONPATH`). To move the pin:
diff the changed files against the pinned commit, port what changes masks, features or
parameters, then update `BRIEFLOW_COMMIT`, the hashes in the test, and the commit above.

## Running Tests

Tests are local-only except the brieflow parity test (see above).

```bash
BRIEFLOW_LIB=/path/to/brieflow pytest tests/test_brieflow_parity.py -v
```
