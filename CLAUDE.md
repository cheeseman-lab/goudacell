# CLAUDE.md

## Overview

GoudaCell is an HPC-compatible cell segmentation toolkit using Cellpose. Produces segmentation masks and morphological/intensity features from microscopy images.

Part of the **fry-python-tools** ecosystem — single-purpose GPU tools for the Whitehead HPC. See also: [emmentalembed](https://github.com/cheeseman-lab/emmentalembed) (protein embeddings + structure prediction).

## Project Structure

```
goudacell/
├── src/goudacell/              # Main package
│   ├── io.py                   # Image I/O (ND2, TIFF, DV, OME-Zarr)
│   ├── segment.py              # Segmentation adapters over brieflow (+ sweeps, cells mode)
│   ├── features.py             # Feature-extraction adapters over brieflow
│   ├── features_cellprofiler.py # CellProfiler headless backend (goudacell-only)
│   ├── environment.py          # Wrong-env check (stdlib only), run on `import goudacell`
│   ├── config.py               # YAML config handling
│   ├── cli.py                  # CLI entry point
│   ├── gpu.py                  # GPU detection / diagnostics
│   ├── notebook.py             # ipywidgets ParameterUI (notebook front-end)
│   ├── viz.py                  # Visualization utilities
│   ├── data/goudacell_default.cppipe # Default CellProfiler pipeline (template)
│   └── brieflow/               # brieflow's phenotype lib, vendored verbatim (do not edit)
├── scripts/sync_brieflow.py    # Re-vendors src/goudacell/brieflow at a brieflow commit
├── scripts/setup_cellprofiler_env.sh # Creates the goudacell_cp CellProfiler env
├── envs/cellprofiler.yml       # goudacell_cp env spec (CellProfiler 4.2.8.1, Python 3.9)
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
4. **Three extraction backends**: `cp_emulator` and `cp_measure` (brieflow's, vendored), `cellprofiler` (a user `.cppipe` run headless by a CellProfiler CLI in its own env)
5. **Zarr v3 / OME-NGFF v0.5**: Follows brieflow zarr3 patterns with pyramid generation

## CLI Commands

```bash
goudacell segment config.yaml      # Batch segmentation
goudacell single in.tif out.tif    # Single file
goudacell version                  # Check versions
```

## TODOs

- [x] Swap to zarr — OME-Zarr v3 read/write in io.py (zarr, ome-zarr, dask deps)
- [x] CellProfiler headless — CLI subprocess backend in features_cellprofiler.py
- [x] CPmeasure — cp_measure backend in features_cp_measure.py
- [ ] Subcellular embeddings — Extract embeddings from subcellular compartments (integrate last)

## Brieflow parity

goudacell runs brieflow's phenotype code unchanged. `src/goudacell/brieflow/` holds brieflow's
`workflow/lib` modules (and brieflow's MIT `LICENSE`) copied verbatim by
`scripts/sync_brieflow.py`, whose only change is
rewriting `lib.` imports to `goudacell.brieflow.`; the pinned commit is `BRIEFLOW_COMMIT` in
`src/goudacell/brieflow/__init__.py` (**brieflow `zarr3` @ `f03a2c8`**). Never edit the vendored
files: fix brieflow upstream, then re-sync. goudacell's own code is thin adapters that map its
config onto the calls brieflow-analysis's `marimo` phenotype notebook
(`analysis/3_phenotype.py`) makes, with that notebook's defaults in the UI.

| Phenotype notebook cell (brieflow call) | goudacell adapter |
|---|---|
| Segmentation parameters: `estimate_diameters` | `segment.estimate_diameters` (UI "Estimate diameters") |
| Segmentation: `segment_cellpose(..., cells=True)` | `segment.segment_nuclei_and_cells` (dual mode) |
| Segmentation: `segment_cellpose(..., cells=False)` | `segment.segment_nuclei` (nuclei mode) |
| Segmentation: `identify_cytoplasm_cellpose` | `segment.identify_cytoplasm` |
| Diameters from masks (cpsam / Cellpose 4) | `segment.derive_diameters` |
| Feature extraction: `extract_phenotype_cp_emulator` / `extract_phenotype_cp_measure` | `features.extract_features` |
| Custom features: `register_custom_features` / `load_custom_features` | `ParameterUI.set_custom_features`, `cli` |
| Secondary objects: `estimate_second_obj_diameter`, `segment_second_objs(_ml)` | `segment.segment_second_objects` (via `segment_second_objs_from_config`) |
| Secondary-object features: `extract_phenotype_second_objs` | `features.extract_second_obj_features` |
| brieflow scripts: `num_nuclei`, secondary-object summary merge | `features.add_num_nuclei`, `features.merge_second_obj_summary` |

Differences kept on purpose (goudacell-only options; the defaults are brieflow's behaviour):
`remove_edge_cells: false` calls `prepare_cellpose` + `segment_cellpose_rgb`/`_nuclei_rgb` with
`remove_edges=False` (brieflow always clears edges); `reconcile: null` (or any masks whose labels
don't pair, `segment.masks_reconciled`) gives no cytoplasm, with a warning, where brieflow's
`identify_cytoplasm_cellpose` raises or pairs unrelated objects; feature `channels`, `compartments` and the
texture/correlation/neighbor toggles select brieflow's per-compartment channel lists or drop
columns from brieflow's table (no compute saved); cells-only mode, sweeps and the CellProfiler
backend have no brieflow counterpart. `dual.nuclei_model` is still accepted but ignored with a
warning: brieflow segments nuclei with `nuclei` (Cellpose 3) or `cpsam` (Cellpose 4).

To re-pin: `git -C <brieflow> fetch origin && git -C <brieflow> checkout <commit>`, then
`python scripts/sync_brieflow.py --brieflow <brieflow>` (reads files at `--ref`, default `HEAD`,
and fails if a vendored module imports an unvendored one at module level: add it to
`MODULES`). Review
`git diff src/goudacell/brieflow`, adapt the adapters if a signature or default changed (compare
the phenotype notebook's cells), then run the parity test against that checkout.

`tests/test_brieflow_parity.py` checks that the vendored files equal brieflow's at the pin
(`--check` does the same from the command line) and that goudacell's adapters, API and CLI give
the same masks and feature tables as calling brieflow's functions directly, on synthetic masks
and with Cellpose on a real phenotype crop (brieflow's small test data) or a synthetic tile:

```bash
python scripts/sync_brieflow.py --brieflow /path/to/brieflow --check
BRIEFLOW_LIB=/path/to/brieflow pytest tests/test_brieflow_parity.py -v
```

Run it on a compute node (Cellpose on CPU). `GOUDACELL_PARITY_TILE` picks the phenotype image
(`GOUDACELL_PARITY_CHANNELS="3,1"` its DAPI and cytoplasm channels); the cp_measure test needs
`.[cp_measure]`.

## CellProfiler backend

`features_cellprofiler.py` stages `<channel>.tif` + `nuclei_mask.tif`/`cell_mask.tif`/
`cytoplasm_mask.tif` (relabeled 1..n so CellProfiler's `ObjectNumber` maps back to the mask
label), runs `cellprofiler -c -r -p -i -o -t` in a non-hidden `goudacell_cp_*` folder in the
cwd (CellProfiler's default Images filter skips dot-folders; `-t` keeps its temp files out
of /tmp), and joins the exported `Nuclei`/`Cells`/`Cytoplasm` CSVs on `label` with
`nucleus_`/`cell_`/`cytoplasm_` prefixes. A failed run or no object table raises. CellProfiler
(4.2.8.1, Python 3.9, OpenJDK from conda-forge) lives in its own env since it needs numpy<2:
`bash scripts/setup_cellprofiler_env.sh` creates `goudacell_cp` from `envs/cellprofiler.yml`
(`--solver=libmamba` when available; the classic solver can hang on it). `find_cellprofiler` resolves
an unset `cellprofiler_cmd`: `GOUDACELL_CELLPROFILER` → `cellprofiler` on PATH → the
`goudacell_cp` env's `bin/cellprofiler` (conda base from `CONDA_EXE`/`sys.prefix`, then
`conda env list --json`), never activating anything. The bare `cellprofiler_cmd: cellprofiler` that 0.3 wrote into
configs counts as unset (logged once) when no `cellprofiler` is on PATH; any other explicit
command wins over discovery. An unset `pipeline_file` runs
`default_pipeline`, which fills `data/goudacell_default.cppipe` for the staged channels and
masks and drops MeasureTexture / MeasureColocalization / MeasureObjectNeighbors per
`include_texture` / `include_correlation` / `include_neighbors`. `check_cellprofiler` runs
`<cmd> --version` (passing commands cached) and raises unless it's CellProfiler 4.2.x, naming
the command and where it came from; the backend, the notebook's CP command status / config
generation, and `goudacell segment` (before any segmentation) all go through it.

## Environment check

`environment.check_environment()` runs at the top of `goudacell/__init__.py`, before the heavy
imports (stdlib only, Python-3.9-safe), and raises if the interpreter is the `goudacell_cp`
CellProfiler env (by name), Python < 3.10, or numpy < 2 (named as a CellProfiler env when
`cellprofiler` is importable there; an importable CellProfiler alone is fine); `ParameterUI` also requires Cellpose. The
notebook's first cell turns a missing `goudacell` (the CellProfiler kernel can't import it)
into the same "switch the kernel to goudacell" error.

## Running Tests

Tests are local-only except the brieflow parity test (see above),
`tests/test_cellprofiler_backend.py` (its CellProfiler runs skip when none is found) and
`tests/test_environment.py`.

```bash
BRIEFLOW_LIB=/path/to/brieflow pytest tests/test_brieflow_parity.py -v
pytest tests/test_cellprofiler_backend.py -v   # finds the goudacell_cp env by itself
```
