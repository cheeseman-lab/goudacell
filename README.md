# GoudaCell

Cell segmentation and feature extraction on the Whitehead HPC using Cellpose.

## Features

- **Segmentation modes**: nuclei-only, cells-only, or dual (both)
- **Secondary objects**: detect objects inside cells (pathogens, organelles) by thresholding or Cellpose
- **Feature extraction**: CellProfiler-equivalent morphological and intensity features, plus your own custom features
- **Brieflow inside**: runs [brieflow](https://github.com/cheeseman-lab/brieflow)'s phenotype code (vendored unchanged), so masks and features match brieflow's for the same parameters
- **File formats**: TIFF, Nikon ND2, DeltaVision (.dv)
- **Cellpose 3 & 4**: Supports both versions with automatic model selection

GoudaCell is a **single-shot tool** — run it once on your images to produce segmentation masks and features, then use the outputs in your downstream analysis. Part of the [fry-python-tools](https://github.com/cheeseman-lab) ecosystem (see also: [emmentalembed](https://github.com/cheeseman-lab/emmentalembed) for protein embeddings).

### What to do with the outputs

**Segmentation masks** (TIFF label images):
- Morphological profiling — extract per-cell features and cluster phenotypes ([Bray et al. 2016, Nature Protocols](https://doi.org/10.1038/nprot.2016.105))
- Perturbation scoring — compare feature distributions between control and perturbed cells ([Celik et al. 2024, eLife](https://elifesciences.org/reviewed-preprints/94964))
- Single-cell tracking — link masks across timepoints for live-cell analysis
- Quality control — filter segmented objects by size, shape, or intensity

**Extracted features** (CSV, ~100+ features per cell):
- Dimensionality reduction — PCA/UMAP on feature space for phenotype discovery
- Classification — train models to distinguish cell states or drug responses
- Correlation analysis — link morphological features to genetic perturbations

## Getting Started

### 1. Set Up Your Environment (one time)

```bash
# Clone the repository on fry
git clone https://github.com/cheeseman-lab/goudacell.git
cd goudacell

# Create the environment
conda create -n goudacell -c conda-forge python=3.11 uv pip -y
conda activate goudacell

# Install goudacell (choose ONE):
uv pip install -e ".[cellpose3]"  # For most cells (rounded shapes)
uv pip install -e ".[cellpose4]"  # For complex cell shapes
# Both extras pin torch to the cu126 wheel to match the fry GPU driver (CUDA 12.6).
# Check the GPU is usable with: goudacell version  (or the notebook's GPU banner)

# Register as a Jupyter kernel
python -m ipykernel install --user --name goudacell --display-name "goudacell"
```

### 2. Test Parameters Interactively

```bash
# Start Jupyter on a GPU node (run from goudacell directory)
cd /path/to/goudacell
sbatch scripts/jupyter_gpu.sh

# Check the output file for the URL
cat goudacell_jupyter-*.out
```

Open the notebook at `notebooks/segmentation.ipynb` and:
1. Set your image directory and file pattern
2. Choose segmentation mode: `"nuclei"`, `"cells"`, or `"dual"`
3. Adjust parameters (diameter, thresholds) using the sweep cells
4. Run feature extraction (optional)
5. Generate batch config when happy with results

### 3. Run Batch Segmentation

```bash
# Use the config the notebook wrote to configs/ (run from the repo root)
sbatch scripts/run_segmentation.sh configs/segmentation_config.yaml
```

### Project layout

Generated artifacts are kept out of the source tree:

| Folder | Contents |
|--------|----------|
| `data/` | Your input images |
| `configs/` | Configs written by the notebook (`segmentation_config.yaml`) |
| `out/` | Masks + feature tables from batch runs |
| `out/logs/` | SLURM `.out` logs |

Configs carry absolute input/output paths, so a config works no matter where it
lives or where you launch from. Run `sbatch` from the repo root so `out/logs/`
exists for the job logs.

## Segmentation Modes

| Mode | Output | Use case |
|------|--------|----------|
| `nuclei` | `*_nuclei_mask.tif` | Nuclear segmentation only |
| `cells` | `*_mask.tif` | Cell segmentation only |
| `dual` | `*_nuclei_mask.tif` + `*_cell_mask.tif` | Both nuclei and cells |

## Feature Extraction

Extract CellProfiler-equivalent features from segmented images (~100+ features per compartment):

- **Intensity**: mean, std, min, max, median, quartiles, edge intensities
- **Shape**: area, perimeter, solidity, eccentricity, Zernike/Hu moments
- **Texture**: Haralick (13), PFTAS (54)
- **Distribution**: radial intensity distribution
- **Correlation**: channel correlation, colocalization metrics
- **Neighbors**: counts, distances, angles
- **Foci**: count and area per channel (optional, `feature_extraction.foci_channel`)
- **Custom**: your own per-cell measurements (`ui.set_custom_features([...])` in the notebook)

With secondary-object detection on (dual mode), each image also gets a
`*_second_obj_mask.tif`, a per-object `*_second_obj_features.csv`, and per-cell object
counts/areas merged into the main feature table.

### CellProfiler backend (headless)

`feature_extraction.method: cellprofiler` measures each image's masks with a real
CellProfiler pipeline, run headless, instead of the built-in extractor.

#### Why a second conda env

CellProfiler 4.2 needs Python 3.9 and numpy<2, and goudacell needs Python ≥3.10 and
numpy≥2, so the two can't share an env. goudacell never imports CellProfiler: it runs the
`cellprofiler` command from a separate env as a subprocess. You keep working (notebook
kernel, CLI) in the `goudacell` env, and only need the CellProfiler env to exist.

#### 1. Create the CellProfiler env (once)

From the repo root:

```bash
bash scripts/setup_cellprofiler_env.sh
```

This creates the conda env `goudacell_cp` from `envs/cellprofiler.yml` (CellProfiler
4.2.8.1, Python 3.9 and Java, from conda-forge and bioconda) and checks it. If the env
already exists, the script only checks it. The solve is heavy, so on a shared cluster run
the script on a compute node. It takes a few minutes with the libmamba solver. If you'd
rather run conda yourself, this is the equivalent command:

```bash
conda env create --solver=libmamba -f envs/cellprofiler.yml
```

conda ≥23.10 uses libmamba by default. On an older conda, install the solver into base
with `conda install -n base -c conda-forge conda-libmamba-solver`. If you can't install
it, drop `--solver=libmamba`: the classic solver still works but can take a very long time.

#### 2. Check it

```bash
conda run -n goudacell_cp cellprofiler --version   # prints 4.2.8.1 (after some warnings)
```

#### 3. Use it

Select the `cellprofiler` method in the notebook, or in the config:

```yaml
feature_extraction:
  enabled: true
  method: cellprofiler
```

Nothing else is needed. goudacell looks for CellProfiler in this order:

1. `cellprofiler_cmd` in the config (the notebook's "CP command" field), if set;
2. the `GOUDACELL_CELLPROFILER` environment variable;
3. a `cellprofiler` command on your PATH;
4. the `goudacell_cp` conda env, found through conda without activating it.

Before running, goudacell checks the command with `cellprofiler --version`. It must be
CellProfiler 4.2.x; if not, the notebook and the CLI stop with an error that says which
command was found and what to do. To use a CellProfiler installed some other way (another
env name, a module, a container wrapper), point goudacell at its executable:

```bash
export GOUDACELL_CELLPROFILER=/path/to/cellprofiler-env/bin/cellprofiler
```

or set `cellprofiler_cmd: /path/to/cellprofiler-env/bin/cellprofiler` in the config.

#### The pipeline

Without a `pipeline_file`, goudacell runs its default pipeline
(`src/goudacell/data/goudacell_default.cppipe`, filled in for your channels and masks).
It runs these modules:

- MeasureObjectIntensity
- MeasureObjectSizeShape (with Zernike)
- MeasureTexture (scale 3)
- MeasureColocalization (within objects, all channel pairs)
- MeasureObjectNeighbors (adjacent nuclei and cells)
- ExportToSpreadsheet

The Texture, Correlation and Neighbors switches turn their modules off. To substitute your
own pipeline, set `pipeline_file` (the notebook's "CP pipeline" field) to a `.cppipe`. As a
starting point, `goudacell.features_cellprofiler.default_pipeline(["DAPI", "GFP"])` returns
the default's text for your channels; save it as a `.cppipe` and open it in the CellProfiler
GUI.

For each image, goudacell writes one input folder:

- every channel as `<channel name>.tif` (the `channel_names`);
- the masks as `nuclei_mask.tif`, `cell_mask.tif` and `cytoplasm_mask.tif`.

It then runs `cellprofiler -c -r -p <pipeline> -i <input> -o <output>`. A pipeline of your
own must follow the same layout:

- In NamesAndTypes, assign each channel file as a grayscale image, and the masks as
  **Objects** named `Nuclei`, `Cells` and `Cytoplasm`.
- Add the Measure modules you want.
- End with ExportToSpreadsheet (CSV, one file per object).

goudacell joins those three object tables on the mask `label`. The columns are
`nucleus_`/`cell_`/`cytoplasm_` plus CellProfiler's names (e.g.
`cell_Intensity_MeanIntensity_GFP`), with intensities scaled to 0–1. A pipeline that loads a
mask the mode doesn't produce (e.g. `Cells` in nuclei mode) finds no image set and fails
with the list of staged files. `pytest tests/test_cellprofiler_backend.py` runs a minimal
pipeline and the default one with the CellProfiler it finds, and skips those tests when it
finds none.

#### Troubleshooting

- **"this notebook runs in the `goudacell` env…"**: the notebook kernel is the CellProfiler
  env. Switch the kernel to `goudacell`.
- **Solver slow or hanging**: use libmamba (see step 1); the classic solver can run for a
  very long time on this recipe.
- **SSL/certificate errors while solving or downloading**: this is usually a proxy or an
  outdated CA bundle. Try `conda update -n base ca-certificates certifi`, or point conda at
  your institution's CA bundle with `conda config --set ssl_verify /path/to/ca-bundle.crt`.
  If a separately installed `mamba` fails this way, use `conda ... --solver=libmamba`
  instead.
- **Java not found / "JVM" errors when CellProfiler reads images**: the env needs its own
  Java. Run `conda install -n goudacell_cp -c conda-forge openjdk`.
- **Wrong CellProfiler version**: goudacell supports 4.2.x. Check which command it found
  (the error names it), unset or fix `GOUDACELL_CELLPROFILER`/`cellprofiler_cmd`, or rebuild
  the env with `conda env remove -n goudacell_cp` and rerun the setup script.
- **Warnings on `cellprofiler --version`** (pkg_resources, SciPy/NumPy version) are harmless.

## Which Cellpose Version?

| Version | Install with | Use for |
|---------|--------------|---------|
| Cellpose 3 | `.[cellpose3]` | Round cells (most common) |
| Cellpose 4 | `.[cellpose4]` | Irregular/complex shapes |

Either version also accepts a path to a custom trained model in place of a model name.

## File Formats Supported

- TIFF (`.tif`, `.tiff`)
- Nikon ND2 (`.nd2`)
- DeltaVision (`.dv`)
