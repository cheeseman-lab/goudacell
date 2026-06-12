# GoudaCell

Cell segmentation and feature extraction on the Whitehead HPC using Cellpose.

## Features

- **Segmentation modes**: nuclei-only, cells-only, or dual (both)
- **Feature extraction**: CellProfiler-equivalent morphological and intensity features
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
- **Foci**: count and area per channel (optional)

## Which Cellpose Version?

| Version | Install with | Use for |
|---------|--------------|---------|
| Cellpose 3 | `.[cellpose3]` | Round cells (most common) |
| Cellpose 4 | `.[cellpose4]` | Irregular/complex shapes |

## File Formats Supported

- TIFF (`.tif`, `.tiff`)
- Nikon ND2 (`.nd2`)
- DeltaVision (`.dv`)
