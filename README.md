# GoudaCell

Check and tune cell segmentation and secondary-object detection on your images before an
optical pooled screen (OPS). GoudaCell runs [brieflow](https://github.com/cheeseman-lab/brieflow)'s
phenotype code unchanged, so what you tune here is what brieflow produces. It also works on its
own to segment cells and extract features, on a Slurm cluster or any GPU machine.

- **Segmentation:** nuclei, cells, or both (dual), with Cellpose 3 or 4.
- **Secondary objects:** objects inside cells (foci, organelles, pathogens), by threshold,
  Cellpose or StarDist.
- **Features:** CellProfiler-equivalent per-cell features, plus your own.
- **Labeler:** a one-file browser page to check masks, and correct them for scoring or
  fine-tuning.

## Workflow

### 1. Install (once)

```bash
git clone https://github.com/cheeseman-lab/goudacell.git && cd goudacell
conda create -n goudacell -c conda-forge python=3.11 uv pip -y
conda activate goudacell
uv pip install -e ".[cellpose]"   # Cellpose 3.1.0 + 4; torch cu126 (NVIDIA driver >= 560)
python -m ipykernel install --user --name goudacell --display-name "goudacell"
goudacell version                 # checks the GPU
```

### 2. Tune parameters in the notebook

```bash
export SBATCH_PARTITION=<gpu-partition>   # see: sinfo -o "%P %G"
sbatch scripts/jupyter_gpu.sh             # from the repo root
cat out/logs/goudacell_jupyter-*.out      # the Jupyter URL
```

Open `notebooks/segmentation.ipynb`, then:

1. Point it at your images.
2. Pick a mode.
3. Sweep diameters and thresholds on previews.
4. Optionally add secondary objects and features.
5. **Generate config**. It writes `configs/<name>.yaml` next to your data folder.

### 3. Run on all images

```bash
sbatch scripts/run_segmentation.sh configs/<name>.yaml
```

- Masks go to `masks/<name>/` and features to `features/<name>/`, both next to your data folder.
- A failed file doesn't stop the run; the log ends with a summary of failures.

### 4. Check the result in the labeler

```bash
goudacell labels from-run configs/<name>.yaml labels/<name>/
```

The labeler runs in a browser on your own computer, so the files go there and back. The command
prints these lines with the cluster host and folder filled in (add `user@` before the host if
your cluster login differs from your computer's):

```bash
scp cluster:/path/to/labels/<name>/<name>_labeler.html .     # on your computer
# open it, check or fix outlines, press Export labels -> ~/Downloads/<name>_labels_<date>.zip
scp ~/Downloads/<name>_labels_*.zip cluster:/path/to/labels/<name>/   # back to the cluster
goudacell labels read labels/<name>/<name>_labels_*.zip labels/<name>/corrected/   # on the cluster
```

You can also send the HTML to whoever knows the images and have them email the zip back. If the
outlines look wrong, go back to step 2. See [Labeler](#labeler).

## Outputs

| Mode | Masks | Features |
|---|---|---|
| `nuclei` | `*_nuclei_mask.tif` | `*_features.csv` |
| `cells` | `*_mask.tif` | `*_features.csv` |
| `dual` | `*_nuclei_mask.tif`, `*_cell_mask.tif` | `*_features.csv` (nucleus, cell, cytoplasm) |
| + secondary objects | `*_second_obj_mask.tif` | `*_second_obj_features.csv`, plus per-cell counts in `*_features.csv` |

- Each mask has a `.meta.json` saying how it was made.
- For puncta in a screen, use the `cell_median_mad` threshold. Its cutoff is `threshold_k`
  robust noise SDs above the in-cell median, so it follows brightness and doesn't invent objects
  when a perturbation removes them. Per-field Otsu does invent them. Generating a config also
  prints the brieflow keys (named `second_objs`, count mode). These need brieflow 1.5.2+.
- Features cover intensity, shape, texture (Haralick, PFTAS), radial distribution, channel
  correlation, neighbours and foci, plus custom per-cell measurements
  (`ui.set_custom_features([...])`).

## Labeler

The labeler is one HTML file. It shows tiles of your images with a model's outlines on top, and
opens in any browser with no install, account or network.

### Making one

```bash
goudacell labels from-run configs/<name>.yaml labels/<name>/ [options]
```

| Option | Default | |
|---|---|---|
| `--objects` | `cells` | `cells` (nuclei in nuclei mode; nuclei shown as reference in dual mode), or `second_objs` (cells shown as reference) |
| `--n-tiles` | 3 | Tiles per image, at full resolution, in windows with a typical object count |
| `--tile-px` | 512 | Tile size in pixels |
| `--max-fields` | all | Use at most this many images |
| `--um-per-px` | none | Pixel size, for a µm scale bar |

It writes `<name>_labeler.html` (the file to open or send) and `<name>.json` (the pack; keep it
for a next round).

### Using it

Magenta lines are the outlines to check. Yellow lines are reference outlines. Press **?** for
help.

| Problem | Fix | Key |
|---|---|---|
| Not an object | Select it, then Delete | `v`, Del |
| One object in pieces | Merge, then click each piece | `m` |
| Two objects in one outline | Split: drag a line through it | `s` |
| Missed object | Draw around it (double-click to close), or Brush | `w`, `b` |
| Outline too big / gap inside | Erase / Fill | `e`, `f` |

- **Accept tile** (`c`) when every object has one correct outline, then go to the next tile (`n`).
- Other keys: `z` undo, `p` previous tile, `[` `]` brush size, hold `h` to hide outlines.
- **Start this tile from** picks which model to correct. Display controls change only the view.
- Work autosaves in that browser. **Export labels (.zip)** saves every edited or accepted tile to
  Downloads; copy it back to the cluster as in [step 4](#4-check-the-result-in-the-labeler).

### Reading labels back

```bash
goudacell labels read labels.zip corrected/                      # <tile_id>.tif masks + labels.json
goudacell labels next run-cells.json labels.zip run-cells_r2.json   # next round starts from the labels
goudacell labels html run-cells_r2.json run-cells_r2_labeler.html
```

`labels.json` gives each tile's position in its image, whether it was accepted, the model it
started from, and how the masks were made. `labels read` also flags objects in several pieces or
under `--min-area` px.

### Several models, or masks from elsewhere

To compare models on the same tiles, write a spec and run `goudacell labels pack spec.json
pack.json`, then `goudacell labels html pack.json labeler.html`:

```json
{
  "pack_id": "my-cells", "name": "My cells", "object": "cell",
  "channels": [{"key": "cyto", "name": "Membrane", "color": "#4fd17a"},
               {"key": "dapi", "name": "DAPI", "color": "#5a8cff"}],
  "prefill_models": [{"key": "cpsam", "name": "Cellpose-SAM"}, {"key": "cyto3", "name": "cyto3"}],
  "fields": [{"id": "A1", "image": "images/A1.tif", "channels": {"cyto": 1, "dapi": 0},
              "um_per_px": 0.65, "tile_px": 512, "n_tiles": 3, "split": "train",
              "prefills": {"cpsam": "masks/A1_cpsam.tif", "cyto3": ["masks/A1.npz", "cyto3"]},
              "reference": "masks/A1_nuclei.tif"}]
}
```

Packs need a `.meta.json` next to each mask. `goudacell segment` writes them; for your own masks,
call `goudacell.labels.write_meta(path, packages=[...], **params)`. All spec keys are documented
in `goudacell.labels.build_pack`.

## CellProfiler backend

`feature_extraction.method: cellprofiler` measures masks with real CellProfiler 4.2 (headless)
instead of the built-in extractor. CellProfiler needs Python 3.9 and numpy<2, so it lives in its
own env, and goudacell calls it as a subprocess:

```bash
bash scripts/setup_cellprofiler_env.sh             # creates goudacell_cp (run on a compute node)
conda run -n goudacell_cp cellprofiler --version   # 4.2.8.1; warnings are harmless
```

Then select `cellprofiler` in the notebook or config. goudacell looks for CellProfiler in this
order:

1. `cellprofiler_cmd` in the config;
2. `$GOUDACELL_CELLPROFILER`;
3. `cellprofiler` on PATH;
4. the `goudacell_cp` env.

**Pipeline:** the default (`src/goudacell/data/goudacell_default.cppipe`) measures intensity,
size/shape, texture, colocalization and neighbours. To use your own pipeline, set
`pipeline_file`; `goudacell.features_cellprofiler.default_pipeline([...])` gives a starting point.
Your pipeline must:

- load the channel files `<channel>.tif` as grayscale images;
- load `nuclei_mask.tif`, `cell_mask.tif` and `cytoplasm_mask.tif` as objects named `Nuclei`,
  `Cells` and `Cytoplasm`;
- end with ExportToSpreadsheet.

Each image gets `cellprofiler_timeout` seconds (default 3600).

**If it fails:**

- **Slow solve:** use `--solver=libmamba`.
- **Java/JVM errors:** `conda install -n goudacell_cp -c conda-forge openjdk`.
- **Wrong version:** fix `cellprofiler_cmd` or `GOUDACELL_CELLPROFILER`.

## Reference

- **Cellpose version:** the model name picks it. `cyto3`, `cyto2`, `cyto` and `nuclei` run on
  Cellpose 3.1.0; `cpsam` runs on Cellpose 4. A custom model runs on the version its weights
  were trained with. In dual mode, nuclei use `cpsam` next to `cpsam` cells, else `nuclei`.
- **Formats:** TIFF, Nikon ND2, DeltaVision (`.dv`), OME-Zarr.
- **Slurm scripts:** they request one GPU and no partition. Set `SBATCH_PARTITION` or pass
  `--partition`. They activate the `goudacell` env; `GOUDACELL_ENV` picks another.
- **Upgrading from 0.3:** masks and features changed in 0.4; see
  [CHANGELOG.md](CHANGELOG.md#upgrading-from-03).

## License

MIT; see [LICENSE](LICENSE). `src/goudacell/brieflow/` is vendored unchanged from brieflow
(MIT, [its LICENSE](src/goudacell/brieflow/LICENSE)) by `scripts/sync_brieflow.py`.
