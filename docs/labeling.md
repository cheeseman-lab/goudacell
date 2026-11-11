# Labeler: check and correct segmentations

The labeler is one HTML file. It shows tiles of your images with a model's outlines on top. It
opens in any browser, with no install, account or network. Use it to:

- **check** a segmentation, for example whether your cell and secondary-object settings work
  before an OPS screen;
- **correct** wrong outlines (optional), and export the masks for scoring or fine-tuning.

## Quick start

```bash
goudacell segment configs/run.yaml
goudacell labels from-run configs/run.yaml labels/run/          # writes labels/run/run-cells_labeler.html
goudacell labels read run-cells_labels_<date>.zip labels/run/corrected/   # after corrections come back
```

The notebook prints the `from-run` command when it writes a config.

## Making a labeler from a run

| Option | Default | |
|---|---|---|
| `--objects` | `cells` | `cells`: cells, or nuclei in nuclei mode; nuclei show as reference in dual mode. `second_objs`: secondary objects, with cells as reference. |
| `--n-tiles` | 3 | Tiles per image, cut at full resolution, in windows with a typical object count. |
| `--tile-px` | 512 | Tile size in pixels. |
| `--max-fields` | all | Use at most this many images, spread through the run. |
| `--um-per-px` | none | Pixel size, for a µm scale bar. |

Output: `<name>_labeler.html` (the only file to send), `<name>.json` (the pack; keep it for the
next round) and `<name>_spec.json` (the recipe).

## In the labeler

Magenta lines are the outlines to check. Yellow lines are reference outlines, there only to help
you orient. Press **?** for help.

| Problem | Fix | Key |
|---|---|---|
| Not an object | Select it, then Delete | `v`, Del |
| One object in pieces | Merge, then click each piece | `m` |
| Two objects in one outline | Split: drag a line through it | `s` |
| Missed object | Draw around it (double-click to close), or Brush | `w`, `b` |
| Outline too big / gap inside | Erase / Fill | `e`, `f` |

- **Accept tile** (`c`) when every object has one correct outline, then go to the next tile (`n`).
- Other keys: `z` undo, `p` previous tile, `[` `]` brush size, hold `h` to hide outlines.
- Brush and Fill don't paint over neighbours unless you hold Shift.
- **Start this tile from** picks which model's outlines to correct. **Show other model** overlays
  a second model in cyan.
- Display controls change only the view, never the labels.
- Work autosaves in that browser. **Export labels (.zip)** saves every edited or accepted tile;
  send that file.

## Reading labels back

`goudacell labels read labels.zip OUT_DIR` writes:

- `OUT_DIR/<tile_id>.tif` (uint16 masks);
- `labels.json`: per tile, its position in the image, whether it was accepted, and the model it
  started from; plus the pack's provenance.

It also flags objects in several pieces, or under `--min-area` px. In Python:
`manifest, masks, problems = goudacell.labels.read_export("labels.zip")`.

To run another round, starting from the returned labels:

```bash
goudacell labels next run-cells.json labels.zip run-cells_r2.json
goudacell labels html run-cells_r2.json run-cells_r2_labeler.html
```

## Several models, or masks from elsewhere

Write a spec, then `goudacell labels pack spec.json pack.json` and
`goudacell labels html pack.json labeler.html`:

```json
{
  "pack_id": "my-cells", "name": "My cells: two models", "object": "cell",
  "channels": [{"key": "cyto", "name": "Membrane", "color": "#4fd17a"},
               {"key": "dapi", "name": "DAPI", "color": "#5a8cff", "gain": 0.5}],
  "prefill_models": [{"key": "cpsam", "name": "Cellpose-SAM"}, {"key": "cyto3", "name": "cyto3"}],
  "fields": [{
    "id": "A1", "image": "images/A1.tif", "channels": {"cyto": 1, "dapi": 0},
    "um_per_px": 0.65, "tile_px": 512, "n_tiles": 3, "split": "train",
    "prefills": {"cpsam": "masks/A1_cpsam.tif", "cyto3": ["masks/A1.npz", "cyto3"]},
    "reference": "masks/A1_nuclei.tif"
  }]
}
```

- `image` can be `.npy` (C×Y×X) or any format goudacell reads.
- A mask is a path or `[file.npz, key]`.
- The first model is the default.
- `split` is carried into the export, so test tiles can stay out of fine-tuning.
- Every key is in `goudacell.labels.build_pack`.

## Provenance

Every pack records how its outlines were made:

- `goudacell segment` writes `<mask>.meta.json` next to each mask (method, parameters, config,
  package versions, host, Slurm job).
- Packs refuse masks without a record. They embed the record and SHA-256 checksums of the image
  and masks, and the export carries them back.
- For your own masks, call `goudacell.labels.write_meta(path, packages=[...], **params)` when
  saving (or pass `--allow-missing-meta`).

## Troubleshooting

- **No masks found:** the masks aren't where the config says. Relative paths are tried from the
  config's folder, then from the current one.
- **No .meta.json record:** the masks predate this version. Re-run them, or use `write_meta`.
- **Work gone:** it lives in the browser it was made in; a private window or cleared site data
  starts empty. Export often.
