# Changelog

## 0.4.0

goudacell now runs brieflow's phenotype code, vendored unchanged from brieflow `zarr3` @
`f03a2c8` by `scripts/sync_brieflow.py` into `src/goudacell/brieflow/`. Masks and features
match brieflow's for the same parameters.

### Upgrading from 0.3

Outputs change for existing configs:

- **Dual-mode masks change slightly.** The cytoplasm channel is prepared with brieflow's
  `image_log_scale`, not `log1p`: typically ±1-2 objects per 150 and a mean IoU of ~0.91
  against 0.3.0 cell masks. `dual.nuclei_model` is ignored; nuclei are always segmented with
  `nuclei` (Cellpose 3) or `cpsam` (Cellpose 4).
- **Nuclei-only mode** segments the DAPI plane after brieflow's preprocessing: it is
  normalized to its 99.5th percentile and converted to uint8 before Cellpose (0.3.0 passed the
  raw image). It uses the config's `model` (0.3.0 forced `nuclei`); a config without `model:`
  now uses `cyto3`, so set `model: nuclei` explicitly.
- **Cytoplasm = cell minus that cell's own nucleus** (0.3.0 subtracted every nucleus). This
  changes cytoplasm features for cells overlapped by a neighbour's nucleus. `reconcile: null`
  now produces no cytoplasm features.
- **Feature columns**: `*_int`/`*_int_edge` are renamed `*_integrated`/`*_integrated_edge`.
  New columns: `*_bounds_0..3`, cytoplasm neighbor measurements and `num_nuclei`. The column
  order changed. With secondary objects on, you also get `*_second_obj_mask.tif`,
  `*_second_obj_features.csv` and per-cell object summary columns.
- **cp_measure** is now brieflow's extractor with cp-measure 0.1.18 (was 0.1.12). With 0.1.18
  `*_IntensityEdge` is ~0 for most objects (brieflow gives the same values), cytoplasm
  columns shift with the cytoplasm definition above, and the table still has no `label`
  column.
- **Notebook defaults** now follow brieflow's: reconcile `contained_in_cells`, cell flow
  threshold 1.0, edge removal on.
- **API**: removed `goudacell.cp_emulator`, `goudacell.feature_table_utils`,
  `goudacell.features_cp_measure`, `features.FEATURES_BASIC` and
  `extract_features(foci_params=...)`. `segment.reconcile_nuclei_cells` is still importable.
- **CellProfiler backend**: fixed. 0.3.0 silently returned empty tables; it now joins the
  Nuclei/Cells/Cytoplasm tables on the mask `label` and raises on a failed run. It finds
  CellProfiler by itself (`GOUDACELL_CELLPROFILER`, PATH, or the `goudacell_cp` env made by
  `scripts/setup_cellprofiler_env.sh` from `envs/cellprofiler.yml`) and runs a default pipeline
  when `pipeline_file` is unset. `cellprofiler_cmd` defaults to unset instead of
  `"cellprofiler"`; an explicit command still wins. The command is checked to be CellProfiler
  4.2.x before any segmentation.
- **Wrong-env check**: `import goudacell` raises in the `goudacell_cp` env, on Python < 3.10 or
  with numpy < 2, with a message saying to switch to the `goudacell` env.
- **`goudacell segment` exit status**: it now ends with a summary of the failed files and exits
  1 when every file failed (0.3.0 always exited 0). A partial failure still exits 0.
- **numpy** is pinned below 2.4: numpy 2.4 removed `np.in1d`, which the vendored foci and
  secondary-object code calls.
