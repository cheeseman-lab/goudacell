"""Interactive ipywidgets UI for tuning segmentation parameters in a notebook.

The notebook stays thin: it constructs a :class:`ParameterUI` and renders four
focused blocks — **data** (look at the image and its channels), **segment**
(tune with sliders, estimate diameters, sweep and compare overlays), **features**
(extract per-cell measurements), and **config** (name the run and generate its
config). All heavy logic lives in the rest of the package, so this module is glue.

Usage in a notebook::

    from goudacell.notebook import ParameterUI
    ui = ParameterUI(input_dir="../data", file_pattern="*.tif")
    ui.data        # block 1 — inspect the image
    ui.segment     # block 2 — tune / sweep
    ui.features    # block 3 — extract features
    ui.config      # block 4 — name the run + generate config
"""

from pathlib import Path
from typing import List, Optional

import ipywidgets as widgets

from goudacell.config import (
    DualSegmentationParams,
    FeatureExtractionParams,
    SegmentationConfig,
)

COMPARTMENTS = ["nucleus", "cell", "cytoplasm"]
METHODS = ["cp_emulator", "cp_measure", "cellprofiler"]

# Cap on grid-sweep combinations rendered as overlays (keeps thumbnails legible).
MAX_SWEEP_CELLS = 25

# Backends fast enough to run live in the notebook preview; the rest are config-only.
PREVIEW_METHODS = {"cp_emulator"}


def available_feature_methods() -> List[str]:
    """Feature backends actually runnable in the current environment.

    `cp_emulator` is built in; `cp_measure` needs its package installed;
    `cellprofiler` needs a CellProfiler CLI on PATH. Unavailable backends are
    hidden so selecting one can't produce a config that fails everywhere.
    """
    import importlib.util
    import shutil

    methods = ["cp_emulator"]
    if importlib.util.find_spec("cp_measure") is not None:
        methods.append("cp_measure")
    if shutil.which("cellprofiler") is not None:
        methods.append("cellprofiler")
    return methods

FLOW_HELP = (
    "<i><b>flow_threshold</b> (0–1+): max allowed flow error per mask. "
    "Higher keeps more (lower-quality) masks → more objects; lower is stricter "
    "→ fewer, cleaner objects. Default 0.4.</i>"
)
CELLPROB_HELP = (
    "<i><b>cellprob_threshold</b> (−6…6): include pixels above this cell "
    "probability. Higher → fewer/smaller objects; lower → more/larger objects. "
    "Default 0.0.</i>"
)


def _parse_int_list(text: str) -> Optional[List[int]]:
    """Parse "0,2" into [0, 2]; blank/whitespace returns None (meaning all)."""
    text = (text or "").strip()
    if not text:
        return None
    return [int(p) for p in text.replace(" ", "").split(",") if p != ""]


def _parse_str_list(text: str) -> Optional[List[str]]:
    """Parse "DAPI,GFP" into ["DAPI", "GFP"]; blank returns None (auto names)."""
    text = (text or "").strip()
    if not text:
        return None
    return [p for p in (s.strip() for s in text.split(",")) if p]


def _parse_float_list(text: str) -> List[float]:
    """Parse "0.2,0.4,0.6" into [0.2, 0.4, 0.6]; blank returns []."""
    text = (text or "").strip()
    if not text:
        return []
    return [float(p) for p in text.replace(" ", "").split(",") if p != ""]


def _default_models() -> tuple:
    """Pick (nuclei_model, cell_model) for the installed Cellpose version."""
    try:
        from goudacell.segment import get_cellpose_version

        if get_cellpose_version()[0] >= 4:
            return "cpsam", "cpsam"
    except Exception:
        pass
    return "nuclei", "cyto3"


def _count(masks) -> int:
    """Count labelled objects in a mask, excluding background."""
    import numpy as np

    return int(len(set(np.unique(masks)) - {0}))


class ParameterUI:
    """ipywidgets panels for interactive segmentation tuning and config export.

    Renders four blocks (``.data``, ``.segment``, ``.features``, ``.config``)
    that share one loaded image and the last previewed masks. Only the controls
    relevant to the chosen mode are shown, so nuclei-only never asks for cell
    parameters.

    Attributes:
        input_dir: Directory scanned for input images.
        file_pattern: Glob used to list images.
        image: The currently loaded image (set by the Load button), or None.
    """

    def __init__(
        self,
        input_dir: str = ".",
        file_pattern: str = "*.tif",
        output_dir: str = "../out",
        config_dir: str = "../configs",
        nuclei_model: Optional[str] = None,
        cell_model: Optional[str] = None,
    ):
        """Build the widget panels.

        Args:
            input_dir: Directory containing input images.
            file_pattern: Glob pattern for listing images.
            output_dir: Where batch masks/features are written (in the config).
            config_dir: Where the generated config YAML is saved.
            nuclei_model: Override the nuclei model (default: auto by version).
            cell_model: Override the cell model (default: auto by version).
        """
        self.input_dir = input_dir
        self.file_pattern = file_pattern
        self.output_dir = output_dir
        self.config_dir = config_dir
        auto_nuc, auto_cell = (None, None)
        if nuclei_model is None or cell_model is None:
            auto_nuc, auto_cell = _default_models()
        self.nuclei_model = nuclei_model or auto_nuc
        self.cell_model = cell_model or auto_cell

        self.image = None
        self._last_masks = (None, None)

        self._build_widgets()
        self._wire_events()
        self._sync_mode_visibility()

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        """Construct all widgets and assemble the three block panels."""
        slider = dict(continuous_update=False, style={"description_width": "120px"})
        wide = widgets.Layout(width="340px")
        label_style = {"description_width": "120px"}

        self.gpu_banner = widgets.HTML(value=self._gpu_banner_html())

        # ---- Block 1: data -------------------------------------------------
        self.input_dir_w = widgets.Text(
            value=str(self.input_dir), description="Input dir:", layout=wide
        )
        self.file_pattern_w = widgets.Text(
            value=self.file_pattern, description="Pattern:", layout=wide
        )
        self.image_w = widgets.Dropdown(description="Image:", layout=wide)
        self.refresh_btn = widgets.Button(description="Refresh files", icon="rotate-right")
        self.load_btn = widgets.Button(
            description="Load image", button_style="info", icon="image"
        )
        self.data_output = widgets.Output()
        self._refresh_images()

        self.data = widgets.VBox(
            [
                widgets.HTML("<h3>1 · Data</h3>"),
                self.gpu_banner,
                widgets.HBox([self.input_dir_w, self.refresh_btn]),
                widgets.HBox([self.file_pattern_w, self.image_w]),
                self.load_btn,
                self.data_output,
            ]
        )

        # ---- Block 2: segment ---------------------------------------------
        self.mode_w = widgets.ToggleButtons(
            options=["nuclei", "cells", "dual"], value="dual", description="Mode:"
        )

        self.nuc_channel = widgets.BoundedIntText(
            value=0, min=0, max=20, description="Nuclei channel:", style=label_style
        )
        self.nuc_diameter = widgets.FloatSlider(
            value=15.0, min=3, max=300, step=1, description="Nuclei diameter:", **slider
        )
        self.nuc_flow = widgets.FloatSlider(
            value=0.4, min=0.0, max=1.0, step=0.05, description="Nuclei flow:", **slider
        )
        self.nuc_cellprob = widgets.FloatSlider(
            value=0.0, min=-6, max=6, step=0.5, description="Nuclei cellprob:", **slider
        )
        self.nuclei_box = widgets.VBox(
            [
                widgets.HTML("<b>Nuclei</b>"),
                self.nuc_channel,
                self.nuc_diameter,
                self.nuc_flow,
                self.nuc_cellprob,
            ]
        )

        self.cell_channel = widgets.BoundedIntText(
            value=1, min=0, max=20, description="Cell channel:", style=label_style
        )
        self.cell_diameter = widgets.FloatSlider(
            value=40.0, min=5, max=600, step=1, description="Cell diameter:", **slider
        )
        self.cell_flow = widgets.FloatSlider(
            value=0.4, min=0.0, max=1.0, step=0.05, description="Cell flow:", **slider
        )
        self.cell_cellprob = widgets.FloatSlider(
            value=0.0, min=-6, max=6, step=0.5, description="Cell cellprob:", **slider
        )
        self.cell_box = widgets.VBox(
            [
                widgets.HTML("<b>Cells</b>"),
                self.cell_channel,
                self.cell_diameter,
                self.cell_flow,
                self.cell_cellprob,
            ]
        )

        self.remove_edge = widgets.Checkbox(value=False, description="Remove edge cells")
        self.z_project = widgets.Checkbox(value=True, description="Z-project stacks")
        self.gpu = widgets.Checkbox(value=True, description="Use GPU")
        self.reconcile = widgets.Checkbox(
            value=True,
            description="Reconcile nuclei ↔ cells (drop nucleus-less cells)",
            indent=False,
            layout=widgets.Layout(width="380px"),
        )

        self.estimate_btn = widgets.Button(
            description="Estimate diameters", icon="ruler", tooltip="Cellpose 3.x only"
        )
        self.run_btn = widgets.Button(
            description="Run preview", button_style="primary", icon="play"
        )
        self.help_box = widgets.HTML(value=f"{FLOW_HELP}<br>{CELLPROB_HELP}")
        self.seg_output = widgets.Output()

        # Sweep sub-panel
        self.sweep_target = widgets.Dropdown(
            options=["nuclei", "cells"], value="cells", description="Sweep target:",
            style=label_style,
        )
        self.sweep_x_param = widgets.Dropdown(
            options=["diameter", "flow", "cellprob"], value="flow", description="X param:",
            style=label_style,
        )
        self.sweep_x_values = widgets.Text(
            value="0.2,0.4,0.6,0.8", description="X values:", style=label_style
        )
        self.sweep_y_param = widgets.Dropdown(
            options=["none", "diameter", "flow", "cellprob"], value="cellprob",
            description="Y param:", style=label_style,
        )
        self.sweep_y_values = widgets.Text(
            value="-2,-1,0,1,2", description="Y values:", style=label_style
        )
        self.sweep_btn = widgets.Button(
            description="Run sweep", icon="th", tooltip="Segment across a parameter grid"
        )
        self.sweep_pick = widgets.Dropdown(
            options=[], description="Pick result:", style=label_style,
            layout=widgets.Layout(width="340px"),
        )
        self.sweep_apply_btn = widgets.Button(
            description="Apply to sliders", icon="check",
            tooltip="Set the segment controls to the picked combination",
        )
        self.sweep_output = widgets.Output()
        self.sweep_box = widgets.VBox(
            [
                widgets.HTML(
                    "<b>Parameter sweep</b> — segment across a grid and compare the "
                    "overlays. Set <i>Y param</i> to <i>none</i> for a 1D sweep, then pick "
                    "the best result below and apply it."
                ),
                self.sweep_target,
                widgets.HBox([self.sweep_x_param, self.sweep_x_values]),
                widgets.HBox([self.sweep_y_param, self.sweep_y_values]),
                self.sweep_btn,
                self.sweep_output,
                widgets.HBox([self.sweep_pick, self.sweep_apply_btn]),
            ]
        )

        self.segment = widgets.VBox(
            [
                widgets.HTML("<h3>2 · Segment</h3>"),
                self.mode_w,
                widgets.HBox([self.nuclei_box, self.cell_box]),
                self.help_box,
                widgets.HBox([self.remove_edge, self.z_project, self.gpu]),
                self.reconcile,
                widgets.HBox([self.estimate_btn, self.run_btn]),
                self.seg_output,
                widgets.HTML("<hr>"),
                self.sweep_box,
            ]
        )

        # ---- Block 3: features --------------------------------------------
        self.feat_enabled = widgets.Checkbox(value=False, description="Extract features")
        self.feat_method = widgets.Dropdown(
            options=available_feature_methods(), value="cp_emulator", description="Method:"
        )
        self.feat_method_note = widgets.HTML(
            "<i>Only backends installed in this environment are listed. To enable more: "
            "<code>cp_measure</code> → <code>uv pip install -e '.[cp_measure]'</code>; "
            "<code>cellprofiler</code> → a CellProfiler env on PATH + a pipeline.</i>"
        )
        self.feat_channels = widgets.Text(
            value="", placeholder="all (e.g. 0,2)", description="Channels:"
        )
        self.feat_channel_names = widgets.Text(
            value="", placeholder="auto (e.g. DAPI,GFP)", description="Names:"
        )
        self.feat_compartments = widgets.SelectMultiple(
            options=COMPARTMENTS, value=tuple(COMPARTMENTS), description="Compartments:"
        )
        self.feat_texture = widgets.Checkbox(value=True, description="Texture")
        self.feat_correlation = widgets.Checkbox(value=True, description="Correlation")
        self.feat_neighbors = widgets.Checkbox(value=True, description="Neighbors")
        self.feat_combine = widgets.Checkbox(
            value=False, description="Combine into one table (with filename)"
        )
        self.preview_feat_btn = widgets.Button(description="Preview features", icon="table")
        self.feat_output = widgets.Output()

        self.features = widgets.VBox(
            [
                widgets.HTML("<h3>3 · Features</h3>"),
                self.feat_enabled,
                self.feat_method,
                self.feat_method_note,
                widgets.HBox([self.feat_channels, self.feat_channel_names]),
                self.feat_compartments,
                widgets.HBox([self.feat_texture, self.feat_correlation, self.feat_neighbors]),
                self.feat_combine,
                self.preview_feat_btn,
                self.feat_output,
            ]
        )

        # ---- Block 4: config ----------------------------------------------
        self.config_name_w = widgets.Text(
            value="segmentation_config", description="Name:", layout=wide
        )
        self.config_btn = widgets.Button(
            description="Generate config", button_style="success", icon="floppy-disk"
        )
        self.config_output = widgets.Output()
        self.config = widgets.VBox(
            [
                widgets.HTML(
                    "<h3>4 · Config</h3>"
                    "<i>Name this run. The config is written to "
                    "<code>configs/&lt;name&gt;.yaml</code>; the batch run dumps masks to "
                    "<code>masks/&lt;name&gt;/</code> and features to "
                    "<code>features/&lt;name&gt;/</code>.</i>"
                ),
                self.config_name_w,
                self.config_btn,
                self.config_output,
            ]
        )

        # Full stack (so a bare `ui` still renders everything).
        self.panel = widgets.VBox(
            [
                self.data,
                widgets.HTML("<hr>"),
                self.segment,
                widgets.HTML("<hr>"),
                self.features,
                widgets.HTML("<hr>"),
                self.config,
            ]
        )

    def _wire_events(self) -> None:
        """Connect widget callbacks."""
        self.mode_w.observe(lambda _c: self._sync_mode_visibility(), names="value")
        self.refresh_btn.on_click(lambda _b: self._refresh_images())
        self.load_btn.on_click(self._on_load)
        self.estimate_btn.on_click(self._on_estimate)
        self.run_btn.on_click(self._on_run)
        self.sweep_btn.on_click(self._on_sweep)
        self.sweep_apply_btn.on_click(self._on_apply_sweep)
        self.preview_feat_btn.on_click(self._on_preview_features)
        self.config_btn.on_click(self._on_generate)

    def _sync_mode_visibility(self) -> None:
        """Show only the controls relevant to the selected mode."""
        mode = self.mode_w.value
        self.nuclei_box.layout.display = "" if mode in ("nuclei", "dual") else "none"
        self.cell_box.layout.display = "" if mode in ("cells", "dual") else "none"
        # Reconcile and sweep target only matter in dual mode.
        self.reconcile.layout.display = "" if mode == "dual" else "none"
        self.sweep_target.layout.display = "" if mode == "dual" else "none"

    def _refresh_images(self) -> None:
        """Repopulate the image dropdown from input_dir/file_pattern."""
        directory = Path(self.input_dir_w.value)
        files = sorted(directory.glob(self.file_pattern_w.value))
        options = [(f.name, str(f)) for f in files]
        self.image_w.options = options
        if options:
            self.image_w.value = options[0][1]

    def _gpu_banner_html(self) -> str:
        """Render the GPU status banner HTML."""
        from goudacell.gpu import detect_gpu

        status = detect_gpu()
        color = "#1a7f37" if status.available else "#b35900"
        mark = "✓" if status.available else "⚠"
        return (
            f"<div style='padding:6px 10px;border-left:4px solid {color};"
            f"background:#f6f8fa'>{mark} <b>GPU:</b> {status.reason}</div>"
        )

    # ------------------------------------------------------------------
    # Logic (testable without a frontend)
    # ------------------------------------------------------------------
    def selected_image_path(self) -> Optional[Path]:
        """Return the currently selected image path, or None if no files."""
        return Path(self.image_w.value) if self.image_w.value else None

    def _effective_target(self) -> str:
        """The compartment a sweep/preview acts on for the current mode."""
        mode = self.mode_w.value
        return self.sweep_target.value if mode == "dual" else mode

    def build_feature_params(self) -> FeatureExtractionParams:
        """Build FeatureExtractionParams from the feature-extraction widgets."""
        compartments = list(self.feat_compartments.value)
        return FeatureExtractionParams(
            enabled=self.feat_enabled.value,
            method=self.feat_method.value,
            channels=_parse_int_list(self.feat_channels.value),
            channel_names=_parse_str_list(self.feat_channel_names.value),
            compartments=compartments if len(compartments) < len(COMPARTMENTS) else None,
            include_texture=self.feat_texture.value,
            include_correlation=self.feat_correlation.value,
            include_neighbors=self.feat_neighbors.value,
            output_path="{stem}_features.csv",
            combine_tables=self.feat_combine.value,
        )

    def _dual_params(self) -> DualSegmentationParams:
        """Build DualSegmentationParams from the current widget values."""
        return DualSegmentationParams(
            nuclei_channel=self.nuc_channel.value,
            cyto_channel=self.cell_channel.value,
            nuclei_diameter=self.nuc_diameter.value,
            cell_diameter=self.cell_diameter.value,
            cell_model=self.cell_model,
            nuclei_model=self.nuclei_model,
            nuclei_flow_threshold=self.nuc_flow.value,
            nuclei_cellprob_threshold=self.nuc_cellprob.value,
            cell_flow_threshold=self.cell_flow.value,
            cell_cellprob_threshold=self.cell_cellprob.value,
        )

    def build_config(self) -> SegmentationConfig:
        """Assemble a SegmentationConfig from the current widget values.

        Paths are written relative to the config file's directory so the SLURM
        script can ``cd`` to it. The config always requests ``gpu=True`` because
        batch jobs run on a GPU node regardless of where this notebook runs.

        Returns:
            A SegmentationConfig ready to serialize with ``to_yaml``.
        """
        mode = self.mode_w.value
        common = dict(
            input_dir=str(Path(self.input_dir_w.value).resolve()),
            output_dir=str(Path(self.output_dir).resolve()),
            file_pattern=self.file_pattern_w.value,
            channels=[0, 0],
            gpu=True,
            remove_edge_cells=self.remove_edge.value,
            z_project=self.z_project.value,
            feature_extraction=self.build_feature_params(),
        )

        if mode == "dual":
            return SegmentationConfig(
                model=self.cell_model,
                diameter=self.cell_diameter.value,
                flow_threshold=self.cell_flow.value,
                cellprob_threshold=self.cell_cellprob.value,
                channel_to_segment=None,
                mode="dual",
                dual=self._dual_params(),
                reconcile="consensus" if self.reconcile.value else None,
                **common,
            )

        if mode == "nuclei":
            return SegmentationConfig(
                model=self.nuclei_model,
                diameter=self.nuc_diameter.value,
                flow_threshold=self.nuc_flow.value,
                cellprob_threshold=self.nuc_cellprob.value,
                channel_to_segment=self.nuc_channel.value,
                mode="nuclei",
                **common,
            )

        return SegmentationConfig(
            model=self.cell_model,
            diameter=self.cell_diameter.value,
            flow_threshold=self.cell_flow.value,
            cellprob_threshold=self.cell_cellprob.value,
            channel_to_segment=self.cell_channel.value,
            mode="cells",
            **common,
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _require_image(self):
        """Return the loaded image, loading the selected one if needed."""
        if self.image is not None:
            return self.image
        from goudacell.io import load_image

        path = self.selected_image_path()
        if path is None:
            return None
        self.image = load_image(path, channel=None, z_project=self.z_project.value)
        return self.image

    def _on_load(self, _btn) -> None:
        """Load the selected image and show a per-channel montage + metadata."""
        import matplotlib.pyplot as plt

        from goudacell.io import load_image

        self.data_output.clear_output(wait=True)
        with self.data_output:
            path = self.selected_image_path()
            if path is None:
                print("No image — check Input dir / Pattern, then click Refresh files.")
                return
            self.image = load_image(path, channel=None, z_project=self.z_project.value)
            image = self.image
            n_channels = image.shape[0] if image.ndim == 3 else 1
            print(f"{path.name}\n  shape {image.shape}, dtype {image.dtype}, channels {n_channels}")

            # Update channel selectors to the real channel count.
            for sel in (self.nuc_channel, self.cell_channel):
                sel.max = max(0, n_channels - 1)

            fig, axes = plt.subplots(1, n_channels, figsize=(4 * n_channels, 4))
            axes = [axes] if n_channels == 1 else list(axes)
            for i, ax in enumerate(axes):
                ch = image[i] if image.ndim == 3 else image
                ax.imshow(ch, cmap="gray")
                ax.set_title(f"Channel {i}")
                ax.axis("off")
            plt.tight_layout()
            plt.show()
            print("Note the channel numbers above, then set them in block 2.")

    def _on_estimate(self, _btn) -> None:
        """Estimate diameters (Cellpose 3.x) and fill the diameter sliders."""
        from goudacell.gpu import resolve_gpu
        from goudacell.segment import estimate_diameter, get_cellpose_version

        self.seg_output.clear_output(wait=True)
        with self.seg_output:
            if get_cellpose_version()[0] >= 4:
                print("Diameter estimation needs Cellpose 3.x; set diameters manually.")
                return
            image = self._require_image()
            if image is None:
                print("Load an image first (block 1).")
                return
            use_gpu = resolve_gpu(self.gpu.value)
            mode = self.mode_w.value
            if mode in ("nuclei", "dual"):
                img = image[self.nuc_channel.value] if image.ndim == 3 else image
                self.nuc_diameter.value = estimate_diameter(
                    img, model=self.nuclei_model, gpu=use_gpu
                )
                print(f"Estimated nuclei diameter: {self.nuc_diameter.value:.1f}")
            if mode in ("cells", "dual"):
                img = image[self.cell_channel.value] if image.ndim == 3 else image
                self.cell_diameter.value = estimate_diameter(
                    img, model=self.cell_model, gpu=use_gpu
                )
                print(f"Estimated cell diameter: {self.cell_diameter.value:.1f}")

    def _on_run(self, _btn) -> None:
        """Segment the loaded image and show an overlay."""
        import matplotlib.pyplot as plt

        from goudacell.gpu import resolve_gpu
        from goudacell.segment import segment, segment_nuclei_and_cells
        from goudacell.viz import make_mask_cmap

        self.seg_output.clear_output(wait=True)
        with self.seg_output:
            image = self._require_image()
            if image is None:
                print("Load an image first (block 1).")
                return

            mode = self.mode_w.value
            use_gpu = resolve_gpu(self.gpu.value)

            if mode == "dual":
                nuclei, cells = segment_nuclei_and_cells(
                    image,
                    nuclei_channel=self.nuc_channel.value,
                    cyto_channel=self.cell_channel.value,
                    nuclei_diameter=self.nuc_diameter.value,
                    cell_diameter=self.cell_diameter.value,
                    cell_model=self.cell_model,
                    nuclei_model=self.nuclei_model,
                    nuclei_flow_threshold=self.nuc_flow.value,
                    nuclei_cellprob_threshold=self.nuc_cellprob.value,
                    cell_flow_threshold=self.cell_flow.value,
                    cell_cellprob_threshold=self.cell_cellprob.value,
                    gpu=use_gpu,
                    remove_edge_cells=self.remove_edge.value,
                    reconcile="consensus" if self.reconcile.value else None,
                )
                panels = [
                    (self.nuc_channel.value, nuclei, f"Nuclei ({_count(nuclei)})"),
                    (self.cell_channel.value, cells, f"Cells ({_count(cells)})"),
                ]
                self._last_masks = (nuclei, cells)
            else:
                channel = self.nuc_channel.value if mode == "nuclei" else self.cell_channel.value
                model = self.nuclei_model if mode == "nuclei" else self.cell_model
                diameter = self.nuc_diameter.value if mode == "nuclei" else self.cell_diameter.value
                flow = self.nuc_flow.value if mode == "nuclei" else self.cell_flow.value
                prob = self.nuc_cellprob.value if mode == "nuclei" else self.cell_cellprob.value
                seg_img = image[channel] if image.ndim == 3 else image
                masks = segment(
                    seg_img, diameter=diameter, model=model, flow_threshold=flow,
                    cellprob_threshold=prob, gpu=use_gpu,
                    remove_edge_cells=self.remove_edge.value,
                )
                panels = [(channel, masks, f"{mode} ({_count(masks)})")]
                self._last_masks = (masks, None)

            fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6))
            axes = [axes] if len(panels) == 1 else list(axes)
            for ax, (ch, masks, title) in zip(axes, panels):
                bg = image[ch] if image.ndim == 3 else image
                ax.imshow(bg, cmap="gray")
                ax.imshow(masks, cmap=make_mask_cmap(masks), alpha=0.4)
                ax.set_title(title)
                ax.axis("off")
            plt.tight_layout()
            plt.show()

    def _on_sweep(self, _btn) -> None:
        """Segment across a parameter grid and show the overlays for comparison."""
        import matplotlib.pyplot as plt
        import numpy as np

        from goudacell.gpu import resolve_gpu
        from goudacell.segment import (
            reproducible_plateau_2d,
            reproducible_range_1d,
            sweep_grid,
        )
        from goudacell.viz import plot_sweep_montage

        self.sweep_output.clear_output(wait=True)
        with self.sweep_output:
            image = self._require_image()
            if image is None:
                print("Load an image first (block 1).")
                return

            x_param = self.sweep_x_param.value
            x_values = _parse_float_list(self.sweep_x_values.value)
            if not x_values:
                print("Enter comma-separated X values (e.g. 0.2,0.4,0.6,0.8).")
                return

            y_is_1d = self.sweep_y_param.value == "none"
            y_param = self.sweep_y_param.value
            y_values = [None] if y_is_1d else _parse_float_list(self.sweep_y_values.value)
            if not y_is_1d and not y_values:
                print("Enter comma-separated Y values, or set Y param to 'none'.")
                return

            n_combos = len(x_values) * len(y_values)
            if n_combos > MAX_SWEEP_CELLS:
                print(f"{n_combos} combinations is a lot to render as overlays "
                      f"(max {MAX_SWEEP_CELLS}). Use fewer values per axis.")
                return

            mode = self.mode_w.value
            target = self._effective_target()
            use_gpu = resolve_gpu(self.gpu.value)
            axis_y = None if y_is_1d else (y_param, y_values)

            cells = sweep_grid(
                image, self._dual_params(), (x_param, x_values), axis_y,
                mode=mode, target=target, gpu=use_gpu,
                remove_edge_cells=self.remove_edge.value,
            )

            display_channel = (
                self.nuc_channel.value if target == "nuclei" else self.cell_channel.value
            )
            display_img = image[display_channel] if image.ndim == 3 else image
            title = (
                f"{target} {x_param} sweep" if y_is_1d
                else f"{target} sweep ({x_param} × {y_param})"
            )
            plot_sweep_montage(display_img, cells, x_values, y_values,
                               x_param=x_param, y_param="" if y_is_1d else y_param, title=title)
            plt.show()

            # Remember the sweep so a pick can be applied to the sliders.
            self._sweep_state = {
                "cells": cells, "x_param": x_param, "y_param": None if y_is_1d else y_param,
                "target": target,
            }
            self.sweep_pick.options = [
                (self._combo_label(c, x_param, None if y_is_1d else y_param), i)
                for i, c in enumerate(cells)
            ]

            # Suggest the most reproducible (count-stable) combination as a hint.
            counts = np.array([c.count for c in cells], dtype=int)
            if counts.max() == 0:
                print("⚠ Found 0 objects everywhere — check the channel, diameter, "
                      "or thresholds before picking anything.")
            elif y_is_1d:
                stable = reproducible_range_1d(x_values, counts)
                if stable:
                    print(f"Most reproducible {x_param}: {stable['lo']}–{stable['hi']} "
                          f"(suggest {stable['setpoint']}).")
            else:
                grid = counts.reshape(len(y_values), len(x_values))
                plateau = reproducible_plateau_2d(grid)
                if plateau:
                    sx, sy = x_values[plateau["ix"]], y_values[plateau["iy"]]
                    print(f"Most reproducible combination: {x_param}={sx}, {y_param}={sy}.")

    def _combo_label(self, cell, x_param, y_param) -> str:
        """Human-readable label for a sweep result (for the picker dropdown)."""
        if y_param is None:
            return f"{x_param}={cell.x} → {cell.count} objects"
        return f"{x_param}={cell.x}, {y_param}={cell.y} → {cell.count} objects"

    def _set_target_param(self, target: str, param: str, value: float) -> None:
        """Set the slider for (target, param) to value."""
        prefix = "nuc" if target == "nuclei" else "cell"
        attr = {
            "diameter": f"{prefix}_diameter",
            "flow": f"{prefix}_flow",
            "cellprob": f"{prefix}_cellprob",
        }[param]
        getattr(self, attr).value = value

    def _on_apply_sweep(self, _btn) -> None:
        """Apply the picked sweep result to the segment sliders."""
        state = getattr(self, "_sweep_state", None)
        with self.sweep_output:
            if not state or self.sweep_pick.value is None:
                print("Run a sweep and pick a result first.")
                return
            cell = state["cells"][self.sweep_pick.value]
            target = state["target"]
            self._set_target_param(target, state["x_param"], cell.x)
            applied = f"{state['x_param']}={cell.x}"
            if state["y_param"] is not None:
                self._set_target_param(target, state["y_param"], cell.y)
                applied += f", {state['y_param']}={cell.y}"
            print(f"Applied to {target} sliders: {applied}")

    def _on_preview_features(self, _btn) -> None:
        """Run a feature-extraction preview on the last previewed masks."""
        from goudacell.features import extract_features

        self.feat_output.clear_output(wait=True)
        with self.feat_output:
            nuclei, cells = self._last_masks
            if nuclei is None:
                print("Run a segmentation preview first (block 2).")
                return
            if not self.feat_enabled.value:
                print("Tick 'Extract features' to enable.")
                return
            fe = self.build_feature_params()
            if fe.method not in PREVIEW_METHODS:
                print(
                    f"'{fe.method}' is saved in the config but not previewed here — it's "
                    f"too slow for interactive use. Generate the config (block 4) and run "
                    f"the batch; it'll populate there. Use 'cp_emulator' to preview live."
                )
                return
            n_ch = len(fe.channels) if fe.channels else (
                self.image.shape[0] if self.image.ndim == 3 else 1
            )
            if fe.include_correlation and n_ch < 2:
                print("Note: correlation needs ≥2 channels — skipped for this selection.")
            df = extract_features(
                self.image, nuclei_masks=nuclei, cell_masks=cells,
                channel_names=fe.channel_names, channels=fe.channels,
                compartments=fe.compartments, include_texture=fe.include_texture,
                include_correlation=fe.include_correlation,
                include_neighbors=fe.include_neighbors, method=fe.method,
            )
            print(f"Features: {len(df)} objects × {len(df.columns)} columns")
            try:
                from IPython.display import display

                display(df.head())
            except Exception:
                print(df.head())

    def _config_recap(self, config) -> List[str]:
        """Human-readable summary of the choices captured in a config."""
        lines = [f"Mode: {config.mode}"]
        if config.mode == "dual" and config.dual is not None:
            d = config.dual
            lines.append(
                f"  Nuclei — channel {d.nuclei_channel}, model {d.nuclei_model}, "
                f"diameter {d.nuclei_diameter}, flow {d.nuclei_flow_threshold}, "
                f"cellprob {d.nuclei_cellprob_threshold}"
            )
            lines.append(
                f"  Cells  — channel {d.cyto_channel}, model {d.cell_model}, "
                f"diameter {d.cell_diameter}, flow {d.cell_flow_threshold}, "
                f"cellprob {d.cell_cellprob_threshold}"
            )
            lines.append(
                f"  Reconcile — {config.reconcile or 'off'} "
                f"(drops nucleus-less cells, matches nucleus↔cell labels)"
            )
        else:
            lines.append(
                f"  {config.mode} — channel {config.channel_to_segment}, "
                f"model {config.model}, diameter {config.diameter}, "
                f"flow {config.flow_threshold}, cellprob {config.cellprob_threshold}"
            )
        lines.append(
            f"Options: edge-removal {'on' if config.remove_edge_cells else 'off'}, "
            f"z-project {'on' if config.z_project else 'off'}, "
            f"gpu {'on' if config.gpu else 'off'}"
        )
        fe = config.feature_extraction
        if fe is not None and fe.enabled:
            chans = "all" if not fe.channels else ",".join(map(str, fe.channels))
            comps = ",".join(fe.compartments) if fe.compartments else "all"
            extras = [
                name for name, flag in (
                    ("texture", fe.include_texture),
                    ("correlation", fe.include_correlation),
                    ("neighbors", fe.include_neighbors),
                ) if flag
            ]
            detail = f"Features: {fe.method}, channels {chans}, compartments {comps}"
            if extras:
                detail += f", +{'+'.join(extras)}"
            if fe.combine_tables:
                detail += ", combined table"
            lines.append(detail)
        else:
            lines.append("Features: off")
        lines.append(f"Inputs: {config.input_dir}/{config.file_pattern}")
        return lines

    def _on_generate(self, _btn) -> None:
        """Write the run config and report where the config, masks, and features go.

        Everything organizes under the run name: the config lands in
        ``configs/<name>.yaml`` and the batch run dumps masks to
        ``masks/<name>/`` and features to ``features/<name>/`` (all siblings of
        the data folder).
        """
        self.config_output.clear_output(wait=True)
        with self.config_output:
            name = Path(self.config_name_w.value.strip() or "segmentation_config").stem
            root = Path(self.input_dir_w.value).resolve().parent
            config_path = root / "configs" / f"{name}.yaml"
            masks_dir = root / "masks" / name
            features_dir = root / "features" / name

            config = self.build_config()
            config.output_dir = str(masks_dir)
            config.features_dir = str(features_dir)
            config.to_yaml(config_path)

            print(f"Config '{name}' saved — here's what you chose:\n")
            for line in self._config_recap(config):
                print(line)
            print(f"\nConfig    → {config_path}")
            print(f"Masks     → {masks_dir}/  (the sbatch run creates this)")
            print(f"Features  → {features_dir}/  (the sbatch run creates this)")
            print("\nRun batch segmentation (from the repo root) with:")
            print(f"  sbatch scripts/run_segmentation.sh {config_path}")

    def _ipython_display_(self) -> None:
        """Display all four blocks when the object is the last cell expression."""
        from IPython.display import display

        display(self.panel)
