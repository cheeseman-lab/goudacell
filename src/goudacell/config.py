"""Configuration management for batch segmentation jobs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Union

import yaml


@dataclass
class FeatureExtractionParams:
    """Parameters for optional feature extraction after segmentation.

    Attributes:
        enabled: Whether to run feature extraction.
        method: Extraction backend. One of "cp_emulator" (built-in, default),
            "cp_measure" (requires cp-measure package), or "cellprofiler"
            (requires cellprofiler-core package).
        channel_names: Names for each channel (auto-detected if None).
        channels: Channel indices to extract from (None = all channels). Use to
            restrict extraction to specific channels (e.g. [2] for one stain).
        compartments: Compartments to measure, any of "nucleus", "cell",
            "cytoplasm" (None = all available). cp_emulator backend only.
        include_texture: Include Haralick/PFTAS texture features (slower).
        include_correlation: Include channel correlation features.
        include_neighbors: Include neighbor measurements.
        output_path: Per-file output path for features CSV. May contain a
            "{stem}" placeholder (relative to output_dir).
        combine_tables: Also write one combined CSV across all input files,
            with a "filename" column identifying each file's rows.
        combined_output: Filename for the combined table (relative to output_dir).
    """

    enabled: bool = False
    method: Literal["cp_emulator", "cp_measure", "cellprofiler"] = "cp_emulator"
    channel_names: Optional[List[str]] = None
    channels: Optional[List[int]] = None
    compartments: Optional[List[str]] = None
    include_texture: bool = True
    include_correlation: bool = True
    include_neighbors: bool = True
    output_path: str = "features.csv"
    combine_tables: bool = False
    combined_output: str = "features_combined.csv"
    # CellProfiler headless options (only used when method="cellprofiler")
    pipeline_file: Optional[str] = None
    cellprofiler_cmd: str = "cellprofiler"


@dataclass
class DualSegmentationParams:
    """Parameters for dual nuclei + cell segmentation mode.

    Attributes:
        nuclei_channel: Index of the nuclear channel (e.g., DAPI).
        cyto_channel: Index of the cytoplasmic channel.
        nuclei_diameter: Estimated nuclear diameter in pixels.
        cell_diameter: Estimated cell diameter in pixels.
        cell_model: Cellpose model for cell segmentation.
        nuclei_model: Cellpose model for nuclei segmentation.
        nuclei_flow_threshold: Flow threshold for nuclei segmentation.
        nuclei_cellprob_threshold: Cell probability threshold for nuclei.
        cell_flow_threshold: Flow threshold for cell segmentation.
        cell_cellprob_threshold: Cell probability threshold for cells.
    """

    nuclei_channel: int = 0
    cyto_channel: int = 1
    nuclei_diameter: float = 15.0
    cell_diameter: float = 40.0
    cell_model: str = "cyto3"
    nuclei_model: str = "nuclei"
    nuclei_flow_threshold: float = 0.4
    nuclei_cellprob_threshold: float = 0.0
    cell_flow_threshold: float = 0.4
    cell_cellprob_threshold: float = 0.0


@dataclass
class SegmentationConfig:
    """Configuration for a segmentation job.

    Attributes:
        input_dir: Directory containing input images.
        output_dir: Directory for output masks.
        file_pattern: Glob pattern for input files (e.g., "*.nd2", "*.tif").
        model: Cellpose model to use.
        diameter: Cell diameter in pixels.
        channels: Channel specification [cytoplasm, nucleus].
        flow_threshold: Flow error threshold.
        cellprob_threshold: Cell probability threshold.
        gpu: Whether to use GPU.
        remove_edge_cells: Whether to remove cells touching borders.
        z_project: Whether to max-project Z-stacks.
        channel_to_segment: Which channel to use (for multi-channel images).
        mode: Segmentation mode ("nuclei", "cells", or "dual").
        dual: Parameters for dual mode (only used if mode="dual").
        reconcile: Dual-mode reconciliation method matching nuclei to cells and
            dropping nucleus-less cells ("consensus" or "contained_in_cells").
            None skips reconciliation. Ignored outside dual mode.
    """

    input_dir: str
    output_dir: str
    features_dir: Optional[str] = None
    file_pattern: str = "*.tif"
    model: str = "cyto3"
    diameter: float = 30.0
    channels: List[int] = field(default_factory=lambda: [0, 0])
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    gpu: bool = True
    remove_edge_cells: bool = True
    z_project: bool = True
    channel_to_segment: Optional[int] = None
    mode: Literal["nuclei", "cells", "dual"] = "cells"
    dual: Optional[DualSegmentationParams] = None
    reconcile: Optional[str] = "consensus"
    feature_extraction: Optional[FeatureExtractionParams] = None

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "SegmentationConfig":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            SegmentationConfig instance.
        """
        yaml_path = Path(yaml_path)
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        # Handle nested dual params
        if "dual" in data and data["dual"] is not None:
            data["dual"] = DualSegmentationParams(**data["dual"])

        # Handle nested feature_extraction params
        if "feature_extraction" in data and data["feature_extraction"] is not None:
            data["feature_extraction"] = FeatureExtractionParams(**data["feature_extraction"])

        return cls(**data)

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to a YAML file.

        Args:
            yaml_path: Path to save the YAML file.
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "features_dir": self.features_dir,
            "file_pattern": self.file_pattern,
            "model": self.model,
            "diameter": self.diameter,
            "channels": self.channels,
            "flow_threshold": self.flow_threshold,
            "cellprob_threshold": self.cellprob_threshold,
            "gpu": self.gpu,
            "remove_edge_cells": self.remove_edge_cells,
            "z_project": self.z_project,
            "channel_to_segment": self.channel_to_segment,
            "mode": self.mode,
        }

        # Add dual params if in dual mode
        if self.mode == "dual" and self.dual is not None:
            data["dual"] = {
                "nuclei_channel": self.dual.nuclei_channel,
                "cyto_channel": self.dual.cyto_channel,
                "nuclei_diameter": self.dual.nuclei_diameter,
                "cell_diameter": self.dual.cell_diameter,
                "cell_model": self.dual.cell_model,
                "nuclei_model": self.dual.nuclei_model,
                "nuclei_flow_threshold": self.dual.nuclei_flow_threshold,
                "nuclei_cellprob_threshold": self.dual.nuclei_cellprob_threshold,
                "cell_flow_threshold": self.dual.cell_flow_threshold,
                "cell_cellprob_threshold": self.dual.cell_cellprob_threshold,
            }
            data["reconcile"] = self.reconcile

        # Add feature extraction params if enabled
        if self.feature_extraction is not None:
            fe = self.feature_extraction
            fe_data = {
                "enabled": fe.enabled,
                "method": fe.method,
                "channel_names": fe.channel_names,
                "channels": fe.channels,
                "compartments": fe.compartments,
                "include_texture": fe.include_texture,
                "include_correlation": fe.include_correlation,
                "include_neighbors": fe.include_neighbors,
                "output_path": fe.output_path,
                "combine_tables": fe.combine_tables,
                "combined_output": fe.combined_output,
            }
            if fe.method == "cellprofiler":
                fe_data["pipeline_file"] = fe.pipeline_file
                fe_data["cellprofiler_cmd"] = fe.cellprofiler_cmd
            data["feature_extraction"] = fe_data

        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def get_input_files(self) -> List[Path]:
        """Get list of input files matching the pattern.

        Returns:
            List of Path objects for matching files.
        """
        input_path = Path(self.input_dir)
        return sorted(input_path.glob(self.file_pattern))

    def get_output_path(self, input_file: Path) -> Path:
        """Get output path for a given input file.

        For nuclei mode: {stem}_nuclei_mask.tif
        For cells mode: {stem}_mask.tif (backward compatible)
        For dual mode: use get_dual_output_paths() instead.

        Args:
            input_file: Input file path.

        Returns:
            Output file path with appropriate suffix.
        """
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.mode == "nuclei":
            return output_dir / f"{input_file.stem}_nuclei_mask.tif"
        else:
            # cells mode (backward compatible)
            return output_dir / f"{input_file.stem}_mask.tif"

    def get_dual_output_paths(self, input_file: Path) -> tuple:
        """Get output paths for dual mode (nuclei and cell masks).

        Args:
            input_file: Input file path.

        Returns:
            Tuple of (nuclei_mask_path, cell_mask_path).
        """
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return (
            output_dir / f"{input_file.stem}_nuclei_mask.tif",
            output_dir / f"{input_file.stem}_cell_mask.tif",
        )

    def get_features_output_path(self, input_file: Path) -> Path:
        """Get output path for feature extraction CSV.

        Args:
            input_file: Input file path.

        Returns:
            Path for the features CSV file.
        """
        output_dir = Path(self.features_dir or self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.feature_extraction and self.feature_extraction.output_path:
            # Use configured output path (may include {stem} placeholder)
            output_name = self.feature_extraction.output_path.replace("{stem}", input_file.stem)
            return output_dir / output_name
        else:
            return output_dir / f"{input_file.stem}_features.csv"

    def get_combined_output_path(self) -> Path:
        """Get output path for the combined feature table.

        Returns:
            Path for the single combined features CSV.
        """
        output_dir = Path(self.features_dir or self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        name = "features_combined.csv"
        if self.feature_extraction and self.feature_extraction.combined_output:
            name = self.feature_extraction.combined_output
        return output_dir / name
