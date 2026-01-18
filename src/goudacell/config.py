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
        channel_names: Names for each channel (auto-detected if None).
        include_texture: Include Haralick/PFTAS texture features (slower).
        include_correlation: Include channel correlation features.
        include_neighbors: Include neighbor measurements.
        output_path: Output path for features CSV (relative to output_dir).
    """

    enabled: bool = False
    channel_names: Optional[List[str]] = None
    include_texture: bool = True
    include_correlation: bool = True
    include_neighbors: bool = True
    output_path: str = "features.csv"


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
    """

    input_dir: str
    output_dir: str
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

        # Add feature extraction params if enabled
        if self.feature_extraction is not None:
            data["feature_extraction"] = {
                "enabled": self.feature_extraction.enabled,
                "channel_names": self.feature_extraction.channel_names,
                "include_texture": self.feature_extraction.include_texture,
                "include_correlation": self.feature_extraction.include_correlation,
                "include_neighbors": self.feature_extraction.include_neighbors,
                "output_path": self.feature_extraction.output_path,
            }

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
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.feature_extraction and self.feature_extraction.output_path:
            # Use configured output path (may include {stem} placeholder)
            output_name = self.feature_extraction.output_path.replace("{stem}", input_file.stem)
            return output_dir / output_name
        else:
            return output_dir / f"{input_file.stem}_features.csv"
