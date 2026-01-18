"""Configuration management for batch segmentation jobs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import yaml


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

        Args:
            input_file: Input file path.

        Returns:
            Output file path with _mask.tif suffix.
        """
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{input_file.stem}_mask.tif"
