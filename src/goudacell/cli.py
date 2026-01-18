"""Command-line interface for GoudaCell."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    name="goudacell",
    help="HPC-compatible cell segmentation using Cellpose.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def segment(
    config: Path = typer.Argument(..., help="Path to YAML configuration file"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would be done"),
) -> None:
    """Run batch segmentation using a YAML config file."""
    from goudacell.config import SegmentationConfig
    from goudacell.io import load_image, save_mask
    from goudacell.segment import segment as run_segment
    from goudacell.segment import segment_nuclei_and_cells

    # Load config
    cfg = SegmentationConfig.from_yaml(config)

    # Get input files
    input_files = cfg.get_input_files()

    if not input_files:
        console.print(f"[red]No files found matching '{cfg.file_pattern}' in {cfg.input_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"Found [green]{len(input_files)}[/green] files to process")
    console.print(f"Mode: [cyan]{cfg.mode}[/cyan]")

    if cfg.mode == "dual" and cfg.dual:
        console.print(
            f"Nuclei: diameter=[cyan]{cfg.dual.nuclei_diameter}[/cyan], "
            f"model=[cyan]{cfg.dual.nuclei_model}[/cyan]"
        )
        console.print(
            f"Cells: diameter=[cyan]{cfg.dual.cell_diameter}[/cyan], "
            f"model=[cyan]{cfg.dual.cell_model}[/cyan]"
        )
    else:
        console.print(f"Model: [cyan]{cfg.model}[/cyan], Diameter: [cyan]{cfg.diameter}[/cyan]")
    console.print(f"GPU: [cyan]{cfg.gpu}[/cyan]")

    if dry_run:
        console.print("\n[yellow]Dry run - files that would be processed:[/yellow]")
        for f in input_files:
            if cfg.mode == "dual":
                nuclei_out, cell_out = cfg.get_dual_output_paths(f)
                console.print(f"  {f.name} -> {nuclei_out.name}, {cell_out.name}")
            else:
                output = cfg.get_output_path(f)
                console.print(f"  {f.name} -> {output.name}")
        return

    # Process each file
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for input_file in input_files:
            task = progress.add_task(f"Processing {input_file.name}...", total=None)

            try:
                # Load image
                image = load_image(
                    input_file,
                    channel=cfg.channel_to_segment,
                    z_project=cfg.z_project,
                )

                if cfg.mode == "dual" and cfg.dual:
                    # Dual mode: segment nuclei and cells
                    nuclei_masks, cell_masks = segment_nuclei_and_cells(
                        image,
                        nuclei_channel=cfg.dual.nuclei_channel,
                        cyto_channel=cfg.dual.cyto_channel,
                        nuclei_diameter=cfg.dual.nuclei_diameter,
                        cell_diameter=cfg.dual.cell_diameter,
                        cell_model=cfg.dual.cell_model,
                        nuclei_model=cfg.dual.nuclei_model,
                        nuclei_flow_threshold=cfg.dual.nuclei_flow_threshold,
                        nuclei_cellprob_threshold=cfg.dual.nuclei_cellprob_threshold,
                        cell_flow_threshold=cfg.dual.cell_flow_threshold,
                        cell_cellprob_threshold=cfg.dual.cell_cellprob_threshold,
                        gpu=cfg.gpu,
                        remove_edge_cells=cfg.remove_edge_cells,
                    )

                    # Save both outputs
                    nuclei_path, cell_path = cfg.get_dual_output_paths(input_file)
                    save_mask(nuclei_masks, nuclei_path)
                    save_mask(cell_masks, cell_path)

                    n_nuclei = len(set(nuclei_masks.flat)) - 1
                    n_cells = len(set(cell_masks.flat)) - 1

                    # Feature extraction if enabled
                    if cfg.feature_extraction and cfg.feature_extraction.enabled:
                        from goudacell.features import extract_features

                        features_df = extract_features(
                            image,
                            nuclei_masks=nuclei_masks,
                            cell_masks=cell_masks,
                            channel_names=cfg.feature_extraction.channel_names,
                            include_texture=cfg.feature_extraction.include_texture,
                            include_correlation=cfg.feature_extraction.include_correlation,
                            include_neighbors=cfg.feature_extraction.include_neighbors,
                        )
                        features_path = cfg.get_features_output_path(input_file)
                        features_df.to_csv(features_path, index=False)
                        n_features = len(features_df.columns)
                        progress.update(
                            task,
                            description=f"[green]Done[/green] {input_file.name} "
                            f"({n_nuclei} nuclei, {n_cells} cells, {n_features} features)",
                        )
                    else:
                        progress.update(
                            task,
                            description=f"[green]Done[/green] {input_file.name} "
                            f"({n_nuclei} nuclei, {n_cells} cells)",
                        )
                else:
                    # Single mode: nuclei or cells
                    model = cfg.model
                    if cfg.mode == "nuclei":
                        model = "nuclei"

                    masks = run_segment(
                        image,
                        diameter=cfg.diameter,
                        model=model,
                        channels=cfg.channels,
                        flow_threshold=cfg.flow_threshold,
                        cellprob_threshold=cfg.cellprob_threshold,
                        gpu=cfg.gpu,
                        remove_edge_cells=cfg.remove_edge_cells,
                    )

                    # Save output
                    output_path = cfg.get_output_path(input_file)
                    save_mask(masks, output_path)

                    n_cells = len(set(masks.flat)) - 1  # Exclude background
                    label = "nuclei" if cfg.mode == "nuclei" else "cells"
                    progress.update(
                        task,
                        description=f"[green]Done[/green] {input_file.name} ({n_cells} {label})",
                    )

            except Exception as e:
                progress.update(task, description=f"[red]Failed[/red] {input_file.name}: {e}")
                console.print_exception()


@app.command()
def single(
    input_file: Path = typer.Argument(..., help="Path to input image"),
    output_file: Path = typer.Argument(..., help="Path to output mask"),
    diameter: float = typer.Option(30.0, "--diameter", "-d", help="Cell diameter in pixels"),
    model: str = typer.Option("cyto3", "--model", "-m", help="Cellpose model"),
    gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="Use GPU"),
    channel: Optional[int] = typer.Option(None, "--channel", "-c", help="Channel to segment"),
) -> None:
    """Segment a single image file."""
    from goudacell.io import load_image, save_mask
    from goudacell.segment import segment as run_segment

    console.print(f"Loading [cyan]{input_file}[/cyan]...")

    # Load image
    image = load_image(input_file, channel=channel)

    console.print(f"Image shape: {image.shape}")
    console.print(
        f"Running segmentation with model=[cyan]{model}[/cyan], diameter=[cyan]{diameter}[/cyan]..."
    )

    # Run segmentation
    masks = run_segment(
        image,
        diameter=diameter,
        model=model,
        gpu=gpu,
    )

    # Save output
    save_mask(masks, output_file)

    n_cells = len(set(masks.flat)) - 1
    console.print(f"[green]Done![/green] Found {n_cells} cells. Saved to {output_file}")


@app.command()
def version() -> None:
    """Show version information."""
    from goudacell import __version__

    console.print(f"GoudaCell version: [cyan]{__version__}[/cyan]")

    try:
        from goudacell.segment import get_cellpose_version

        cp_version = get_cellpose_version()
        console.print(f"Cellpose version: [cyan]{cp_version[0]}.{cp_version[1]}[/cyan]")

        if cp_version[0] >= 4:
            console.print("  Available models: [yellow]cpsam[/yellow]")
        else:
            console.print("  Available models: [yellow]cyto3, nuclei, cyto2, cyto[/yellow]")
    except ImportError:
        console.print("[yellow]Cellpose not installed[/yellow]")
        console.print("  Install with: uv pip install -e '.[cellpose3]' or '.[cellpose4]'")


if __name__ == "__main__":
    app()
