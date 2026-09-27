# ruff: noqa: E501
"""CellProfiler headless backend on a tiny synthetic image and masks.

Runs the CellProfiler goudacell finds (``GOUDACELL_CELLPROFILER``, ``cellprofiler`` on
PATH, or the ``goudacell_cp`` conda env from ``scripts/setup_cellprofiler_env.sh``), and
skips when there is none. The pipeline below loads the files goudacell stages
(``DAPI.tif``, ``GFP.tif`` and the masks as objects ``Nuclei``, ``Cells``, ``Cytoplasm``),
measures intensity and size/shape, and exports CSVs; the default pipeline is run too.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from goudacell import features_cellprofiler
from goudacell.features import extract_features
from goudacell.features_cellprofiler import (
    _label_object_table,
    check_cellprofiler,
    default_pipeline,
    extract_features_cellprofiler,
    find_cellprofiler,
)

CELLPROFILER = find_cellprofiler()
needs_cellprofiler = pytest.mark.skipif(
    CELLPROFILER is None or shutil.which(CELLPROFILER) is None,
    reason="CellProfiler not found (run scripts/setup_cellprofiler_env.sh)",
)

HEADER = """CellProfiler Pipeline: http://www.cellprofiler.org
Version:5
DateRevision:4281
GitHash:
ModuleCount:{count}
HasImagePlaneDetails:False
"""

INPUT_MODULES = r"""
Images:[module_num:1|svn_version:'Unknown'|variable_revision_number:2|show_window:False|notes:[]|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    :
    Filter images?:Images only
    Select the rule criteria:and (extension does isimage) (directory doesnot containregexp "[\\\\/]\\.")

Metadata:[module_num:2|svn_version:'Unknown'|variable_revision_number:6|show_window:False|notes:[]|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    Extract metadata?:No
    Metadata data type:Text
    Metadata types:{}
    Extraction method count:1
    Metadata extraction method:Extract from file/folder names
    Metadata source:File name
    Regular expression to extract from file name:^(?P<Plate>.*)_(?P<Well>[A-P][0-9]{2})_s(?P<Site>[0-9])_w(?P<ChannelNumber>[0-9])
    Regular expression to extract from folder name:(?P<Date>[0-9]{4}_[0-9]{2}_[0-9]{2})$
    Extract metadata from:All images
    Select the filtering criteria:and (file does contain "")
    Metadata file location:Elsewhere...|
    Match file and image metadata:[]
    Use case insensitive matching?:No
    Metadata file name:None
    Does cached metadata exist?:No

NamesAndTypes:[module_num:3|svn_version:'Unknown'|variable_revision_number:8|show_window:False|notes:[]|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    Assign a name to:Images matching rules
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Match metadata:[]
    Image set matching method:Order
    Set intensity range from:Image metadata
    Assignments count:5
    Single images count:0
    Maximum intensity:255.0
    Process as 3D?:No
    Relative pixel spacing in X:1.0
    Relative pixel spacing in Y:1.0
    Relative pixel spacing in Z:1.0
    Select the rule criteria:and (file does containregexp "^DAPI\.tif$")
    Name to assign these images:DAPI
    Name to assign these objects:Cell
    Select the image type:Grayscale image
    Set intensity range from:Image metadata
    Maximum intensity:255.0
    Select the rule criteria:and (file does containregexp "^GFP\.tif$")
    Name to assign these images:GFP
    Name to assign these objects:Nucleus
    Select the image type:Grayscale image
    Set intensity range from:Image metadata
    Maximum intensity:255.0
    Select the rule criteria:and (file does containregexp "^nuclei_mask\.tif$")
    Name to assign these images:DNA
    Name to assign these objects:Nuclei
    Select the image type:Objects
    Set intensity range from:Image metadata
    Maximum intensity:255.0
    Select the rule criteria:and (file does containregexp "^cell_mask\.tif$")
    Name to assign these images:Actin
    Name to assign these objects:Cells
    Select the image type:Objects
    Set intensity range from:Image metadata
    Maximum intensity:255.0
    Select the rule criteria:and (file does containregexp "^cytoplasm_mask\.tif$")
    Name to assign these images:Channel1
    Name to assign these objects:Cytoplasm
    Select the image type:Objects
    Set intensity range from:Image metadata
    Maximum intensity:255.0

Groups:[module_num:4|svn_version:'Unknown'|variable_revision_number:2|show_window:False|notes:[]|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    Do you want to group your images?:No
    grouping metadata count:1
    Metadata category:None
"""

MEASURE_MODULES = """
MeasureObjectIntensity:[module_num:5|svn_version:'Unknown'|variable_revision_number:4|show_window:False|notes:[]|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    Select images to measure:DAPI, GFP
    Select objects to measure:Nuclei, Cells, Cytoplasm

MeasureObjectSizeShape:[module_num:6|svn_version:'Unknown'|variable_revision_number:3|show_window:False|notes:[]|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    Select object sets to measure:Nuclei, Cells, Cytoplasm
    Calculate the Zernike features?:No
    Calculate the advanced features?:No

ExportToSpreadsheet:[module_num:7|svn_version:'Unknown'|variable_revision_number:13|show_window:False|notes:[]|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    Select the column delimiter:Comma (",")
    Add image metadata columns to your object data file?:No
    Add image file and folder names to your object data file?:No
    Select the measurements to export:No
    Calculate the per-image mean values for object measurements?:No
    Calculate the per-image median values for object measurements?:No
    Calculate the per-image standard deviation values for object measurements?:No
    Output file location:Default Output Folder|
    Create a GenePattern GCT file?:No
    Select source of sample row name:Metadata
    Select the image to use as the identifier:None
    Select the metadata to use as the identifier:None
    Export all measurement types?:Yes
    Press button to select measurements:
    Representation of Nan/Inf:NaN
    Add a prefix to file names?:Yes
    Filename prefix:MyExpt_
    Overwrite existing files without warning?:No
    Data to export:Do not use
    Combine these object measurements with those of the previous object?:No
    File name:DATA.csv
    Use the object name for the file name?:Yes
"""

# Labels 1, 2, 5, 7: gaps on purpose, CellProfiler numbers objects 1..n
LABELS = [1, 2, 5, 7]
CENTERS = [(24, 24), (24, 72), (72, 24), (72, 72)]


@pytest.fixture
def tile():
    """(image, nuclei, cells): two channels, four cells with a nucleus each."""
    from skimage.draw import disk

    rng = np.random.default_rng(0)
    nuclei = np.zeros((96, 96), np.uint16)
    cells = np.zeros_like(nuclei)
    for label, center in zip(LABELS, CENTERS):
        cells[disk(center, 18, shape=cells.shape)] = label
        nuclei[disk(center, 8, shape=cells.shape)] = label
    noise = rng.integers(0, 100, (2, *nuclei.shape))
    image = np.stack([(nuclei > 0) * 3000, (cells > 0) * 1000 + cells * 50]) + noise
    return image.astype(np.uint16), nuclei, cells


def _pipeline(path: Path, measure: bool = True) -> Path:
    modules = INPUT_MODULES + (MEASURE_MODULES if measure else "")
    path.write_text(HEADER.format(count=7 if measure else 4) + modules)
    return path


@needs_cellprofiler
def test_cellprofiler_table(tile, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image, nuclei, cells = tile
    df = extract_features(
        image, nuclei, cells, channel_names=["DAPI", "GFP"], method="cellprofiler",
        pipeline_file=_pipeline(tmp_path / "measure.cppipe"), cellprofiler_cmd=CELLPROFILER,
    )

    # One row per mask label, keyed by the label, not CellProfiler's ObjectNumber
    assert df["label"].tolist() == LABELS
    counts = np.bincount(nuclei.ravel())
    assert df["nucleus_AreaShape_Area"].tolist() == [counts[label] for label in LABELS]
    assert df["cell_AreaShape_Area"].tolist() == [np.sum(cells == lab) for lab in LABELS]
    # GFP scales with the cell label, so a misaligned join would scramble the order
    assert df["cell_Intensity_MeanIntensity_GFP"].is_monotonic_increasing
    for compartment in ("nucleus", "cell", "cytoplasm"):
        for channel in ("DAPI", "GFP"):
            assert f"{compartment}_Intensity_MeanIntensity_{channel}" in df.columns
    assert not [c for c in df.columns if df[c].isna().all()]
    assert not list(tmp_path.glob("goudacell_cp_*"))

    nucleus_only = extract_features(
        image, nuclei, cells, channel_names=["DAPI", "GFP"], method="cellprofiler",
        pipeline_file=tmp_path / "measure.cppipe", cellprofiler_cmd=CELLPROFILER,
        compartments=["nucleus"],
    )
    assert list(nucleus_only.columns) == [
        c for c in df.columns if not c.startswith(("cell_", "cytoplasm_"))
    ]


@needs_cellprofiler
def test_pipeline_without_object_tables_raises(tile, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image, nuclei, cells = tile
    with pytest.raises(RuntimeError, match="no Nuclei/Cells/Cytoplasm table"):
        extract_features(
            image, nuclei, cells, channel_names=["DAPI", "GFP"], method="cellprofiler",
            pipeline_file=_pipeline(tmp_path / "load_only.cppipe", measure=False),
            cellprofiler_cmd=CELLPROFILER,
        )


def test_object_numbers_map_to_labels():
    exported = pd.DataFrame(
        {"ImageNumber": [1, 1, 1], "ObjectNumber": [2, 1, 3], "AreaShape_Area": [20, 10, 30]}
    )
    table = _label_object_table(exported, "Cells", np.array([4, 9, 12]))
    assert table.columns.tolist() == ["label", "cell_AreaShape_Area"]
    assert table["label"].tolist() == [9, 4, 12]


@needs_cellprofiler
def test_default_pipeline_runs(tile, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image, nuclei, cells = tile
    # No pipeline and no command: the default pipeline, run by the CellProfiler found
    df = extract_features(image, nuclei, cells, channel_names=["DAPI", "GFP"], method="cellprofiler")

    assert df["label"].tolist() == LABELS
    assert df["cell_AreaShape_Area"].tolist() == [np.sum(cells == lab) for lab in LABELS]
    assert df["cell_Intensity_MeanIntensity_GFP"].is_monotonic_increasing
    for column in (
        "nucleus_AreaShape_Zernike_0_0",
        "cytoplasm_Intensity_MeanIntensity_DAPI",
        "cell_Texture_Contrast_GFP_3_00_256",
        "cell_Correlation_Correlation_DAPI_GFP",
        "nucleus_Neighbors_NumberOfNeighbors_Adjacent",
        "cell_Neighbors_NumberOfNeighbors_Adjacent",
    ):
        assert column in df.columns
    assert not list(tmp_path.glob("goudacell_cp_*"))

    lean = extract_features(
        image, nuclei, None, channel_names=["DAPI", "GFP"], method="cellprofiler",
        include_texture=False, include_correlation=False, include_neighbors=False,
    )
    assert lean["label"].tolist() == LABELS
    assert not [c for c in lean.columns if c.startswith(("cell_", "cytoplasm_"))]
    assert not [c for c in lean.columns if "Texture" in c or "Correlation_" in c]


def test_default_pipeline_modules():
    full = default_pipeline(["DAPI", "GFP"])
    assert "ModuleCount:11" in full
    assert full.count("module_num:") == 11 and "module_num:11|" in full
    assert "@" not in full
    assert "Select objects to measure:Nuclei, Cells, Cytoplasm" in full
    assert full.count("MeasureObjectNeighbors:") == 2

    lean = default_pipeline(["DAPI"], objects=("Nuclei",), include_texture=False)
    assert "ModuleCount:8" in lean
    for module in ("MeasureTexture:", "MeasureColocalization:", "Cells"):
        assert module not in lean
    assert lean.count("MeasureObjectNeighbors:") == 1


def _executable(path: Path, body: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def no_cellprofiler(tmp_path, monkeypatch):
    """An environment with no CellProfiler anywhere find_cellprofiler looks."""
    monkeypatch.delenv("GOUDACELL_CELLPROFILER", raising=False)
    monkeypatch.delenv("CONDA_EXE", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "path"))
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "python"))
    return tmp_path


def test_find_cellprofiler_order(no_cellprofiler, monkeypatch):
    root = no_cellprofiler
    assert find_cellprofiler() is None

    # A goudacell_cp env under an unrelated conda base, known only to conda env list
    listed = _executable(root / "elsewhere/goudacell_cp/bin/cellprofiler")
    envs = {"envs": [str(root / "elsewhere/other"), str(listed.parents[1])]}
    _executable(root / "path/conda", f"echo '{json.dumps(envs)}'")
    assert find_cellprofiler() == str(listed)

    # The env under the conda base of this interpreter, then of CONDA_EXE
    in_prefix_base = _executable(root / "prefix_base/envs/goudacell_cp/bin/cellprofiler")
    monkeypatch.setattr(sys, "prefix", str(root / "prefix_base/envs/goudacell"))
    assert find_cellprofiler() == str(in_prefix_base)
    in_conda_base = _executable(root / "base/envs/goudacell_cp/bin/cellprofiler")
    monkeypatch.setenv("CONDA_EXE", str(root / "base/bin/conda"))
    assert find_cellprofiler() == str(in_conda_base)

    # cellprofiler on PATH beats any conda env, GOUDACELL_CELLPROFILER beats PATH
    on_path = _executable(root / "path/cellprofiler")
    assert find_cellprofiler() == str(on_path)
    monkeypatch.setenv("GOUDACELL_CELLPROFILER", "/opt/cp/bin/cellprofiler")
    assert find_cellprofiler() == "/opt/cp/bin/cellprofiler"


def test_missing_cellprofiler_names_setup_script(no_cellprofiler, tile):
    image, nuclei, cells = tile
    with pytest.raises(RuntimeError, match="setup_cellprofiler_env.sh"):
        extract_features_cellprofiler(image, nuclei, cells)


@pytest.fixture
def fresh_checks(monkeypatch):
    """check_cellprofiler without the commands earlier tests cached."""
    monkeypatch.setattr(features_cellprofiler, "_CHECKED_VERSIONS", {})


def test_check_cellprofiler(no_cellprofiler, fresh_checks, monkeypatch):
    root = no_cellprofiler
    with pytest.raises(RuntimeError, match="CellProfiler not found.*setup_cellprofiler_env.sh"):
        check_cellprofiler()

    # A supported version passes, and `--version` runs once per command
    calls = root / "calls"
    good = _executable(root / "good/cellprofiler", f"echo run >> {calls}; echo 4.2.8.1")
    monkeypatch.setenv("GOUDACELL_CELLPROFILER", str(good))
    assert check_cellprofiler() == str(good)
    assert check_cellprofiler(str(good)) == str(good)
    assert calls.read_text().count("run") == 1

    old = _executable(root / "old/cellprofiler", "echo 4.1.3")
    with pytest.raises(RuntimeError, match="from cellprofiler_cmd.*is CellProfiler 4.1.3, not 4.2.x"):
        check_cellprofiler(str(old))
    not_cp = _executable(root / "other/cellprofiler", "echo boom >&2; exit 2")
    with pytest.raises(RuntimeError, match="(?s)not a working CellProfiler.*exited 2.*boom"):
        check_cellprofiler(str(not_cp))
    monkeypatch.setenv("GOUDACELL_CELLPROFILER", str(root / "missing/cellprofiler"))
    with pytest.raises(RuntimeError, match="from the GOUDACELL_CELLPROFILER env var.*doesn't exist"):
        check_cellprofiler()


def test_legacy_default_command_falls_through_to_discovery(
    no_cellprofiler, fresh_checks, monkeypatch, caplog
):
    root = no_cellprofiler
    monkeypatch.setattr(features_cellprofiler, "_legacy_noted", False)
    found = _executable(root / "env/cellprofiler", "echo 4.2.8.1")
    monkeypatch.setenv("GOUDACELL_CELLPROFILER", str(found))
    with caplog.at_level("WARNING", logger="goudacell.features_cellprofiler"):
        assert check_cellprofiler("cellprofiler") == str(found)
        assert check_cellprofiler("cellprofiler") == str(found)
    assert sum("not on PATH" in r.message for r in caplog.records) == 1

    # On PATH, and any explicit path, the configured command still wins
    on_path = _executable(root / "path/cellprofiler", "echo 4.2.8.1")
    assert check_cellprofiler("cellprofiler") == "cellprofiler"
    assert check_cellprofiler(str(on_path)) == str(on_path)


def test_cli_fails_before_segmenting(no_cellprofiler, fresh_checks, monkeypatch):
    from typer.testing import CliRunner

    from goudacell.cli import app

    root = no_cellprofiler
    (root / "data").mkdir()
    config = root / "config.yaml"
    config.write_text(
        f"input_dir: {root / 'data'}\noutput_dir: {root / 'out'}\n"
        "feature_extraction:\n  enabled: true\n  method: cellprofiler\n"
    )
    monkeypatch.setenv("GOUDACELL_CELLPROFILER", str(root / "missing/cellprofiler"))
    result = CliRunner().invoke(app, ["segment", str(config)])
    assert result.exit_code == 1
    assert "GOUDACELL_CELLPROFILER" in result.output and "No files found" not in result.output
