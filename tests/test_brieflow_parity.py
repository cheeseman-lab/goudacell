"""Parity of goudacell with brieflow's phenotype code.

goudacell vendors brieflow's phenotype modules verbatim (``src/goudacell/brieflow``, written
by ``scripts/sync_brieflow.py``). These tests check that:

1. the vendored files equal brieflow's at the pinned commit (after the import rewrite);
2. goudacell's adapters, API and CLI give the same masks and feature tables as calling
   brieflow's functions directly, as brieflow-analysis's phenotype notebook does.

Skipped unless ``BRIEFLOW_LIB`` points at a brieflow git checkout. Test 1 reads the pinned
commit from git, whatever the checkout's state; tests 2 import brieflow's ``lib`` from the
checkout and need it at the pinned commit. The Cellpose tests run on CPU (use a compute
node) on ``GOUDACELL_PARITY_TILE`` (a phenotype image; ``GOUDACELL_PARITY_CHANNELS="3,1"``
sets the DAPI and cytoplasm channels, default last and 1), else on the first image in the
checkout's ``tests/small_test_analysis/small_test_data/phenotype/real_images``, else on a
synthetic tile.

    BRIEFLOW_LIB=/path/to/brieflow pytest tests/test_brieflow_parity.py -v
"""

import contextlib
import importlib.util
import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from skimage.draw import disk
from skimage.measure import regionprops

from goudacell.brieflow import BRIEFLOW_COMMIT

REPO = Path(__file__).resolve().parents[1]
_LIB = os.environ.get("BRIEFLOW_LIB")
ROOT = Path(_LIB).expanduser().resolve() if _LIB else None
pytestmark = pytest.mark.skipif(ROOT is None, reason="BRIEFLOW_LIB not set")


def _checkout_commit():
    out = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return out.stdout.strip()


AT_PIN = ROOT is not None and _checkout_commit() == BRIEFLOW_COMMIT
at_pin = pytest.mark.skipif(
    not AT_PIN, reason=f"brieflow checkout is not at the pinned commit {BRIEFLOW_COMMIT[:7]}"
)


def _sync_module():
    path = REPO / "scripts" / "sync_brieflow.py"
    spec = importlib.util.spec_from_file_location("sync_brieflow", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _brieflow(module):
    """Import a module of brieflow's own ``lib`` from the checkout."""
    workflow = str(ROOT / "workflow")
    if workflow not in sys.path:
        sys.path.insert(0, workflow)
    return importlib.import_module(module)


def _quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return func(*args, **kwargs)


# ---------------------------------------------------------------------------
# 1. Vendored files are brieflow's, byte for byte after the import rewrite
# ---------------------------------------------------------------------------
def test_vendored_files_match_pinned_commit():
    sync = _sync_module()
    expected = sync.vendored_sources(ROOT, BRIEFLOW_COMMIT)
    on_disk = {
        str(p.relative_to(sync.DEST)): p.read_text()
        for p in sync.DEST.rglob("*")
        if p.suffix == ".py" or p.name == "LICENSE"
    }
    assert sorted(on_disk) == sorted(expected), "vendored file set differs; re-run the sync"
    stale = [path for path, text in expected.items() if on_disk[path] != text]
    assert not stale, f"vendored files differ from brieflow {BRIEFLOW_COMMIT[:7]}: {stale}"


def test_check_defaults_to_the_pin(tmp_path):
    """``--check`` compares against the pin even when the checkout's HEAD is elsewhere."""
    sync = _sync_module()
    bare = tmp_path / "brieflow.git"
    subprocess.run(
        ["git", "clone", "-q", "--bare", "--shared", str(ROOT), str(bare)], check=True
    )

    def git(*args, **kwargs):
        return subprocess.run(
            ["git", "-C", str(bare), *args], capture_output=True, text=True, check=True,
            env={**os.environ, "GIT_INDEX_FILE": str(tmp_path / "index"),
                 "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"},
            **kwargs,
        ).stdout.strip()

    # A HEAD one commit past the pin that changes a vendored module
    path = "workflow/lib/shared/log_filter.py"
    text = git("show", f"{BRIEFLOW_COMMIT}:{path}") + "\n# changed after the pin\n"
    git("read-tree", BRIEFLOW_COMMIT)
    blob = git("hash-object", "-w", "--stdin", input=text)
    git("update-index", "--cacheinfo", f"100644,{blob},{path}")
    head = git("commit-tree", git("write-tree"), "-p", BRIEFLOW_COMMIT, "-m", "after pin")
    git("update-ref", "HEAD", head)
    assert sync.resolve_commit(bare) == head != BRIEFLOW_COMMIT

    assert sync.main(["--brieflow", str(bare), "--check"]) == 0
    assert sync.main(["--brieflow", str(bare), "--check", "--ref", "HEAD"]) == 1


def test_only_imports_are_rewritten():
    sync = _sync_module()
    for module in sync.MODULES:
        original = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{BRIEFLOW_COMMIT}:workflow/lib/{module}"],
            capture_output=True, text=True, check=True,
        ).stdout.splitlines()
        vendored = (sync.DEST / module).read_text().splitlines()
        assert len(original) == len(vendored), module
        for a, b in zip(original, vendored):
            if a != b:
                assert b.lstrip().startswith(("from goudacell.brieflow.", "import goudacell.")), b
                assert a.replace(" lib.", " goudacell.brieflow.", 1) == b


# ---------------------------------------------------------------------------
# 2a. Feature extraction and secondary objects on synthetic masks
# ---------------------------------------------------------------------------
def total_second_channel(region):
    return float(region.intensity_image[..., 1].sum())


def first_channel_spread(region):
    import numpy as np

    return float(np.std(region.intensity_image[..., 0]))


CUSTOM_FEATURES = [
    total_second_channel,
    (first_channel_spread, "cell"),
    (total_second_channel, "cytoplasm"),
]
CHANNEL_NAMES = ["DAPI", "TUB", "FOCI", "VAC"]


def _tile(seed=0, size=192):
    """A synthetic 4-channel tile with reconciled nuclei/cell masks.

    Channels: DAPI (bright nuclei), TUB (cells), FOCI (puncta), VAC (secondary objects).
    One cell has two nuclei and one nucleus reaches into its neighbour.
    """
    rng = np.random.default_rng(seed)
    nuclei = np.zeros((size, size), int)
    cells = np.zeros((size, size), int)
    label = 0
    for y in range(24, size - 12, 36):
        for x in range(24, size - 12, 36):
            label += 1
            cells[disk((y, x), 17, shape=cells.shape)] = label
            centre = (y + rng.integers(-3, 4), x + rng.integers(-3, 4))
            nuclei[disk(centre, 7, shape=cells.shape)] = label
    nuclei[disk((24, 36), 5, shape=cells.shape)] = 1
    nuclei[disk((60, 66), 3, shape=cells.shape)] = 5

    image = rng.poisson(40, (4, size, size)).astype(np.uint16)
    image[0][nuclei > 0] += 900
    image[1][cells > 0] += 250
    for _ in range(120):
        image[2][disk(tuple(rng.integers(4, size - 4, 2)), 2, shape=cells.shape)] += 1800
    for _ in range(40):
        radius = rng.integers(3, 7)
        image[3][disk(tuple(rng.integers(8, size - 8, 2)), radius, shape=cells.shape)] += 1500
    return image, nuclei, cells


@pytest.fixture(scope="module")
def tile():
    return _tile()


@at_pin
@pytest.mark.parametrize("with_cells", [True, False])
def test_cp_emulator_features(tile, with_cells):
    bemu = _brieflow("lib.phenotype.extract_phenotype_cp_emulator")
    bcf = _brieflow("lib.phenotype.custom_features")
    bcyto = _brieflow("lib.phenotype.identify_cytoplasm_cellpose")
    from goudacell.brieflow.phenotype.custom_features import load_custom_features
    from goudacell.features import extract_features

    image, nuclei, cells = tile
    custom = [total_second_channel] + (CUSTOM_FEATURES[1:] if with_cells else [])
    definitions = bcf.register_custom_features(custom)
    cells_in = cells if with_cells else None
    cytoplasms = _quiet(bcyto.identify_cytoplasm_cellpose, nuclei, cells) if with_cells else None

    theirs = bemu.extract_phenotype_cp_emulator(
        image, nuclei, cells_in, wildcards={}, cytoplasms=cytoplasms, foci_channel=[2],
        channel_names=CHANNEL_NAMES, custom_features=bcf.load_custom_features(definitions),
    )
    ours = _quiet(
        extract_features, image, nuclei, cells_in, channel_names=CHANNEL_NAMES,
        foci_channel=[2], custom_features=load_custom_features(definitions),
    )
    pd.testing.assert_frame_equal(ours, theirs)


@at_pin
def test_feature_selection_is_a_column_subset(tile):
    """Channel, compartment and block options select from brieflow's own table."""
    from goudacell.features import extract_features

    image, nuclei, cells = tile
    selection = dict(channel_names=["DAPI", "TUB"], channels=[0, 1])
    full = _quiet(extract_features, image, nuclei, cells, **selection)
    subset = _quiet(
        extract_features, image, nuclei, cells, compartments=["nucleus", "cytoplasm"],
        include_texture=False, include_correlation=False, include_neighbors=False, **selection,
    )
    assert set(subset.columns) < set(full.columns)
    assert not any(c.startswith("cell_") for c in subset.columns)
    assert not any("pftas" in c or "haralick" in c or "correlation" in c for c in subset.columns)
    assert not any("neighbor" in c or "FOCI" in c or "VAC" in c for c in subset.columns)
    pd.testing.assert_frame_equal(subset, full[subset.columns])


@at_pin
def test_cp_measure_features(tile):
    pytest.importorskip("cp_measure")
    bcm = _brieflow("lib.phenotype.extract_phenotype_cp_measure")
    bcyto = _brieflow("lib.phenotype.identify_cytoplasm_cellpose")
    from goudacell.features import extract_features

    image, nuclei, cells = tile
    theirs = _quiet(
        bcm.extract_phenotype_cp_measure, image[:2], nuclei, cells,
        cytoplasms=_quiet(bcyto.identify_cytoplasm_cellpose, nuclei, cells),
        channel_names=CHANNEL_NAMES[:2],
    )
    ours = _quiet(
        extract_features, image, nuclei, cells, channel_names=CHANNEL_NAMES[:2],
        channels=[0, 1], method="cp_measure",
    )
    pd.testing.assert_frame_equal(ours, theirs)


SECOND_OBJ_CASES = [
    {},
    {"size_filter_method": "area", "second_obj_min_size": 20, "second_obj_max_size": 400},
    {"declump_method": "intensity", "declump_mode": "propagate", "fill_holes": "declump"},
    {"threshold_method": "min_cross_entropy", "use_shape_refinement": True},
]


@at_pin
@pytest.mark.parametrize("overrides", SECOND_OBJ_CASES)
def test_secondary_objects(tile, overrides):
    bso = _brieflow("lib.phenotype.segment_secondary_object")
    bfeat = _brieflow("lib.phenotype.extract_phenotype_second_objs")
    bcyto = _brieflow("lib.phenotype.identify_cytoplasm_cellpose")
    from goudacell.config import FeatureExtractionParams, SecondaryObjectParams
    from goudacell.features import extract_second_obj_features
    from goudacell.segment import segment_second_objects

    image, nuclei, cells = tile
    params = SecondaryObjectParams(
        second_obj_detection=True, second_obj_channel_index=3, **overrides
    )
    # The phenotype notebook: nucleus centroids from the nuclei mask
    centroids = {r.label: r.centroid for r in regionprops(nuclei)}
    theirs = _quiet(
        bso.segment_second_objs_from_config, image, cells,
        _quiet(bcyto.identify_cytoplasm_cellpose, nuclei, cells),
        params.to_brieflow_params(gpu=False), nuclei_centroids=centroids,
    )
    ours = _quiet(segment_second_objects, image, nuclei, cells, params, False)
    np.testing.assert_array_equal(ours[0], theirs[0])
    np.testing.assert_array_equal(ours[2], theirs[2])
    for key in ("cell_summary", "second_obj_cell_mapping"):
        pd.testing.assert_frame_equal(ours[1][key], theirs[1][key])
    assert len(np.unique(ours[0])) > 5

    fe = FeatureExtractionParams(enabled=True, channel_names=CHANNEL_NAMES, foci_channel=2)
    their_features = _quiet(
        bfeat.extract_phenotype_second_objs, image, second_objs=theirs[0], wildcards={},
        second_obj_cell_mapping_df=theirs[1]["second_obj_cell_mapping"], foci_channel=2,
        channel_names=CHANNEL_NAMES,
    )
    our_features = _quiet(extract_second_obj_features, fe, image, ours[0], ours[1])
    pd.testing.assert_frame_equal(our_features, their_features)


# ---------------------------------------------------------------------------
# 2b. Cellpose segmentation, the API and the CLI end to end on one tile
# ---------------------------------------------------------------------------
def _real_tile():
    path = os.environ.get("GOUDACELL_PARITY_TILE")
    if path is None:
        images = ROOT / "tests/small_test_analysis/small_test_data/phenotype/real_images"
        found = sorted(images.glob("*.nd2")) if images.is_dir() else []
        path = found[0] if found else None
    if path is None:
        return None
    from goudacell.io import load_image

    image = load_image(Path(path), channel=None, z_project=True)
    h, w = image.shape[-2:]
    image = image[:, h // 2 - 256 : h // 2 + 256, w // 2 - 256 : w // 2 + 256]
    channels = os.environ.get("GOUDACELL_PARITY_CHANNELS")
    dapi, cyto = map(int, channels.split(",")) if channels else (image.shape[0] - 1, 1)
    return np.ascontiguousarray(image), dapi, cyto


@pytest.fixture(scope="module")
def cellpose_tile():
    """(image, dapi_index, cyto_index): a real phenotype crop, else a smooth synthetic tile."""
    pytest.importorskip("cellpose")
    real = _real_tile()
    if real is not None:
        return real
    from scipy import ndimage

    image, _, _ = _tile(seed=3, size=224)
    smooth = [ndimage.gaussian_filter(image[c].astype(float), 1.5) for c in range(4)]
    return np.stack(smooth).astype(np.uint16), 0, 1


def _cellpose_4x():
    return _brieflow("lib.shared.segment_cellpose").CELLPOSE_4X


# The phenotype notebook's defaults
THRESHOLDS = dict(
    nuclei_flow_threshold=0.4,
    nuclei_cellprob_threshold=0.0,
    cell_flow_threshold=1,
    cell_cellprob_threshold=0,
)


@pytest.fixture(scope="module")
def diameters(cellpose_tile):
    image, dapi, cyto = cellpose_tile
    if _cellpose_4x():
        return 20.0, 45.0
    bsc = _brieflow("lib.shared.segment_cellpose")
    return _quiet(
        bsc.estimate_diameters, image, dapi_index=dapi, cyto_index=cyto, cellpose_model="cyto3"
    )


@pytest.fixture(scope="module")
def brieflow_masks(cellpose_tile, diameters):
    """The phenotype notebook's segmentation cell, run with brieflow's own lib."""
    bsc = _brieflow("lib.shared.segment_cellpose")
    image, dapi, cyto = cellpose_tile
    return _quiet(
        bsc.segment_cellpose, image, dapi_index=dapi, cyto_index=cyto,
        nuclei_diameter=diameters[0], cell_diameter=diameters[1],
        cellpose_kwargs=dict(THRESHOLDS), cellpose_model="cpsam" if _cellpose_4x() else "cyto3",
        helper_index=None, gpu=False, reconcile="contained_in_cells", cells=True,
        return_counts=True,
    )


@at_pin
def test_estimate_diameters(cellpose_tile, diameters):
    if _cellpose_4x():
        pytest.skip("Cellpose 4.x has no size model")
    from goudacell.segment import estimate_diameters

    image, dapi, cyto = cellpose_tile
    ours = _quiet(estimate_diameters, image, dapi, cyto, cell_model="cyto3", gpu=False)
    assert ours == diameters


@at_pin
def test_dual_segmentation(cellpose_tile, diameters, brieflow_masks):
    from goudacell.segment import segment_nuclei_and_cells

    image, dapi, cyto = cellpose_tile
    nuclei, cells, nuclei_per_cell = _quiet(
        segment_nuclei_and_cells, image, nuclei_channel=dapi, cyto_channel=cyto,
        nuclei_diameter=diameters[0], cell_diameter=diameters[1],
        cell_model="cpsam" if _cellpose_4x() else "cyto3", gpu=False,
        reconcile="contained_in_cells", return_nuclei_per_cell=True, **THRESHOLDS,
    )
    np.testing.assert_array_equal(nuclei, brieflow_masks[0])
    np.testing.assert_array_equal(cells, brieflow_masks[1])
    assert nuclei_per_cell == brieflow_masks[3]
    assert len(np.unique(cells)) > 4


@at_pin
def test_nuclei_only_segmentation(cellpose_tile, diameters):
    bsc = _brieflow("lib.shared.segment_cellpose")
    from goudacell.segment import brieflow_nuclei_model, segment_nuclei

    image, dapi, cyto = cellpose_tile
    model = brieflow_nuclei_model()
    theirs = _quiet(
        bsc.segment_cellpose, image, dapi_index=dapi, cyto_index=cyto,
        nuclei_diameter=diameters[0], cell_diameter=None, cellpose_model=model,
        cellpose_kwargs=dict(nuclei_flow_threshold=0.4, nuclei_cellprob_threshold=0.0),
        cells=False, gpu=False,
    )
    ours = _quiet(segment_nuclei, image, dapi, diameters[0], model=model, gpu=False)
    np.testing.assert_array_equal(ours, theirs)
    assert len(np.unique(ours)) > 4


@at_pin
def test_cli_matches_brieflow(cellpose_tile, diameters, brieflow_masks, tmp_path):
    """Config -> ``goudacell segment`` -> masks and features equal brieflow's."""
    import tifffile

    CliRunner = pytest.importorskip("typer.testing").CliRunner

    from goudacell.cli import app
    from goudacell.config import (
        DualSegmentationParams,
        FeatureExtractionParams,
        SegmentationConfig,
    )
    from goudacell.segment import brieflow_nuclei_model

    bemu = _brieflow("lib.phenotype.extract_phenotype_cp_emulator")
    bcyto = _brieflow("lib.phenotype.identify_cytoplasm_cellpose")
    image, dapi, cyto = cellpose_tile
    names = [f"ch{i}" for i in range(image.shape[0])]
    (tmp_path / "in").mkdir()
    tifffile.imwrite(tmp_path / "in" / "tile.tif", image)

    config = SegmentationConfig(
        input_dir=str(tmp_path / "in"), output_dir=str(tmp_path / "out"),
        mode="dual", gpu=False, reconcile="contained_in_cells",
        dual=DualSegmentationParams(
            nuclei_channel=dapi, cyto_channel=cyto, nuclei_diameter=diameters[0],
            cell_diameter=diameters[1], cell_model="cpsam" if _cellpose_4x() else "cyto3",
            nuclei_model=brieflow_nuclei_model(), **THRESHOLDS,
        ),
        feature_extraction=FeatureExtractionParams(
            enabled=True, channel_names=names, output_path="{stem}_features.csv"
        ),
    )
    config.to_yaml(tmp_path / "config.yaml")
    result = CliRunner().invoke(app, ["segment", str(tmp_path / "config.yaml")])
    assert result.exit_code == 0, result.output

    nuclei, cells = brieflow_masks[0], brieflow_masks[1]
    np.testing.assert_array_equal(tifffile.imread(tmp_path / "out/tile_nuclei_mask.tif"), nuclei)
    np.testing.assert_array_equal(tifffile.imread(tmp_path / "out/tile_cell_mask.tif"), cells)

    # brieflow's extract_phenotype.py: the cp_emulator table plus num_nuclei
    theirs = _quiet(
        bemu.extract_phenotype_cp_emulator, image, nuclei, cells, wildcards={},
        cytoplasms=_quiet(bcyto.identify_cytoplasm_cellpose, nuclei, cells),
        channel_names=names,
    )
    counts = pd.Series(brieflow_masks[3], dtype=float)
    theirs["num_nuclei"] = theirs["label"].map(counts).fillna(1).astype(int)
    theirs.to_csv(tmp_path / "theirs.csv", index=False)
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "out/tile_features.csv"), pd.read_csv(tmp_path / "theirs.csv")
    )
