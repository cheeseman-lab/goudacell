"""Parity of goudacell's phenotype code with brieflow's.

Runs brieflow's and goudacell's segmentation post-processing, secondary-object detection
and feature extraction on the same synthetic inputs and asserts identical outputs.
Skipped unless ``BRIEFLOW_LIB`` points at a brieflow checkout (its root, ``workflow/`` or
``workflow/lib``); see CLAUDE.md "Brieflow parity" for the pinned commit.

    BRIEFLOW_LIB=/path/to/brieflow pytest tests/test_brieflow_parity.py -v
"""

import contextlib
import hashlib
import importlib
import io
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from skimage.draw import disk

# brieflow commit goudacell is in parity with (zarr3), and each mapped file's sha256 prefix.
BRIEFLOW_COMMIT = "6beb71a531022e117064e814e80998dd8f465b5f"
BRIEFLOW_SOURCES = {
    "lib/external/cp_emulator.py": "eaacc7911d08f166",
    "lib/phenotype/constants.py": "601b23fb779e69af",
    "lib/phenotype/custom_features.py": "674415e9c179482e",
    "lib/phenotype/extract_phenotype_cp_emulator.py": "22743fddf4133150",
    "lib/phenotype/extract_phenotype_cp_measure.py": "308f88e5e80f8cd4",
    "lib/phenotype/extract_phenotype_second_objs.py": "e9d0a6ab022f0f7a",
    "lib/phenotype/identify_cytoplasm_cellpose.py": "23d275f8e05f43ba",
    "lib/phenotype/segment_secondary_object.py": "72c542764c8d6111",
    "lib/shared/feature_extraction.py": "80ce2d4be8177e72",
    "lib/shared/feature_table_utils.py": "ca807bd48fd53444",
    "lib/shared/log_filter.py": "4880db17651e0d0e",
    "lib/shared/segment_cellpose.py": "7c917d79d43765c6",
    "lib/shared/segmentation_utils.py": "8e0e4fd1aab91192",
    "scripts/phenotype/extract_phenotype.py": "20d3209975ddf75a",
    "scripts/phenotype/identify_second_objs.py": "53c75e9b25acfcb2",
    "scripts/phenotype/merge_second_objs_phenotype_cp.py": "36d1a6f488e2391d",
    "scripts/shared/segment.py": "bc9d05c67b8b56d3",
}


def _workflow_dir():
    root = os.environ.get("BRIEFLOW_LIB")
    if not root:
        return None
    root = Path(root).expanduser().resolve()
    for candidate in (root / "workflow", root, root.parent):
        if (candidate / "lib" / "phenotype").is_dir():
            return candidate
    raise RuntimeError(f"BRIEFLOW_LIB={root} does not contain workflow/lib/phenotype")


WORKFLOW = _workflow_dir()
pytestmark = pytest.mark.skipif(WORKFLOW is None, reason="BRIEFLOW_LIB not set")


def _import_brieflow(module):
    """Import a brieflow ``lib`` module, stubbing what only its plotting helpers need."""
    if str(WORKFLOW) not in sys.path:
        sys.path.insert(0, str(WORKFLOW))
    if "microfilm" not in sys.modules and importlib.util.find_spec("microfilm") is None:
        microplot = types.ModuleType("microfilm.microplot")
        microplot.Microimage = microplot.Micropanel = object
        sys.modules["microfilm"] = types.ModuleType("microfilm")
        sys.modules["microfilm.microplot"] = microplot
    try:
        importlib.import_module("lib.shared.configuration_utils")
    except ImportError:
        plotting = types.ModuleType("lib.shared.configuration_utils")
        plotting.create_micropanel = plotting.random_cmap = None
        sys.modules["lib.shared.configuration_utils"] = plotting
    return importlib.import_module(module)


def _quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


# Custom features: module-level and self-contained, as register_custom_features requires.
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


def _raw_masks(seed, size=160):
    """Independently labelled nuclei and cells, with the overlaps reconcile must resolve."""
    rng = np.random.default_rng(seed)
    nuclei = np.zeros((size, size), int)
    cells = np.zeros((size, size), int)
    for k in range(1, 30):
        centre = rng.integers(6, size - 6, 2)
        cells[disk(tuple(centre), rng.integers(7, 14), shape=cells.shape)] = k
        offset = centre + rng.integers(-5, 6, 2)
        nucleus_label = k + 7 * rng.integers(0, 3)
        nuclei[disk(tuple(offset), rng.integers(2, 6), shape=cells.shape)] = nucleus_label
    return nuclei, cells


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
    nuclei[disk((24, 36), 5, shape=cells.shape)] = 1  # reaches into cell 2
    nuclei[disk((60, 66), 3, shape=cells.shape)] = 5  # second nucleus of cell 5

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


def test_brieflow_sources_unchanged():
    """Fail when a mapped brieflow file changes, so the change gets reviewed and ported."""
    changed = []
    for rel, expected in BRIEFLOW_SOURCES.items():
        path = WORKFLOW / rel
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else None
        if digest != expected:
            changed.append(rel)
    assert not changed, (
        f"brieflow files changed since parity commit {BRIEFLOW_COMMIT[:7]}: {changed}. "
        "Diff them against that commit, port what changes masks/features/parameters, "
        "then update BRIEFLOW_COMMIT and the hashes (see CLAUDE.md 'Brieflow parity')."
    )


@pytest.mark.parametrize("helper_index", [None, 2, 3])
def test_prepare_cellpose(tile, helper_index):
    pytest.importorskip("cellpose")
    bsc = _import_brieflow("lib.shared.segment_cellpose")
    from goudacell.segment import image_log_scale, prepare_cellpose

    image = tile[0].copy()
    image[3] = 0
    ours = prepare_cellpose(image, 0, 1, helper_index=helper_index)
    theirs = bsc.prepare_cellpose(image, 0, 1, helper_index=helper_index)
    assert ours.dtype == theirs.dtype
    np.testing.assert_array_equal(ours, theirs)
    np.testing.assert_array_equal(
        prepare_cellpose(image, 0, 1, logscale=False),
        bsc.prepare_cellpose(image, 0, 1, logscale=False),
    )
    bsu = _import_brieflow("lib.shared.segmentation_utils")
    np.testing.assert_array_equal(image_log_scale(image[1]), bsu.image_log_scale(image[1]))


@pytest.mark.parametrize("how", ["consensus", "contained_in_cells"])
@pytest.mark.parametrize("seed", range(8))
def test_reconcile_nuclei_per_cell_cytoplasm(seed, how):
    bsu = _import_brieflow("lib.shared.segmentation_utils")
    bcyto = _import_brieflow("lib.phenotype.identify_cytoplasm_cellpose")
    from goudacell.segment import count_nuclei_per_cell, identify_cytoplasm, reconcile_nuclei_cells

    nuclei, cells = _raw_masks(seed)
    ours = reconcile_nuclei_cells(nuclei.copy(), cells.copy(), how=how)
    theirs = _quiet(bsu.reconcile_nuclei_cells, nuclei.copy(), cells.copy(), how=how)
    for a, b in zip(ours, theirs):
        np.testing.assert_array_equal(a, b)
    assert count_nuclei_per_cell(nuclei, ours[1]) == bsu.count_nuclei_per_cell(nuclei, theirs[1])

    np.testing.assert_array_equal(
        identify_cytoplasm(*ours), _quiet(bcyto.identify_cytoplasm_cellpose, *theirs)
    )
    raw_theirs = _quiet(bcyto.identify_cytoplasm_cellpose, nuclei, cells)
    raw_ours = identify_cytoplasm(nuclei, cells)
    assert (raw_ours is None) == (raw_theirs is None)
    if raw_ours is not None:
        np.testing.assert_array_equal(raw_ours, raw_theirs)


def test_identify_cytoplasm_overlapping_nuclei(tile):
    bcyto = _import_brieflow("lib.phenotype.identify_cytoplasm_cellpose")
    from goudacell.segment import identify_cytoplasm

    _, nuclei, cells = tile
    np.testing.assert_array_equal(
        identify_cytoplasm(nuclei, cells), _quiet(bcyto.identify_cytoplasm_cellpose, nuclei, cells)
    )


def test_custom_feature_registration():
    bcf = _import_brieflow("lib.phenotype.custom_features")
    from goudacell.custom_features import register_custom_features

    assert register_custom_features(CUSTOM_FEATURES) == bcf.register_custom_features(
        CUSTOM_FEATURES
    )


@pytest.mark.parametrize("with_cells", [True, False])
def test_cp_emulator_features(tile, with_cells):
    bemu = _import_brieflow("lib.phenotype.extract_phenotype_cp_emulator")
    bcf = _import_brieflow("lib.phenotype.custom_features")
    bcyto = _import_brieflow("lib.phenotype.identify_cytoplasm_cellpose")
    from goudacell.custom_features import load_custom_features
    from goudacell.features import extract_features

    image, nuclei, cells = tile
    custom = [total_second_channel] + (CUSTOM_FEATURES[1:] if with_cells else [])
    definitions = bcf.register_custom_features(custom)
    cells_in = cells if with_cells else None
    cytoplasms = _quiet(bcyto.identify_cytoplasm_cellpose, nuclei, cells) if with_cells else None

    theirs = bemu.extract_phenotype_cp_emulator(
        image,
        nuclei,
        cells_in,
        wildcards={},
        cytoplasms=cytoplasms,
        foci_channel=[2],
        channel_names=CHANNEL_NAMES,
        custom_features=bcf.load_custom_features(definitions),
    )
    ours = extract_features(
        image,
        nuclei,
        cells_in,
        channel_names=CHANNEL_NAMES,
        foci_channel=[2],
        custom_features=load_custom_features(definitions),
    )
    assert list(ours.columns) == list(theirs.columns)
    pd.testing.assert_frame_equal(ours, theirs, check_dtype=False)


def test_num_nuclei_column(tile):
    from goudacell.features import add_num_nuclei

    df = pd.DataFrame({"label": [1, 2, 3]})
    counts = {1: 2, 3: 1}
    # brieflow extract_phenotype.py: map the per-cell count, default 1
    expected = df["label"].map(pd.Series(counts)).fillna(1).astype(int)
    pd.testing.assert_series_equal(
        add_num_nuclei(df.copy(), counts)["num_nuclei"], expected, check_names=False
    )


SECOND_OBJ_CASES = [
    {},
    {"size_filter_method": "area", "second_obj_min_size": 20, "second_obj_max_size": 400},
    {"declump_method": "intensity", "declump_mode": "propagate", "fill_holes": "declump"},
    {"threshold_method": "min_cross_entropy", "use_shape_refinement": True},
    {"maxima_reduction_factor": 0.2, "use_morphological_opening": False},
]


@pytest.mark.parametrize("overrides", SECOND_OBJ_CASES)
def test_secondary_objects_threshold(tile, overrides):
    bso = _import_brieflow("lib.phenotype.segment_secondary_object")
    bfeat = _import_brieflow("lib.phenotype.extract_phenotype_second_objs")
    from goudacell.config import SecondaryObjectParams
    from goudacell.features_second_objs import extract_phenotype_second_objs
    from goudacell.secondary_objects import segment_second_objs_from_config
    from goudacell.segment import identify_cytoplasm

    image, nuclei, cells = tile
    cytoplasms = identify_cytoplasm(nuclei, cells)
    centroids = {i: (float(i * 7), float(i * 5)) for i in range(1, 9)}
    params = SecondaryObjectParams(
        second_obj_detection=True, second_obj_channel_index=3, **overrides
    ).to_brieflow_params(gpu=False)

    args = (image, cells, cytoplasms, params, centroids)
    ours = _quiet(segment_second_objs_from_config, *args)
    theirs = _quiet(bso.segment_second_objs_from_config, *args)
    np.testing.assert_array_equal(ours[0], theirs[0])
    np.testing.assert_array_equal(ours[2], theirs[2])
    for key in ("cell_summary", "second_obj_cell_mapping"):
        pd.testing.assert_frame_equal(ours[1][key], theirs[1][key])

    if np.any(ours[0]):
        kwargs = dict(
            second_obj_cell_mapping_df=theirs[1]["second_obj_cell_mapping"],
            foci_channel=2,
            channel_names=CHANNEL_NAMES,
        )
        pd.testing.assert_frame_equal(
            extract_phenotype_second_objs(image, ours[0], {}, **kwargs),
            _quiet(bfeat.extract_phenotype_second_objs, image, theirs[0], {}, **kwargs),
        )


def test_secondary_objects_detected(tile):
    """Guard against the threshold case passing trivially on an empty result."""
    from goudacell.config import SecondaryObjectParams
    from goudacell.secondary_objects import segment_second_objs_from_config
    from goudacell.segment import identify_cytoplasm

    image, nuclei, cells = tile
    params = SecondaryObjectParams(second_obj_detection=True, second_obj_channel_index=3)
    masks, _, _ = _quiet(
        segment_second_objs_from_config,
        image,
        cells,
        identify_cytoplasm(nuclei, cells),
        params.to_brieflow_params(gpu=False),
    )
    assert len(np.unique(masks)) > 5


def test_cp_measure_features(tile):
    pytest.importorskip("cp_measure")
    bcm = _import_brieflow("lib.phenotype.extract_phenotype_cp_measure")
    bcyto = _import_brieflow("lib.phenotype.identify_cytoplasm_cellpose")
    from goudacell.features import extract_features

    image, nuclei, cells = tile
    image = image[:2]
    theirs = _quiet(
        bcm.extract_phenotype_cp_measure,
        image,
        nuclei,
        cells,
        cytoplasms=_quiet(bcyto.identify_cytoplasm_cellpose, nuclei, cells),
        channel_names=CHANNEL_NAMES[:2],
    )
    ours = _quiet(
        extract_features,
        image,
        nuclei,
        cells,
        channel_names=CHANNEL_NAMES[:2],
        method="cp_measure",
    )
    assert list(ours.columns) == list(theirs.columns)
    pd.testing.assert_frame_equal(ours, theirs, check_dtype=False)


@pytest.fixture(scope="module")
def cellpose_tile():
    """A small tile Cellpose can segment on CPU: smooth nuclei and cells."""
    pytest.importorskip("cellpose")
    from scipy import ndimage

    image, nuclei, cells = _tile(seed=3, size=224)
    smooth = np.stack(
        [ndimage.gaussian_filter(image[c].astype(float), 1.5) for c in range(4)]
    ).astype(np.uint16)
    return smooth


@pytest.mark.parametrize("reconcile", ["contained_in_cells", "consensus"])
def test_cellpose_dual_segmentation(cellpose_tile, reconcile):
    bsc = _import_brieflow("lib.shared.segment_cellpose")
    if bsc.CELLPOSE_4X:
        pytest.skip("brieflow's nuclei model is fixed to cpsam on Cellpose 4.x")
    from goudacell.segment import segment_nuclei_and_cells

    thresholds = dict(
        nuclei_flow_threshold=0.4,
        nuclei_cellprob_threshold=0.0,
        cell_flow_threshold=1.0,
        cell_cellprob_threshold=0.0,
    )
    theirs = _quiet(
        bsc.segment_cellpose,
        cellpose_tile,
        dapi_index=0,
        cyto_index=1,
        nuclei_diameter=14,
        cell_diameter=34,
        cellpose_model="cyto3",
        helper_index=2,
        cellpose_kwargs=dict(thresholds),
        reconcile=reconcile,
        return_counts=True,
        gpu=False,
    )
    ours = segment_nuclei_and_cells(
        cellpose_tile,
        nuclei_channel=0,
        cyto_channel=1,
        nuclei_diameter=14,
        cell_diameter=34,
        cell_model="cyto3",
        nuclei_model="nuclei",
        gpu=False,
        reconcile=reconcile,
        helper_channel=2,
        return_nuclei_per_cell=True,
        **thresholds,
    )
    np.testing.assert_array_equal(ours[0], theirs[0])
    np.testing.assert_array_equal(ours[1], theirs[1])
    assert ours[2] == theirs[3]
    assert len(np.unique(ours[1])) > 4


def test_cellpose_nuclei_only_and_diameters(cellpose_tile):
    bsc = _import_brieflow("lib.shared.segment_cellpose")
    if bsc.CELLPOSE_4X:
        pytest.skip("diameter estimation needs Cellpose 3.x")
    from goudacell.segment import estimate_diameters, segment_nuclei

    theirs = _quiet(
        bsc.segment_cellpose,
        cellpose_tile,
        dapi_index=0,
        cyto_index=1,
        nuclei_diameter=14,
        cell_diameter=34,
        cellpose_model="nuclei",
        cellpose_kwargs=dict(flow_threshold=0.4, cellprob_threshold=0),
        cells=False,
        gpu=False,
    )
    ours = segment_nuclei(cellpose_tile, 0, 14, model="nuclei", gpu=False)
    np.testing.assert_array_equal(ours, theirs)
    assert len(np.unique(ours)) > 4

    assert estimate_diameters(cellpose_tile, 0, 1, cell_model="cyto3", gpu=False) == _quiet(
        bsc.estimate_diameters, cellpose_tile, dapi_index=0, cyto_index=1, cellpose_model="cyto3"
    )
