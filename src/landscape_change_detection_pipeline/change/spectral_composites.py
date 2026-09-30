"""Per-scene/per-month spectral-index computation, for change-detection analysis.

Purpose
-------
:mod:`landscape_change_detection_pipeline.inference.composites` (Stage 7) reduces
per-scene *classification* rasters to one categorical class map per
tile-month. NDVI regrowth, dNBR burn severity, and the other index-based
change layers need the underlying *continuous reflectance*, not the classes
derived from it -- that information no longer exists once Stage 7 has voted
it down to a single class id per pixel. This module re-reads the same raw
per-scene reflectance Stage 7's classifier consumed
(``data/tiles/<tile_id>.zarr/<sensor>/toa``), computes every spectral index
of interest per scene, and reduces each index to one tile-month composite --
the same tile/month granularity as Stage 7, built in parallel from a
different source, never derived from Stage 7's own output.

This module provides the computation only -- it does not persist a
per-tile-month store on disk. An earlier revision did (one ``indices.npz``
per tile-month, every index x 4 statistics, unconditionally); at real-AOI
scale that reached hundreds of GB for a superset the only real consumer
(:mod:`change.regrowth_severity`) never read more than two indices of.
Callers that need a month's composite now call :func:`compute_scene_indices`
+ :func:`build_month_index_composite` directly, on demand (see
``change.regrowth_severity.build_month_composite_cache`` for the standard
memoizing wrapper).

Cloud/shadow exclusion
------------------------
A pixel flagged ``cloud`` or ``shadow`` by the trained classifier
(``class_config.ClassDef.change_eligible=False``) for a given scene must not
contribute to that pixel's index composite for that scene -- an imaging
artifact, not a real reflectance measurement of the land surface. Since the
raw reflectance array and that scene's own classification
(``outputs/inference/<tile_id>/<sensor>/<scene_id>/class_map.npz``) are
computed from the exact same pixel grid (confirmed on real pilot data: same
transform, same shape, same CRS -- Stage 6 classifies the untouched fetched
grid, it is never reprojected first), the two can be masked together
directly with no alignment step.

Reduction statistics
----------------------
For each index and tile-month, four per-pixel statistics are kept across
that month's valid (non-cloud/shadow) scene observations:

- ``median`` -- the robust central tendency; the general-purpose trend
  value for any index.
- ``min`` -- the lowest value observed that month. For NBR specifically,
  this is the pixel's most severe burn signal that month (used directly by
  dNBR, which needs the worst observed state, not an average that would
  dilute a single severe-fire scene against several unaffected ones).
- ``max`` -- the highest value observed that month. For NDVI, the pixel's
  peak vigour that month (a regrowth signal a median could understate if
  most of the month's scenes were early-season); for NDSI, effectively an
  "any snow observed this month" flag, since one snow-covered scene should
  not be voted out by several snow-free scenes in the same median the way
  it would be under Stage 7's own ``median`` class-reduction rule.
- ``n_obs`` -- how many valid (non-cloud/shadow) scene observations
  contributed to this pixel this month, so a median/min/max resting on a
  single scene can be told apart from one resting on a dozen.

All four are computed unconditionally for whichever indices are requested
(dNBR only reads NBR's ``median``, NDVI regrowth only reads NDVI's
``median``) -- cheap since nothing is persisted, and keeps
:func:`build_month_index_composite` usable as-is for any future caller that
does need ``min``/``max``/``n_obs``.

Indices computed
-------------------
Every index in :mod:`features.spectral_indices` (the same eight used by the
land-cover model) plus the two analysis-only indices in
:mod:`change.analysis_indices` (Tasseled Cap Brightness/Greenness/Wetness,
McFeeters' NDWI) -- computed here independently of the model's own feature
pipeline, never read from or written back into it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from landscape_change_detection_pipeline.change.analysis_indices import tasseled_cap
from landscape_change_detection_pipeline.classes.class_config import ClassConfig
from landscape_change_detection_pipeline.features.spectral_indices import INDEX_NAMES, compute_indices_dict
from landscape_change_detection_pipeline.features.training_cache import bands_for_scene, scene_year
from landscape_change_detection_pipeline.inference.composites import scene_month
from landscape_change_detection_pipeline.inference.engine import read_class_map, scene_output_paths
from landscape_change_detection_pipeline.scenes.zarr_store import read_sensor_group_attrs

#: Every index this module computes per scene: the model's own eight
#: band-pair/chromaticity indices, plus the three Tasseled Cap components and
#: McFeeters' NDWI (analysis-only, see change/analysis_indices.py).
ALL_INDEX_NAMES: tuple[str, ...] = (*INDEX_NAMES, "TC_BRIGHTNESS", "TC_GREENNESS", "TC_WETNESS")

_TC_LABELS = {"TC_BRIGHTNESS": "brightness", "TC_GREENNESS": "greenness", "TC_WETNESS": "wetness"}

SENSORS: tuple[str, ...] = ("l5", "l7", "l8", "l9", "s2")

#: Reduction statistics kept per index per tile-month (see module docstring).
STATS: tuple[str, ...] = ("median", "min", "max", "n_obs")


@dataclass(frozen=True)
class SceneIndices:
    sensor: str
    scene_id: str
    year: int
    month: int
    indices: dict[str, np.ndarray]  # name -> (H, W) float32
    valid_mask: np.ndarray  # (H, W) bool -- True where not cloud/shadow
    transform: tuple[float, ...]
    crs_wkt: str


def indices_from_bands(
    bands: dict[str, np.ndarray],
    provenance: dict[str, str],
    sensor: str,
    index_names: tuple[str, ...] = ALL_INDEX_NAMES,
) -> dict[str, np.ndarray]:
    """Every requested index (of :data:`ALL_INDEX_NAMES`) computed from one
    scene's reflectance bands (``{label: array}``, labels blue/green/red/nir/
    swir1/swir2). Works on whole scenes or on any sub-window of one."""
    indices: dict[str, np.ndarray] = {}
    normalized_difference_and_chroma = {n for n in index_names if n in INDEX_NAMES}
    if normalized_difference_and_chroma:
        computed = compute_indices_dict(bands, provenance, names=tuple(normalized_difference_and_chroma))
        for name, (array, _provenance) in computed.items():
            indices[name] = array

    tc_names = {n for n in index_names if n in _TC_LABELS}
    if tc_names:
        tc = tasseled_cap(bands, sensor.upper())
        for name in tc_names:
            indices[name] = tc[_TC_LABELS[name]]
    return indices


def compute_scene_indices(
    tile_dir: str | Path,
    inference_root: str | Path,
    tile_id: str,
    sensor: str,
    scene_id: str,
    class_config: ClassConfig,
    index_names: tuple[str, ...] = ALL_INDEX_NAMES,
) -> SceneIndices:
    """Every requested index for one scene, plus its cloud/shadow validity
    mask from that scene's own classification.

    Raises ``FileNotFoundError`` if this scene has not been classified yet
    (Stage 6) -- a scene with no classification has no cloud/shadow mask to
    exclude, so it cannot safely contribute to an index composite.
    """
    bands, provenance = bands_for_scene(tile_dir, tile_id, sensor.upper(), scene_id)
    attrs = read_sensor_group_attrs(tile_dir, tile_id, sensor.upper())

    class_map_path = scene_output_paths(inference_root, tile_id, sensor.lower(), scene_id)
    if not class_map_path.is_file():
        raise FileNotFoundError(
            f"No Stage 6 classification found for {tile_id}/{sensor}/{scene_id} at '{class_map_path}' -- "
            "run scripts/06_run_inference.py first (spectral composites need a scene's own classification "
            "to mask out cloud/shadow pixels)."
        )
    class_map_data = read_class_map(class_map_path)
    class_map = class_map_data["class_map"]
    change_eligible_ids = set(class_config.change_eligible_ids())
    valid_mask = np.isin(class_map, list(change_eligible_ids)) & (class_map != class_map_data["nodata"])

    indices = indices_from_bands(bands, provenance, sensor, index_names)

    year = scene_year(sensor.lower(), scene_id)
    month = scene_month(sensor.lower(), scene_id)

    return SceneIndices(
        sensor=sensor.lower(),
        scene_id=scene_id,
        year=year,
        month=month,
        indices=indices,
        valid_mask=valid_mask,
        transform=tuple(float(v) for v in attrs["transform"]),
        crs_wkt=str(attrs["crs_wkt"]),
    )


def discover_tile_scenes(tile_dir: str | Path, tile_id: str) -> list[tuple[str, str]]:
    """Every ``(sensor, scene_id)`` stored for this tile, across sensors,
    sorted for determinism."""
    import zarr

    from landscape_change_detection_pipeline.scenes.zarr_store import zarr_path_for_tile

    zarr_path = zarr_path_for_tile(tile_dir, tile_id)
    if not zarr_path.exists():
        return []
    store = zarr.open_group(str(zarr_path), mode="r")

    results: list[tuple[str, str]] = []
    for sensor in SENSORS:
        if sensor.upper() not in store:
            continue
        scene_ids = sorted(store[sensor.upper()].attrs.get("scene_ids", []))
        results.extend((sensor, scene_id) for scene_id in scene_ids)
    return results


def month_key(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def group_scenes_by_month(scenes: list[SceneIndices]) -> dict[str, list[SceneIndices]]:
    groups: dict[str, list[SceneIndices]] = {}
    for scene in scenes:
        groups.setdefault(month_key(scene.year, scene.month), []).append(scene)
    return groups


def _fast_nan_median(sorted_stack: np.ndarray, valid_count: np.ndarray) -> np.ndarray:
    """Per-pixel median over the scene axis, given a stack already sorted
    along axis 0 with NaNs pushed to the end (``np.sort`` puts NaN last) and
    the count of non-NaN entries at each pixel.

    Avoids ``np.nanmedian`` deliberately: profiling this module's real
    workload (23 real pilot-tile months x 15 indices = 345 calls) showed
    ``np.nanmedian`` spending the bulk of its time -- 44s of a 44.6s total --
    inside NumPy's internal ``numpy.ma``-based fallback, which emits (and
    then immediately suppresses) millions of per-slice Python warnings
    objects even when the *result* is correct and already re-masked to NaN
    below. Indexing directly into an already-sorted stack by each pixel's own
    valid-count parity is the same definition of "median of the valid
    values" without ever constructing a masked array, at a fraction of the
    cost -- the sort is shared across the median/min/max/n_obs computation
    in :func:`_reduce_index_stack` rather than repeated per statistic."""
    n = sorted_stack.shape[0]
    lower_idx = np.clip((valid_count - 1) // 2, 0, n - 1)
    upper_idx = np.clip(valid_count // 2, 0, n - 1)
    lower = np.take_along_axis(sorted_stack, lower_idx[np.newaxis, :, :], axis=0)[0]
    upper = np.take_along_axis(sorted_stack, upper_idx[np.newaxis, :, :], axis=0)[0]
    return ((lower + upper) / 2.0).astype(np.float32)


def _reduce_index_stack(stack: np.ndarray, valid_stack: np.ndarray) -> dict[str, np.ndarray]:
    """One index's ``(n_scenes, H, W)`` value stack + validity mask reduced to
    the four per-pixel statistics (see module docstring). Pixels with zero
    valid observations get ``nan`` for median/min/max and ``0`` for
    ``n_obs``."""
    masked = np.where(valid_stack, stack, np.nan).astype(np.float32)
    n_obs = valid_stack.sum(axis=0).astype(np.int16)
    no_obs = n_obs == 0

    # NaN sorts to the end of each pixel's column, so the first n_obs entries
    # along axis 0 are exactly that pixel's valid values, ascending -- reused
    # for median (via _fast_nan_median), min (index 0), and max (index
    # n_obs-1), one sort instead of three separate nan-aware reductions.
    sorted_stack = np.sort(masked, axis=0)

    median = _fast_nan_median(sorted_stack, n_obs)
    min_ = sorted_stack[0]
    max_idx = np.clip(n_obs - 1, 0, stack.shape[0] - 1)
    max_ = np.take_along_axis(sorted_stack, max_idx[np.newaxis, :, :], axis=0)[0]

    median = np.where(no_obs, np.nan, median).astype(np.float32)
    min_ = np.where(no_obs, np.nan, min_).astype(np.float32)
    max_ = np.where(no_obs, np.nan, max_).astype(np.float32)

    return {"median": median, "min": min_, "max": max_, "n_obs": n_obs}


def build_month_index_composite(
    scenes: list[SceneIndices],
    index_names: tuple[str, ...] = ALL_INDEX_NAMES,
) -> dict:
    """Reduce one tile-month's per-scene indices to per-index statistics on
    that month's reference grid (the first scene's grid -- every scene of
    one tile/sensor already shares one grid per ``zarr_store``'s own
    same-grid invariant; a month can still mix sensors at *different*
    resolutions, so scenes are reprojected onto the finest one present, the
    same discipline Stage 7 uses for class maps)."""
    if not scenes:
        raise ValueError("build_month_index_composite requires at least one scene")

    # Every sensor is fetched onto the unified 10 m grid pinned to the DEM, so
    # this is normally a no-op tie -- but the reference is still picked by
    # each scene's own actual pixel size (from its transform), not assumed,
    # so a scene that is for any reason off the unified grid (stale data, a
    # bug elsewhere) still can't silently become the reference grid over a
    # genuinely finer one. Ties (the expected case) break on (sensor,
    # scene_id) for determinism.
    reference = min(scenes, key=lambda s: (abs(s.transform[0]), s.sensor, s.scene_id))
    dst_shape = next(iter(reference.indices.values())).shape

    aligned_per_index: dict[str, list[np.ndarray]] = {name: [] for name in index_names}
    aligned_valid: list[np.ndarray] = []

    for scene in scenes:
        if scene is reference:
            for name in index_names:
                aligned_per_index[name].append(scene.indices[name])
            aligned_valid.append(scene.valid_mask)
            continue

        valid_f32 = _reproject_nearest(
            scene.valid_mask.astype(np.uint8), scene.transform, scene.crs_wkt,
            reference.transform, reference.crs_wkt, dst_shape,
        ).astype(bool)
        aligned_valid.append(valid_f32)
        for name in index_names:
            aligned_per_index[name].append(
                _reproject_bilinear(
                    scene.indices[name], scene.transform, scene.crs_wkt,
                    reference.transform, reference.crs_wkt, dst_shape,
                )
            )

    valid_stack = np.stack(aligned_valid, axis=0)
    stats: dict[str, dict[str, np.ndarray]] = {}
    for name in index_names:
        stack = np.stack(aligned_per_index[name], axis=0)
        stats[name] = _reduce_index_stack(stack, valid_stack)

    sensors_present = sorted({s.sensor for s in scenes})
    resolution_m = abs(reference.transform[0])

    return {
        "stats": stats,
        "transform": reference.transform,
        "crs_wkt": reference.crs_wkt,
        "resolution_m": resolution_m,
        "n_scenes": len(scenes),
        "sensors_present": sensors_present,
    }


def _reproject_nearest(array, src_transform, src_crs_wkt, dst_transform, dst_crs_wkt, dst_shape):
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    destination = np.zeros(dst_shape, dtype=array.dtype)
    reproject(
        source=array,
        destination=destination,
        src_transform=Affine(*src_transform[:6]),
        src_crs=CRS.from_user_input(src_crs_wkt),
        dst_transform=Affine(*dst_transform[:6]),
        dst_crs=CRS.from_user_input(dst_crs_wkt),
        resampling=Resampling.nearest,
    )
    return destination


def _reproject_bilinear(array, src_transform, src_crs_wkt, dst_transform, dst_crs_wkt, dst_shape):
    """Continuous-valued index reprojection: bilinear, unlike Stage 7's
    nearest-neighbour class-map reprojection -- these are reflectance-derived
    indices, not categorical labels, so interpolating between neighbouring
    values is correct here (see ``dem/*``'s own bilinear convention for the
    same continuous-vs-categorical distinction)."""
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    destination = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=array.astype(np.float32),
        destination=destination,
        src_transform=Affine(*src_transform[:6]),
        src_crs=CRS.from_user_input(src_crs_wkt),
        dst_transform=Affine(*dst_transform[:6]),
        dst_crs=CRS.from_user_input(dst_crs_wkt),
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return destination


def discover_periods_for_tile(
    composites_root: str | Path, tile_id: str, filename: str = "indices.npz"
) -> list[str]:
    """Every ``"{year:04d}-{month:02d}"`` period with a stored composite
    file for this one tile, sorted -- the single-tile analogue of
    ``mosaic.mosaic.discover_periods`` (which scans across every tile).
    ``filename`` defaults to ``"indices.npz"`` for historical reasons but is
    always passed explicitly by every current caller -- both
    ``change.change_maps`` and ``change.regrowth_severity`` use this to walk
    Stage 7's plain classification composites (``"composite.npz"``), the
    only per-tile-month directory layout still produced by this
    pipeline."""
    tile_dir = Path(composites_root) / tile_id
    if not tile_dir.is_dir():
        return []
    return sorted(
        p.name for p in tile_dir.iterdir() if p.is_dir() and (p / filename).is_file()
    )


