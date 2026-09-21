"""Monthly per-tile spectral-index composites, for change-detection analysis.

Purpose
-------
:mod:`landscape_change_detection_pipeline.inference.composites` (Stage 7) reduces
per-scene *classification* rasters to one categorical class map per
tile-month. NDVI regrowth, dNBR burn severity, and the other index-based
change layers the roadmap calls for ("Then: change-detection layers") need
the underlying *continuous reflectance*, not the classes derived from it --
that information no longer exists once Stage 7 has voted it down to a single
class id per pixel. This module re-reads the same raw per-scene reflectance
Stage 7's classifier consumed (``data/tiles/<tile_id>.zarr/<sensor>/toa``),
computes every spectral index of interest per scene, and reduces each index
to one tile-month composite -- the same tile/month granularity as Stage 7,
built in parallel from a different source, never derived from Stage 7's own
output.

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

This is deliberately a superset of what any one index actually needs (dNBR
only reads ``min``, NDVI regrowth mainly reads ``median``/``max``) so the
same monthly cache also supports later seasonal/annual re-aggregation
without recomputing from raw scenes again -- annual or seasonal statistics
can be derived from these monthly per-pixel stats directly (e.g. an annual
median-of-medians, weighted by ``n_obs``), which is why all four are stored
unconditionally rather than only the one field each Prompt-17 layer reads
today.

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

from landscape_change_detection_pipeline.change.analysis_indices import ndwi_mcfeeters, tasseled_cap
from landscape_change_detection_pipeline.classes.class_config import ClassConfig
from landscape_change_detection_pipeline.features.spectral_indices import INDEX_NAMES, compute_indices_dict
from landscape_change_detection_pipeline.features.training_cache import bands_for_scene, scene_year
from landscape_change_detection_pipeline.inference.composites import scene_month
from landscape_change_detection_pipeline.inference.engine import read_class_map, scene_output_paths
from landscape_change_detection_pipeline.scenes.zarr_store import read_sensor_group_attrs

#: Every index this module computes per scene: the model's own eight
#: band-pair/chromaticity indices, plus the three Tasseled Cap components and
#: McFeeters' NDWI (analysis-only, see change/analysis_indices.py).
ALL_INDEX_NAMES: tuple[str, ...] = (*INDEX_NAMES, "TC_BRIGHTNESS", "TC_GREENNESS", "TC_WETNESS", "NDWI_MCFEETERS")

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

    if "NDWI_MCFEETERS" in index_names:
        indices["NDWI_MCFEETERS"] = ndwi_mcfeeters(bands["green"], bands["nir"])

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

    from landscape_change_detection_pipeline.inference.composites import SENSOR_RESOLUTION_M

    reference = min(scenes, key=lambda s: SENSOR_RESOLUTION_M.get(s.sensor, float("inf")))
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
    resolution_m = SENSOR_RESOLUTION_M.get(reference.sensor, float("nan"))

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


def index_composite_output_path(output_root: str | Path, tile_id: str, month: str) -> Path:
    """``<output_root>/<tile_id>/<month>/indices.npz``."""
    return Path(output_root) / tile_id / month / "indices.npz"


def discover_periods_for_tile(
    composites_root: str | Path, tile_id: str, filename: str = "indices.npz"
) -> list[str]:
    """Every ``"{year:04d}-{month:02d}"`` period with a stored composite
    file for this one tile, sorted -- the single-tile analogue of
    ``mosaic.mosaic.discover_periods`` (which scans across every tile).
    ``filename`` defaults to this module's own ``"indices.npz"`` (needed by
    e.g. ``change.regrowth_severity``'s NDVI trajectory), but the same
    ``<root>/<tile_id>/<period>/<filename>`` layout is shared by Stage 7's
    plain classification composites (``"composite.npz"``), so
    ``change.change_maps`` reuses this function for that directory too
    rather than duplicating the same directory walk."""
    tile_dir = Path(composites_root) / tile_id
    if not tile_dir.is_dir():
        return []
    return sorted(
        p.name for p in tile_dir.iterdir() if p.is_dir() and (p / filename).is_file()
    )


def write_index_composite(path: str | Path, result: dict) -> Path:
    """Write one tile-month's index statistics as an uncompressed ``.npz``.

    Deliberately ``np.savez`` (uncompressed), not ``savez_compressed`` --
    unlike the categorical, small-integer-alphabet class maps every other
    ``.npz`` writer in this pipeline stores (which zlib compresses fast and
    well), this payload is up to 15 indices x 4 float32 statistics of
    continuous, high-entropy reflectance-derived values. Measured on real
    pilot-tile data: zlib compression time dominated this module's entire
    runtime (85s of a 107s full-tile build, ~80%) for an ~18% size reduction
    -- a bad trade this module does not make.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "transform": np.array(list(result["transform"])[:6], dtype=np.float64),
        "crs_wkt": np.array(result["crs_wkt"]),
        "resolution_m": np.array(result["resolution_m"], dtype=np.float64),
        "n_scenes": np.array(result["n_scenes"], dtype=np.int32),
        "sensors_present": np.array(result["sensors_present"]),
        "index_names": np.array(list(result["stats"].keys())),
    }
    for name, stats in result["stats"].items():
        for stat_name, array in stats.items():
            payload[f"{name}__{stat_name}"] = array
    np.savez(out, **payload)
    return out


def read_index_composite(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as data:
        index_names = [str(n) for n in data["index_names"]]
        stats: dict[str, dict[str, np.ndarray]] = {}
        for name in index_names:
            stats[name] = {stat: np.array(data[f"{name}__{stat}"]) for stat in STATS}
        return {
            "stats": stats,
            "transform": tuple(float(v) for v in data["transform"]),
            "crs_wkt": str(data["crs_wkt"]),
            "resolution_m": float(data["resolution_m"]),
            "n_scenes": int(data["n_scenes"]),
            "sensors_present": [str(s) for s in data["sensors_present"]],
        }


def build_all_month_index_composites(
    tile_dir: str | Path,
    inference_root: str | Path,
    output_root: str | Path,
    tile_id: str,
    class_config: ClassConfig,
    index_names: tuple[str, ...] = ALL_INDEX_NAMES,
    overwrite: bool = False,
) -> list[Path]:
    """Build every tile-month index composite available for one tile.
    Scenes with no Stage 6 classification yet are skipped (a coverage gap,
    not an error -- classification may simply not have reached that scene
    yet). Returns the paths actually (re)written."""
    written: list[Path] = []
    scene_keys = discover_tile_scenes(tile_dir, tile_id)

    scenes: list[SceneIndices] = []
    for sensor, scene_id in scene_keys:
        try:
            scenes.append(
                compute_scene_indices(tile_dir, inference_root, tile_id, sensor, scene_id, class_config, index_names)
            )
        except FileNotFoundError:
            continue

    groups = group_scenes_by_month(scenes)
    for month in sorted(groups):
        out_path = index_composite_output_path(output_root, tile_id, month)
        if out_path.is_file() and not overwrite:
            continue
        result = build_month_index_composite(groups[month], index_names)
        write_index_composite(out_path, result)
        written.append(out_path)
    return written
