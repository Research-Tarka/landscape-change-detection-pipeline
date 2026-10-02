"""Monthly categorical composites.

For each tile and month, every sensor's per-scene classification raster
(see :mod:`.engine`) that falls in that month is reprojected onto the
reference grid, then reduced to one composite per class rule.

Reference grid
---------------
Every sensor is fetched onto the unified 10 m grid pinned to the DEM, so
there is no per-sensor resolution left to pick between -- ``composites.sensor_resolution_priority``
(default ``s2 > l9 > l8 > l7 > l5``) only picks *which* sensor's scene
supplies the reference transform when more than one is present that
tile-month (an arbitrary but deterministic tie-break, not a resolution
choice), and every other sensor's class map is reprojected onto it,
nearest-neighbour (categorical labels, never continuous-value resampling).
The achieved resolution is read directly off the reference scene's own
transform and stored as an explicit per-composite attribute
(``resolution_m``), rather than assumed -- confirms the unified grid held
rather than silently trusting it.

Per-class reduction rule
-------------------------
Configurable per class (``composites.class_rules``), defaulting to
``composites.default_rule``:

- ``"median"`` -- for stable classes (forest, water, wetland, built-up):
  the per-pixel majority/median vote across the month's scenes.
- ``"any_occurrence"`` -- for transient-but-important flags that must
  survive even a single occurrence and must win outright (e.g. a
  cutblock/burn flag: a real, durable disturbance event that a
  majority-median vote could otherwise vote out just because most of the
  month's scenes still show the pre-disturbance cover). Always overrides
  the median result wherever it fires.
- ``"fallback_occurrence"`` -- for transient, naturally-fluctuating states
  that should be recorded when nothing more stable is present, but must
  never mask a real land-cover change underneath them (e.g. snow/ice: a
  single snowy scene should not make a pixel read as "snow" for the whole
  month if most scenes that month show the snow has melted -- that melt is
  exactly the signal a snow/ice-trajectory analysis needs to see, so a
  transient snow observation must not out-vote it). Only applied where the
  median pass left a pixel unresolved (no class reached a majority) --
  never overrides a genuine median winner.

Computed as: per-pixel per-class presence counts across the month's scenes,
the configured rule applied independently per class, then conflicts (more
than one class of the same rule firing on the same pixel) resolved by
``CompositeClassRule.priority`` (lower wins). The exact rule table is a
config default, not hardcoded logic -- the right per-class choice needs
ecologist input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from landscape_change_detection_pipeline.inference.engine import read_class_map, scene_output_paths


@dataclass(frozen=True)
class SceneClassMap:
    sensor: str
    scene_id: str
    year: int
    month: int
    class_map: np.ndarray
    transform: tuple[float, ...]
    crs_wkt: str
    nodata: int


def month_key(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def discover_scene_class_maps(
    inference_root: str | Path,
    tile_id: str,
    sensors: tuple[str, ...] = ("l5", "l7", "l8", "l9", "s2"),
) -> list[SceneClassMap]:
    """Find every stored per-scene class map for one tile, across sensors,
    with its acquisition (year, month) parsed from the scene id. Sorted by
    ``(sensor, scene_id)`` for determinism."""
    from landscape_change_detection_pipeline.features.training_cache import scene_year

    root = Path(inference_root) / tile_id
    results: list[SceneClassMap] = []
    if not root.is_dir():
        return results

    for sensor in sensors:
        sensor_dir = root / sensor
        if not sensor_dir.is_dir():
            continue
        for scene_dir in sorted(p for p in sensor_dir.iterdir() if p.is_dir()):
            npz_path = scene_dir / "class_map.npz"
            if not npz_path.is_file():
                continue
            scene_id = scene_dir.name
            data = read_class_map(npz_path)
            year = scene_year(sensor, scene_id)
            month = scene_month(sensor, scene_id)
            results.append(
                SceneClassMap(
                    sensor=sensor,
                    scene_id=scene_id,
                    year=year,
                    month=month,
                    class_map=data["class_map"],
                    transform=data["transform"],
                    crs_wkt=data["crs_wkt"],
                    nodata=data["nodata"],
                )
            )

    results.sort(key=lambda s: (s.sensor, s.scene_id))
    return results


def scene_month(sensor: str, scene_id: str) -> int:
    """Parse the acquisition month out of a scene id (same id formats as
    :func:`landscape_change_detection_pipeline.features.training_cache.scene_year`)."""
    import re

    if sensor.upper() == "S2":
        match = re.match(r"^\d{4}(\d{2})\d{2}T\d{6}", scene_id)
    else:
        match = re.search(r"_\d{4}(\d{2})\d{2}$", scene_id)
    if not match:
        raise ValueError(f"Could not parse an acquisition month from scene_id '{scene_id}' (sensor '{sensor}').")
    return int(match.group(1))


def group_by_tile_month(scenes: list[SceneClassMap]) -> dict[str, list[SceneClassMap]]:
    """Group scene class maps by ``"{year:04d}-{month:02d}"``, sorted keys
    determined by the caller (dict preserves insertion order here, built
    from an already-sorted ``scenes`` list)."""
    groups: dict[str, list[SceneClassMap]] = {}
    for scene in scenes:
        key = month_key(scene.year, scene.month)
        groups.setdefault(key, []).append(scene)
    return groups


def pick_reference_grid(
    scenes: list[SceneClassMap],
    sensor_priority: tuple[str, ...],
) -> SceneClassMap:
    """The scene whose grid every other scene in this tile-month is
    reprojected onto: the first sensor in ``sensor_priority`` with any scene
    present, breaking ties by scene id for determinism."""
    by_sensor: dict[str, list[SceneClassMap]] = {}
    for scene in scenes:
        by_sensor.setdefault(scene.sensor, []).append(scene)

    for sensor in sensor_priority:
        candidates = by_sensor.get(sensor)
        if candidates:
            return sorted(candidates, key=lambda s: s.scene_id)[0]

    # No configured-priority sensor present (an unlisted sensor only) --
    # fall back to the first scene, sorted, rather than failing.
    return sorted(scenes, key=lambda s: (s.sensor, s.scene_id))[0]


def reproject_class_map_nearest(
    class_map: np.ndarray,
    src_transform: tuple[float, ...],
    src_crs_wkt: str,
    dst_transform,
    dst_crs_wkt: str,
    dst_shape: tuple[int, int],
    nodata: int,
) -> np.ndarray:
    """Reproject one categorical class map onto a reference grid,
    nearest-neighbour (never any continuous-value resampling -- these are
    class labels, not reflectance).

    Despite the ``crs_wkt`` name (inherited from
    ``scenes.zarr_store``/``inference.engine``'s own attrs, which stores
    whatever CRS string the pipeline was configured with -- typically an
    EPSG code like ``"EPSG:26910"``, not actual WKT text), this is parsed
    with :meth:`CRS.from_user_input`, which accepts both forms, rather than
    :meth:`CRS.from_wkt`, which raises on an EPSG string.
    """
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    src_affine = Affine(*src_transform[:6])
    destination = np.full(dst_shape, nodata, dtype=np.uint8)
    reproject(
        source=class_map.astype(np.uint8),
        destination=destination,
        src_transform=src_affine,
        src_crs=CRS.from_user_input(src_crs_wkt),
        dst_transform=dst_transform,
        dst_crs=CRS.from_user_input(dst_crs_wkt),
        resampling=Resampling.nearest,
        src_nodata=nodata,
        dst_nodata=nodata,
    )
    return destination


def _aligned_stack(scenes: list[SceneClassMap], sensor_priority: tuple[str, ...]) -> tuple[np.ndarray, SceneClassMap]:
    """Every scene's class map reprojected onto the tile-month's reference
    grid, stacked as ``(n_scenes, H, W)``."""
    from affine import Affine

    reference = pick_reference_grid(scenes, sensor_priority)
    dst_transform = Affine(*reference.transform[:6])
    dst_shape = reference.class_map.shape

    aligned = []
    for scene in scenes:
        if scene is reference:
            aligned.append(scene.class_map.astype(np.uint8))
            continue
        aligned.append(
            reproject_class_map_nearest(
                scene.class_map,
                scene.transform,
                scene.crs_wkt,
                dst_transform,
                reference.crs_wkt,
                dst_shape,
                nodata=scene.nodata,
            )
        )
    return np.stack(aligned, axis=0), reference


def _per_class_vote_counts(stack: np.ndarray, num_classes: int, nodata: int) -> np.ndarray:
    """Per-pixel, per-class vote counts across the scene axis, in one pass.

    Returns ``(num_classes, H, W)`` int32 counts. This one-hot-and-sum
    (``stack[..., None] == arange(num_classes)``, summed over the scene
    axis) replaces the previous per-class-in-a-Python-loop ``stack ==
    class_id`` scan -- num_classes full ``(n_scenes, H, W)`` passes over the
    same array -- with a single vectorized pass, since
    ``_median_reduce_from_counts``/``_any_occurrence_reduce_from_counts``
    below need nothing from the stack except each class's own count per
    pixel. Nodata votes count toward no class (excluded via ``valid``
    below), matching the previous per-class scan's own semantics.
    """
    valid = stack != nodata
    class_ids = np.arange(num_classes, dtype=stack.dtype).reshape(num_classes, 1, 1, 1)
    one_hot = (stack[np.newaxis, ...] == class_ids) & valid[np.newaxis, ...]
    return one_hot.sum(axis=1, dtype=np.int32)


def _median_reduce_from_counts(counts: np.ndarray, class_id: int, votes_valid: np.ndarray) -> np.ndarray:
    """Per-pixel majority vote for ``class_id`` from precomputed vote counts:
    true wherever more of the valid (non-nodata) votes at that pixel are this
    class than any other (strictly more than half, so a tie never wins)."""
    votes_for = counts[class_id]
    return (votes_valid > 0) & (votes_for * 2 > votes_valid)


def _any_occurrence_reduce_from_counts(counts: np.ndarray, class_id: int) -> np.ndarray:
    """True wherever ``class_id`` appears in even one scene of the stack,
    from precomputed vote counts."""
    return counts[class_id] > 0


def build_monthly_composite(
    scenes: list[SceneClassMap],
    num_classes: int,
    sensor_priority: tuple[str, ...],
    default_rule: str,
    class_rules: dict[int, tuple[str, int]],
    nodata: int = 255,
    ignore_class_ids: tuple[int, ...] = (),
) -> dict:
    """Reduce one tile-month's scene class maps to one composite.

    ``ignore_class_ids``: raw class ids whose pixels cast no vote at all
    (treated as nodata) -- typically cloud/shadow, so a month is decided by
    its clear observations only.

    ``class_rules`` is ``{class_id: (rule, priority)}`` for classes with a
    non-default rule; every other class uses ``default_rule`` with priority
    0. Returns a dict with the ``(H, W)`` composite, the reference
    georeferencing, the achieved resolution, and the scene count.
    """
    if not scenes:
        raise ValueError("build_monthly_composite requires at least one scene")

    stack, reference = _aligned_stack(scenes, sensor_priority)
    if ignore_class_ids:
        stack = np.where(np.isin(stack, list(ignore_class_ids)), nodata, stack).astype(stack.dtype)
    height, width = stack.shape[1:]

    counts = _per_class_vote_counts(stack, num_classes, nodata)
    votes_valid = counts.sum(axis=0)

    any_occurrence_hits: list[tuple[int, int, np.ndarray]] = []  # (priority, class_id, mask)
    fallback_occurrence_hits: list[tuple[int, int, np.ndarray]] = []  # (priority, class_id, mask)
    # -1 is a sentinel distinct from `nodata` (which can be any uint8 value,
    # e.g. 255): a pixel starts "unresolved" and only becomes a genuine
    # median winner when some class's _median_reduce mask actually covers
    # it, so "unresolved" and "nodata" must never be conflated even when
    # nodata happens to be a value >= 0.
    median_winner = np.full((height, width), -1, dtype=np.int32)

    for class_id in range(num_classes):
        rule, priority = class_rules.get(class_id, (default_rule, 0))
        if rule == "any_occurrence":
            mask = _any_occurrence_reduce_from_counts(counts, class_id)
            if mask.any():
                any_occurrence_hits.append((priority, class_id, mask))
        elif rule == "fallback_occurrence":
            mask = _any_occurrence_reduce_from_counts(counts, class_id)
            if mask.any():
                fallback_occurrence_hits.append((priority, class_id, mask))
        else:
            mask = _median_reduce_from_counts(counts, class_id, votes_valid)
            median_winner = np.where(mask, class_id, median_winner)

    composite = np.where(median_winner >= 0, median_winner, nodata).astype(np.uint8)

    # fallback_occurrence classes fill in only where the median pass left a
    # pixel unresolved -- never overriding a genuine median winner, unlike
    # any_occurrence below (see module docstring for why: a transient
    # snow/ice observation must not mask a real land-cover class that a
    # majority of the month's scenes actually agree on).
    unresolved = median_winner < 0
    for priority, class_id, mask in sorted(fallback_occurrence_hits, key=lambda t: -t[0]):
        composite = np.where(unresolved & mask, class_id, composite).astype(np.uint8)

    # any_occurrence classes override the median (and fallback_occurrence)
    # result wherever they fire, highest priority (lowest number) last so it
    # wins the final overwrite.
    for priority, class_id, mask in sorted(any_occurrence_hits, key=lambda t: -t[0]):
        composite = np.where(mask, class_id, composite).astype(np.uint8)

    sensors_present = sorted({s.sensor for s in scenes})
    # Every sensor is fetched onto the unified 10 m grid pinned to the DEM --
    # there is no per-sensor resolution left to pick between, so the achieved resolution is read
    # directly off the reference scene's own transform (its actual pixel
    # size) rather than a per-sensor native-resolution table, which would
    # report each sensor's pre-unified-grid resolution (e.g. 15/30 m for
    # Landsat) even though every scene has already been resampled to 10 m.
    resolution_m = abs(reference.transform[0])

    return {
        "composite": composite,
        "transform": reference.transform,
        "crs_wkt": reference.crs_wkt,
        "resolution_m": resolution_m,
        "n_scenes": len(scenes),
        "sensors_present": sensors_present,
        "nodata": nodata,
    }


def composite_output_path(output_root: str | Path, tile_id: str, month: str) -> Path:
    """``<output_root>/<tile_id>/<month>/composite.npz``."""
    return Path(output_root) / tile_id / month / "composite.npz"


def write_composite(path: str | Path, result: dict) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        composite=result["composite"],
        transform=np.array(list(result["transform"])[:6], dtype=np.float64),
        crs_wkt=np.array(result["crs_wkt"]),
        resolution_m=np.array(result["resolution_m"], dtype=np.float64),
        n_scenes=np.array(result["n_scenes"], dtype=np.int32),
        sensors_present=np.array(result["sensors_present"]),
        nodata=np.array(result["nodata"], dtype=np.uint8),
    )
    return out


def read_composite(path: str | Path) -> dict:
    """Read a monthly composite written by :func:`write_composite`."""
    with np.load(Path(path), allow_pickle=False) as data:
        return {
            "composite": np.array(data["composite"]),
            "transform": tuple(float(v) for v in data["transform"]),
            "crs_wkt": str(data["crs_wkt"]),
            "resolution_m": float(data["resolution_m"]),
            "n_scenes": int(data["n_scenes"]),
            "sensors_present": [str(s) for s in data["sensors_present"]],
            "nodata": int(data["nodata"]),
        }


def build_all_monthly_composites(
    inference_root: str | Path,
    output_root: str | Path,
    tile_id: str,
    num_classes: int,
    sensor_priority: tuple[str, ...],
    default_rule: str,
    class_rules: dict[int, tuple[str, int]],
    nodata: int = 255,
    overwrite: bool = False,
    progress_callback=None,
    ignore_class_ids: tuple[int, ...] = (),
) -> list[Path]:
    """Build every tile-month composite available for one tile. Returns the
    paths actually (re)written.

    ``progress_callback(tile_id, month, month_index, total_months, written)``,
    if given, is called once per month *processed* (written or skipped as
    already built) -- a long tile (hundreds of months) otherwise gives no
    signal at all until the whole tile is done, which is unusable for
    tracking a real run's progress. Must be picklable if this runs inside a
    ``ProcessPoolExecutor`` worker (a plain function or a
    ``functools.partial`` of one, never a closure/lambda over local state).
    """
    scenes = discover_scene_class_maps(inference_root, tile_id)
    groups = group_by_tile_month(scenes)
    months = sorted(groups)

    written: list[Path] = []
    for month_index, month in enumerate(months, start=1):
        out_path = composite_output_path(output_root, tile_id, month)
        wrote_this_month = False
        if out_path.is_file() and not overwrite:
            pass
        else:
            result = build_monthly_composite(
                groups[month], num_classes, sensor_priority, default_rule, class_rules, nodata,
                ignore_class_ids,
            )
            write_composite(out_path, result)
            written.append(out_path)
            wrote_this_month = True
        if progress_callback is not None:
            progress_callback(tile_id, month, month_index, len(months), wrote_this_month)
    return written
