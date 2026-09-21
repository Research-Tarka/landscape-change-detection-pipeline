"""Monthly categorical composites.

For each tile and month, every sensor's per-scene classification raster
(see :mod:`.engine`) that falls in that month is reprojected onto the
finest grid available that tile-month, then reduced to one composite per
class rule.

Resolution priority
--------------------
``composites.sensor_resolution_priority`` (default ``s2 > l9 > l8 > l7 >
l5``) picks each tile-month's composite grid: the first sensor in that list
with any scene that tile-month sets the resolution every other sensor's
class map is reprojected onto, nearest-neighbour (categorical labels, never
continuous-value resampling). The achieved resolution is stored as an
explicit per-composite attribute (``resolution_m``) rather than assumed
uniform across the time series, since a tile can have Sentinel-2 coverage
one month and only Landsat 5 the next.

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

#: Landsat/Sentinel-2 native resolution, metres per pixel, used only to
#: report ``resolution_m`` on each composite (the actual grid comes from the
#: winning sensor's own stored transform).
SENSOR_RESOLUTION_M: dict[str, float] = {"s2": 10.0, "l9": 15.0, "l8": 15.0, "l7": 15.0, "l5": 30.0}


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


def _median_reduce(stack: np.ndarray, class_id: int, nodata: int) -> np.ndarray:
    """Per-pixel majority vote for ``class_id``: true wherever more of the
    valid (non-nodata) votes at that pixel are this class than any other."""
    is_class = stack == class_id
    valid = stack != nodata
    votes_for = is_class.sum(axis=0)
    votes_valid = valid.sum(axis=0)
    # Majority within the valid votes -- strictly more than half, so a class
    # tied with another does not spuriously win under "median".
    return (votes_valid > 0) & (votes_for * 2 > votes_valid)


def _any_occurrence_reduce(stack: np.ndarray, class_id: int) -> np.ndarray:
    """True wherever ``class_id`` appears in even one scene of the stack."""
    return (stack == class_id).any(axis=0)


def build_monthly_composite(
    scenes: list[SceneClassMap],
    num_classes: int,
    sensor_priority: tuple[str, ...],
    default_rule: str,
    class_rules: dict[int, tuple[str, int]],
    nodata: int = 255,
) -> dict:
    """Reduce one tile-month's scene class maps to one composite.

    ``class_rules`` is ``{class_id: (rule, priority)}`` for classes with a
    non-default rule; every other class uses ``default_rule`` with priority
    0. Returns a dict with the ``(H, W)`` composite, the reference
    georeferencing, the achieved resolution, and the scene count.
    """
    if not scenes:
        raise ValueError("build_monthly_composite requires at least one scene")

    stack, reference = _aligned_stack(scenes, sensor_priority)
    height, width = stack.shape[1:]

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
            mask = _any_occurrence_reduce(stack, class_id)
            if mask.any():
                any_occurrence_hits.append((priority, class_id, mask))
        elif rule == "fallback_occurrence":
            mask = _any_occurrence_reduce(stack, class_id)
            if mask.any():
                fallback_occurrence_hits.append((priority, class_id, mask))
        else:
            mask = _median_reduce(stack, class_id, nodata)
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
    resolution_m = min(SENSOR_RESOLUTION_M.get(s, float("nan")) for s in sensors_present)

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
) -> list[Path]:
    """Build every tile-month composite available for one tile. Returns the
    paths actually (re)written."""
    scenes = discover_scene_class_maps(inference_root, tile_id)
    groups = group_by_tile_month(scenes)

    written: list[Path] = []
    for month in sorted(groups):
        out_path = composite_output_path(output_root, tile_id, month)
        if out_path.is_file() and not overwrite:
            continue
        result = build_monthly_composite(
            groups[month], num_classes, sensor_priority, default_rule, class_rules, nodata
        )
        write_composite(out_path, result)
        written.append(out_path)
    return written
