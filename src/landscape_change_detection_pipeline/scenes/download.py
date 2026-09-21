"""Scene discovery, filtering, and sanity-checking.

Purpose
-------
For each tile (:mod:`landscape_change_detection_pipeline.tiles.registry`) and sensor
(:mod:`.sensors`), find which scenes are worth downloading: query the GEE
collection over the tile's unbuffered search bbox and configured date
range/year, read back scene id + cloud property, then filter down to a
manageable, already-not-cached, low-cloud, well-covered set -- before
the fetch stage spends any real download quota on them.

This module only *decides* which scenes to fetch; it does not fetch pixels
(that is ``gee_fetch.py``). Filtering logic (resume, cloud
threshold, AOI coverage, least-cloudy budget cap), the empty-collection
``toList(0)`` guard, and the ``reduceRegion(minMax())`` reflectance sanity
check follow the same pattern as ``scenes/gee_fetch.py::reflectance_sanity_check``.

Server-side geometry note
--------------------------
The search geometry passed to Earth Engine is built directly in the tile
registry's own CRS (``aoi_crs``, e.g. EPSG:26910), never as a reprojected
lon/lat rectangle -- see ``tiles/registry.py``'s module docstring for the
confirmed ~1.5 km grid-shift bug this avoids at this AOI's latitude.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .sensors import SensorSpec, date_bounds, get_sensor

#: Landsat scene ids end in an 8-digit acquisition date, e.g.
#: "LT05_048021_20100725" -> month 7. Sentinel-2 ids start with one, e.g.
#: "20230607T191911_..." -> month 6. Mirrors
#: ``features.training_cache``'s own scene-id date regexes; duplicated here
#: (rather than imported) since ``features`` already imports from
#: ``scenes``, and importing back would be circular.
_LANDSAT_DATE_RE = re.compile(r"_(\d{4})(\d{2})(\d{2})$")
_S2_DATE_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})T\d{6}")


def scene_month(sensor_key: str, scene_id: str) -> int:
    """Parse the acquisition month (1-12) out of a scene id.

    Used to bucket a year's search results into per-month groups for the
    ``max_scenes_per_tile_month`` budget cap, without an extra GEE call per
    scene (the acquisition date is already encoded in the scene id).
    """
    if sensor_key.upper() == "S2":
        match = _S2_DATE_RE.match(scene_id)
    else:
        match = _LANDSAT_DATE_RE.search(scene_id)
    if not match:
        raise ValueError(f"Could not parse an acquisition month from scene_id '{scene_id}' (sensor '{sensor_key}').")
    return int(match.group(2))

#: Minimum fraction of the tile's search bbox a scene's own footprint must
#: cover to be considered usable -- screens out scenes that only clip a
#: corner of the tile (e.g. at a swath edge), which would otherwise pass
#: every other filter yet contribute almost no real pixels.
DEFAULT_MIN_AOI_COVERAGE_PCT = 50.0

#: Reflectance sanity check (mirrors gee_fetch.py::reflectance_sanity_check):
#: the brightest pixel over the window must reach at least this TOA
#: reflectance, or the window is assumed to sit on the swath's fill/no-data
#: edge rather than containing real surface reflectance.
DEFAULT_MIN_PLAUSIBLE_REFLECTANCE = 0.15

#: Reflective bands checked by the sanity check, common to all five sensors'
#: band-naming conventions used here (visible/NIR bands, present natively at
#: full resolution for every sensor in :mod:`.sensors`).
SANITY_CHECK_BANDS = ("B3", "B4")


@dataclass
class SceneCandidate:
    """One scene as returned by :func:`search_scenes`, before filtering."""

    scene_id: str
    sensor: str
    tile_id: str
    year: int
    cloud_pct: float
    aoi_coverage_pct: float = 100.0
    month: int = field(init=False)

    def __post_init__(self) -> None:
        self.month = scene_month(self.sensor, self.scene_id)


@dataclass
class FilterResult:
    """The outcome of :func:`filter_scenes` for one tile/sensor/year."""

    kept: list[SceneCandidate] = field(default_factory=list)
    rejected: list[tuple[SceneCandidate, str]] = field(default_factory=list)
    already_cached: list[str] = field(default_factory=list)


def build_search_geometry(bbox: tuple[float, float, float, float], crs: str):
    """Build an ``ee.Geometry.Rectangle`` directly in ``crs`` (never lon/lat).

    Mirrors the fix documented in ``tiles/registry.py``: passing ``proj``
    explicitly means Earth Engine treats ``bbox`` as already being in that
    CRS, rather than reprojecting a lon/lat rectangle server-side.
    """
    import ee

    return ee.Geometry.Rectangle(list(bbox), proj=crs, evenOdd=True, geodesic=False)


def search_scenes(
    sensor: SensorSpec,
    tile_id: str,
    search_bbox: tuple[float, float, float, float],
    crs: str,
    year: int,
    date_start_mmdd: str,
    date_end_mmdd: str,
) -> list[SceneCandidate]:
    """Query one sensor's GEE collection for one tile/year, returning candidates.

    Guards the empty-collection ``.toList(0)`` gotcha: calling ``.getInfo()``
    directly on an ``ee.ImageCollection`` filtered down to zero images is a
    documented pitfall (some
    call patterns raise or return malformed results on an empty collection);
    the fix is to check ``.size().getInfo()`` first and
    short-circuit to an empty list without ever touching ``.getInfo()`` on
    the collection itself.
    """
    import ee

    start, end = date_bounds(year, date_start_mmdd, date_end_mmdd)
    geometry = build_search_geometry(search_bbox, crs)

    collection = (
        ee.ImageCollection(sensor.collection)
        .filterDate(start, end)
        .filterBounds(geometry)
    )

    size = collection.size().getInfo()
    if size == 0:
        return []

    ids_and_props = (
        collection.toList(size)
        .map(
            lambda img: ee.Feature(
                None,
                {
                    "scene_id": ee.Image(img).get("system:index"),
                    "cloud_pct": ee.Image(img).get(sensor.cloud_cover_property),
                },
            )
        )
        .getInfo()
    )

    candidates = []
    for feat in ids_and_props:
        props = feat["properties"]
        cloud_pct = props.get("cloud_pct")
        if cloud_pct is None:
            continue
        candidates.append(
            SceneCandidate(
                scene_id=str(props["scene_id"]),
                sensor=sensor.key,
                tile_id=tile_id,
                year=year,
                cloud_pct=float(cloud_pct),
            )
        )
    return candidates


def compute_aoi_coverage_pct(
    sensor: SensorSpec,
    scene_id: str,
    search_bbox: tuple[float, float, float, float],
    crs: str,
) -> float:
    """What percentage of the tile's search bbox this scene's footprint covers.

    A cheap server-side computation (image footprint intersected with the
    search geometry, area ratio via ``reduceRegion`` is not needed here --
    footprint geometry intersection is enough and far cheaper than a pixel
    reduction).
    """
    import ee

    geometry = build_search_geometry(search_bbox, crs)
    image = ee.Image(f"{sensor.collection}/{scene_id}")
    footprint = image.geometry()
    error_margin = ee.ErrorMargin(1)
    intersection_area = footprint.intersection(geometry, error_margin).area(error_margin)
    bbox_area = geometry.area(error_margin)
    ratio = intersection_area.divide(bbox_area).getInfo()
    return 100.0 * float(ratio)


def reflectance_sanity_check(
    sensor: SensorSpec,
    scene_id: str,
    window_bbox: tuple[float, float, float, float],
    crs: str,
    min_plausible_reflectance: float = DEFAULT_MIN_PLAUSIBLE_REFLECTANCE,
    bands: tuple[str, ...] = SANITY_CHECK_BANDS,
) -> bool:
    """Cheap server-side check that a scene has real (non-fill) data over the window.

    Mirrors ``gee_fetch.py::reflectance_sanity_check``: one small
    ``reduceRegion(ee.Reducer.minMax())`` server-side call (no pixels
    downloaded) over the tile's analysis window, rejecting a scene if the
    brightest pixel across ``bands`` never reaches ``min_plausible_reflectance``
    -- the signature of a swath-edge fill/no-data window passing every
    metadata-only filter above.
    """
    import ee

    geometry = build_search_geometry(window_bbox, crs)
    image = ee.Image(f"{sensor.collection}/{scene_id}")

    stats = (
        image.select(list(bands))
        .reduceRegion(reducer=ee.Reducer.minMax(), geometry=geometry, scale=sensor.native_resolution_m, bestEffort=True)
        .getInfo()
    )
    max_values = [v for k, v in stats.items() if k.endswith("_max") and v is not None]
    if not max_values:
        return False
    return max(max_values) >= min_plausible_reflectance


def filter_scenes(
    candidates: list[SceneCandidate],
    already_cached_ids: set[str],
    max_cloud_pct: float,
    min_aoi_coverage_pct: float = DEFAULT_MIN_AOI_COVERAGE_PCT,
    max_scenes_per_tile_month: Optional[int] = None,
) -> FilterResult:
    """Apply resume/cloud/coverage filters, then cap each month to its least-cloudy budget.

    1. Skip scenes already resolved in the resume cache.
    2. Reject scenes over ``max_cloud_pct``.
    3. Reject scenes under ``min_aoi_coverage_pct``.
    4. Within each acquisition month, if more than
       ``max_scenes_per_tile_month`` survivors remain, keep only the
       least-cloudy ones up to the budget (``None`` = unlimited). The cap
       applies per month, not per year, so every month keeps its own scenes
       rather than one cloudy month crowding out another.
    """
    result = FilterResult()
    survivors: list[SceneCandidate] = []

    for candidate in candidates:
        if candidate.scene_id in already_cached_ids:
            result.already_cached.append(candidate.scene_id)
            continue
        if candidate.cloud_pct > max_cloud_pct:
            result.rejected.append((candidate, f"cloud_pct {candidate.cloud_pct:.1f} > max {max_cloud_pct}"))
            continue
        if candidate.aoi_coverage_pct < min_aoi_coverage_pct:
            result.rejected.append(
                (candidate, f"aoi_coverage_pct {candidate.aoi_coverage_pct:.1f} < min {min_aoi_coverage_pct}")
            )
            continue
        survivors.append(candidate)

    if max_scenes_per_tile_month is not None:
        by_month: dict[int, list[SceneCandidate]] = {}
        for candidate in survivors:
            by_month.setdefault(candidate.month, []).append(candidate)

        kept: list[SceneCandidate] = []
        for month in sorted(by_month):
            group = sorted(by_month[month], key=lambda c: c.cloud_pct)
            kept.extend(group[:max_scenes_per_tile_month])
            for candidate in group[max_scenes_per_tile_month:]:
                result.rejected.append(
                    (candidate, f"over max_scenes_per_tile_month budget ({max_scenes_per_tile_month})")
                )
        survivors = kept

    result.kept = survivors
    return result


def discover_scenes_for_tile_sensor_year(
    sensor_key: str,
    tile_id: str,
    search_bbox: tuple[float, float, float, float],
    window_bbox: tuple[float, float, float, float],
    crs: str,
    year: int,
    cache_conn,
    date_start_mmdd: str = "01-01",
    date_end_mmdd: str = "12-31",
    max_cloud_pct: float = 20.0,
    min_aoi_coverage_pct: float = DEFAULT_MIN_AOI_COVERAGE_PCT,
    max_scenes_per_tile_month: Optional[int] = None,
    min_plausible_reflectance: float = DEFAULT_MIN_PLAUSIBLE_REFLECTANCE,
) -> FilterResult:
    """End-to-end discovery for one tile/sensor/year: search, filter, sanity-check, cache.

    The resume cache (:mod:`.cache`) is consulted before any GEE call for
    scenes already resolved, and every kept/rejected scene is recorded back
    into it so a re-run skips straight past this tile/sensor/year's work.
    """
    from . import cache as cache_mod

    spec = get_sensor(sensor_key)
    already_cached_ids = cache_mod.cached_scene_ids(cache_conn, tile_id, sensor_key, year)

    candidates = search_scenes(spec, tile_id, search_bbox, crs, year, date_start_mmdd, date_end_mmdd)

    # Cheap, metadata-only filters (resume cache, cloud_pct -- both already
    # in hand from search_scenes) are applied *before* the one-candidate-at-
    # a-time compute_aoi_coverage_pct call below, which is a live GEE
    # round trip per candidate. Computing coverage for every candidate up
    # front was confirmed live to make discovery pay for a
    # reduceRegion-class call on every scene a sensor ever collected over
    # this tile in a year (e.g. ~dozens for Landsat 8's ~16-day revisit
    # across a full year) before the cloud filter ever got a chance to
    # narrow anything down.
    cheap_result = filter_scenes(
        candidates,
        already_cached_ids,
        max_cloud_pct=max_cloud_pct,
        min_aoi_coverage_pct=0.0,  # not yet computed; coverage filter re-applied below
        max_scenes_per_tile_month=None,  # budget cap re-applied below, after coverage is known
    )

    for candidate in cheap_result.kept:
        candidate.aoi_coverage_pct = compute_aoi_coverage_pct(spec, candidate.scene_id, search_bbox, crs)

    result = filter_scenes(
        cheap_result.kept,
        already_cached_ids=set(),  # already excluded by the cheap pass above
        max_cloud_pct=max_cloud_pct,
        min_aoi_coverage_pct=min_aoi_coverage_pct,
        max_scenes_per_tile_month=max_scenes_per_tile_month,
    )
    result.already_cached = cheap_result.already_cached
    result.rejected = cheap_result.rejected + result.rejected

    sanity_failed = []
    still_kept = []
    for candidate in result.kept:
        if reflectance_sanity_check(spec, candidate.scene_id, window_bbox, crs, min_plausible_reflectance):
            still_kept.append(candidate)
        else:
            sanity_failed.append((candidate, "reflectance sanity check failed (edge-of-swath fill)"))
    result.kept = still_kept
    result.rejected.extend(sanity_failed)

    for candidate in result.kept:
        cache_mod.record_scene(
            cache_conn, tile_id, sensor_key, candidate.scene_id, year, "kept",
            cloud_pct=candidate.cloud_pct, month=candidate.month,
        )
    for candidate, reason in result.rejected:
        cache_mod.record_scene(
            cache_conn, tile_id, sensor_key, candidate.scene_id, year, "rejected",
            reason=reason, cloud_pct=candidate.cloud_pct, month=candidate.month,
        )

    return result
