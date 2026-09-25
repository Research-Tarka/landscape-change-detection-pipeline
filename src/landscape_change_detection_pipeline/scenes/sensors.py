"""Sensor specifications for the Google Earth Engine scene search/download.

Purpose
-------
One declarative table describing the sensors this pipeline pulls scenes from,
so per-sensor constants (collection id, cloud-cover property, year range)
live in a single place rather than being restated inline once per
sensor-specific module.

Sensors
-------
=================  ==========  ======  ================================
Sensor             Period      Res.    Collection
=================  ==========  ======  ================================
Landsat 5 TM       1984-2011   10 m    LANDSAT/LT05/C02/T1_TOA
Landsat 7 ETM+     1999-       10 m    LANDSAT/LE07/C02/T1_TOA
Landsat 8 OLI      2013-       10 m    LANDSAT/LC08/C02/T1_TOA
Landsat 9 OLI-2    2021-       10 m    LANDSAT/LC09/C02/T1_TOA
Sentinel-2 MSI     2015-       10 m    COPERNICUS/S2_HARMONIZED
=================  ==========  ======  ================================

All five collections are Collection-2/harmonized **TOA reflectance** (not
atmospherically corrected Surface Reflectance) -- kept consistent across
sensors so downstream composites and indices are computed on the same
radiometric basis regardless of which sensor a scene came from. ``Res.`` is
the pipeline's uniform working resolution (every sensor resampled onto the
DEM's exact 10 m grid, see ``docs/decisions/unified_10m_grid.md``), not a
per-sensor native pixel size -- see :mod:`.band_specs` for the genuine
per-band native resolution each collection actually provides.

TOA vs. Surface Reflectance
----------------------------
This pipeline briefly switched to Surface Reflectance (SR) collections
(``docs/decisions/unified_10m_grid.md``'s original revision), then rolled
back to TOA (``docs/decisions/toa_rollback.md``): the SR-derived RGB visual
composites (:mod:`.composites`) were frequently unusable (black blotches,
whole-scene color casts) for QA/annotation, and maintaining two parallel
datasets (TOA + SR) was judged far too heavy in storage for no benefit the
project actually needed -- TOA already worked well visually and for the
classifier. Every collection here is therefore back to plain TOA reflectance,
already in the [0, 1] range as delivered by Earth Engine, no scale/offset
conversion needed.

Landsat TOA band names are plain ``B1``...``B8`` (``B8`` is the 15 m
panchromatic band, not stored -- see :mod:`.band_specs`, no pansharpening is
performed even though a pan band exists in the TOA product: every sensor
stays on the unified 10 m grid via plain bilinear resample, per
``docs/decisions/toa_rollback.md``). ``COPERNICUS/S2_HARMONIZED`` uses the
same ``B1``...``B12``/``B8A`` band names as the SR collection
(``COPERNICUS/S2_SR_HARMONIZED``) and the same x10000 scale, so nothing
about Sentinel-2's spec changes between TOA and SR.

Landsat 8 vs. Landsat 9
------------------------
Landsat 8 and Landsat 9 are kept as two fully separate :class:`SensorSpec`
rows (never merged into one "L8/9" entry), so downstream code and zarr groups
always know exactly which physical sensor a scene came from.

Landsat 7 SLC-off note
-----------------------
The Scan Line Corrector failed permanently in May 2003 and was never
repaired -- every L7 scene from 2003 onward (not just a 2003-2012 window)
has ~22% of its pixels missing in a fixed diagonal stripe pattern. This
pipeline has no gap-fill or per-pixel striping mask for L7 (a distinct,
unimplemented concern -- a possible future LSTM-based gap-filling
approach would target persistent cloud/shadow, not SLC striping), so L7 is only usable for
its pre-failure years: ``SensorSpec.last_year=2002``. This is enforced twice
-- once structurally via L7's own ``SensorSpec.last_year``, and again via the
explicit, unbounded :data:`SLC_OFF_FIRST_YEAR` guard in
:func:`is_year_allowed` -- so a future accidental widening of one cannot
silently reintroduce striped scenes. Neither is overridable via config.

Date range
----------
This pipeline wants whole-year coverage, not a fixed ablation-season window:
the date range for scene search is fully configurable per
:mod:`landscape_change_detection_pipeline.config` (``scene_download.date_start_mmdd`` /
``date_end_mmdd``) and defaults to the full calendar year
(``01-01``-``12-31``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

#: Default full calendar year window (inclusive), used unless config
#: overrides it. Kept broad and configurable rather than a fixed
#: ablation-season window.
DEFAULT_DATE_START_MMDD = "01-01"
DEFAULT_DATE_END_MMDD = "12-31"

#: Landsat 7 SLC failure (May 2003): permanent, never repaired. Every scene
#: from this year onward has ~22% of its pixels missing in a fixed diagonal
#: stripe pattern -- there is no "SLC-on again" year. Kept as an explicit,
#: unbounded guard (defense in depth alongside L7's own SensorSpec.last_year
#: cutoff below) so a future accidental widening of L7's SensorSpec cannot
#: silently reintroduce striped scenes. Hardcoded, not overridable via
#: config (see module docstring).
SLC_OFF_FIRST_YEAR = 2003


@dataclass(frozen=True)
class SensorSpec:
    """Everything the scene search/download needs to know about one sensor."""

    key: str
    name: str
    collection: str
    first_year: int
    last_year: Optional[int]  # None -> up to the current year
    native_resolution_m: float
    #: Image property holding cloud-cover percentage; differs per sensor
    #: family (Landsat: CLOUD_COVER, Sentinel-2: CLOUDY_PIXEL_PERCENTAGE).
    cloud_cover_property: str
    is_sentinel: bool = False
    #: Divisor turning this collection's raw band values into true [0, 1]
    #: reflectance: ``reflectance = raw / reflectance_scale_divisor +
    #: reflectance_scale_offset``. COPERNICUS/S2_HARMONIZED (TOA) stores
    #: reflectance x10000 as its raw pixel values with no additive offset,
    #: same convention as its SR counterpart (confirmed live, 2026-09-18: raw
    #: B5-B12 values over the pilot AOI ranged ~90-4400, not ~0.01-0.44) --
    #: dividing by this before quantizing to uint16 in
    #: gee_fetch.reflectance_to_uint16 (which itself multiplies by 10000) is
    #: what keeps stored values in the correct [0, 10000] code range. The
    #: four Landsat TOA collections (``T1_TOA``) need no scale/offset at
    #: all -- Earth Engine already delivers them as plain [0, 1] float
    #: reflectance -- so every Landsat ``SensorSpec`` below leaves this at
    #: its default (``1.0``, a no-op divisor).
    reflectance_scale_divisor: float = 1.0
    #: Additive term applied *after* dividing by ``reflectance_scale_divisor``
    #: (see above). Zero for every sensor under TOA (this field exists for
    #: the scale-**and**-offset convention Landsat Collection 2 Level-2
    #: *Surface Reflectance* used during this pipeline's brief SR period --
    #: see ``docs/decisions/toa_rollback.md`` -- and is kept here, defaulted
    #: to a no-op, rather than removed, in case SR is ever revisited).
    reflectance_scale_offset: float = 0.0

    def years(self, until: Optional[int] = None) -> list[int]:
        """Inclusive list of acquisition years for this sensor."""
        end = self.last_year if self.last_year is not None else (until or date.today().year)
        if until is not None:
            end = min(end, until)
        return list(range(self.first_year, end + 1))


SENSORS: dict[str, SensorSpec] = {
    "L5": SensorSpec(
        key="L5",
        name="Landsat 5 TM",
        collection="LANDSAT/LT05/C02/T1_TOA",
        first_year=1984,
        last_year=2011,
        native_resolution_m=30.0,
        cloud_cover_property="CLOUD_COVER",
    ),
    "L7": SensorSpec(
        key="L7",
        name="Landsat 7 ETM+",
        collection="LANDSAT/LE07/C02/T1_TOA",
        first_year=1999,
        # The Scan Line Corrector failed permanently in May 2003 and was
        # never repaired: every scene from 2003 onward carries ~22% missing
        # pixels in a fixed diagonal-stripe pattern, not just a 2003-2012
        # window. This pipeline has no SLC gap-fill/masking step (that is a
        # distinct, unimplemented concern -- a possible future LSTM-based
        # gap-filling approach would target persistent cloud/shadow, not SLC
        # striping), so L7 is usable only for its pre-failure years.
        last_year=2002,
        native_resolution_m=30.0,
        cloud_cover_property="CLOUD_COVER",
    ),
    "L8": SensorSpec(
        key="L8",
        name="Landsat 8 OLI",
        collection="LANDSAT/LC08/C02/T1_TOA",
        first_year=2013,
        last_year=None,
        native_resolution_m=30.0,
        cloud_cover_property="CLOUD_COVER",
    ),
    "L9": SensorSpec(
        key="L9",
        name="Landsat 9 OLI-2",
        collection="LANDSAT/LC09/C02/T1_TOA",
        first_year=2021,
        last_year=None,
        native_resolution_m=30.0,
        cloud_cover_property="CLOUD_COVER",
    ),
    "S2": SensorSpec(
        key="S2",
        name="Sentinel-2 MSI",
        collection="COPERNICUS/S2_HARMONIZED",
        first_year=2015,
        last_year=None,
        native_resolution_m=10.0,
        cloud_cover_property="CLOUDY_PIXEL_PERCENTAGE",
        is_sentinel=True,
        reflectance_scale_divisor=10000.0,
    ),
}

#: Processing order, oldest sensor first, so a resumed download processes
#: sensors in the same predictable order every time.
SENSOR_ORDER: tuple[str, ...] = ("L5", "L7", "L8", "L9", "S2")


def get_sensor(key: str) -> SensorSpec:
    """Look up a sensor spec by key, with a helpful error listing valid keys."""
    try:
        return SENSORS[key.upper()]
    except KeyError:
        raise KeyError(
            f"Unknown sensor '{key}'. Valid sensors: {', '.join(SENSOR_ORDER)}"
        ) from None


def is_year_allowed(sensor_key: str, year: int, skip_landsat7_slc_off: bool = True) -> bool:
    """Whether ``year`` should be downloaded for ``sensor_key``.

    Guards the (permanent, unbounded) Landsat 7 SLC-off period explicitly,
    so it stays excluded even if a caller widens the year range or a future
    edit relaxes L7's own ``SensorSpec.last_year``. ``skip_landsat7_slc_off``
    exists only to let tests exercise the guarded branch; it is not exposed
    via config.
    """
    spec = get_sensor(sensor_key)
    if year < spec.first_year:
        return False
    if spec.last_year is not None and year > spec.last_year:
        return False
    if skip_landsat7_slc_off and spec.key == "L7" and year >= SLC_OFF_FIRST_YEAR:
        return False
    return True


def date_bounds(
    year: int,
    start_mmdd: str = DEFAULT_DATE_START_MMDD,
    end_mmdd: str = DEFAULT_DATE_END_MMDD,
) -> tuple[str, str]:
    """Return the ``(start, end)`` ISO dates of the search window for a year.

    The end date is exclusive, as Earth Engine's ``filterDate`` expects. The
    default window is the full calendar year (01-01 through the day after
    12-31, i.e. the next year's 01-01).
    """
    start = f"{year}-{start_mmdd}"
    if end_mmdd == "12-31":
        end = f"{year + 1}-01-01"
    else:
        end = f"{year}-{end_mmdd}"
    return start, end


def sensors_for_years(
    years: list[int], skip_landsat7_slc_off: bool = True
) -> dict[str, list[int]]:
    """Map each sensor to the subset of ``years`` it can actually provide."""
    return {
        key: [y for y in years if is_year_allowed(key, y, skip_landsat7_slc_off)]
        for key in SENSOR_ORDER
    }
