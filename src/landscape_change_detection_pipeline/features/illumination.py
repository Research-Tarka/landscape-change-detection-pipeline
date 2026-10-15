"""Terrain illumination channels: local sun incidence and terrain shadow.

Purpose
-------
A dark, steep, shaded slope (forest, rock, snow in shade) is spectrally close to
open water, and a spectral-only classifier confuses them. What separates them is
geometry the sensor does not see directly: where the sun was, and whether the
terrain faces it or hides from it. This module derives two scene-level, physics-
based channels from the DEM layers already in the feature stack and the
acquisition date/time, with no extra download:

``illum_cos``
    Cosine of the angle between the sun and the local surface normal
    (``IL`` of the SCS+C topographic correction, see
    :mod:`landscape_change_detection_pipeline.features.topographic_correction`).
    Negative / near zero = facing away from the sun.
``terrain_shadow``
    1 where the cell is shaded (self shadow: ``illum_cos`` below
    :data:`SELF_SHADOW_COS`, or cast shadow: terrain along the sun azimuth
    rises above the sun ray), 0 elsewhere.

Sun position comes from the NOAA low-precision series: accurate to a fraction
of a degree, enough to say "this slope is in shade". Sentinel-2 ids carry the
sensing time; Landsat ids only the date, so the overpass is taken at ~10:10
local mean solar time (sun-synchronous orbit, within ~30 min for L5-L9).

Conventions: azimuth clockwise from north, elevation above the horizon, aspect
= compass direction the slope faces (0 = N), as in
:func:`landscape_change_detection_pipeline.dem.processing.compute_slope_aspect`. The scene grid must be
north-up in a projection whose grid north is ~true north (UTM), exactly like
every tile of this pipeline.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np

ILLUMINATION_FEATURE_NAMES: tuple[str, ...] = ("illum_cos", "terrain_shadow")

#: DEM layers the channels are derived from (names as in ``features.dem_layer_names``).
ILLUMINATION_DEM_LAYERS: tuple[str, ...] = ("elevation", "slope", "aspect")

#: Landsat 5/7/8/9 cross the equator at ~10:00-10:15 local mean solar time.
LANDSAT_LOCAL_SOLAR_HOUR = 10.17
#: ``illum_cos`` below this = self-shaded slope.
SELF_SHADOW_COS = 0.25
#: How far along the sun azimuth a cast shadow is searched (metres).
CAST_SHADOW_MAX_DIST_M = 2500.0


def scene_datetime_utc(sensor: str, scene_id: str, lon_deg: float) -> datetime:
    """Acquisition instant (UTC) from a pipeline scene id (see module docstring)."""
    if sensor.lower() == "s2":
        return datetime.strptime(scene_id[:15], "%Y%m%dT%H%M%S")
    date = datetime.strptime(scene_id[-8:], "%Y%m%d")
    return date + timedelta(hours=(LANDSAT_LOCAL_SOLAR_HOUR - lon_deg / 15.0) % 24.0)


def sun_position(dt_utc: datetime, lat_deg: float, lon_deg: float) -> tuple[float, float]:
    """``(elevation_deg, azimuth_deg)`` of the sun (NOAA low-precision series)."""
    doy = dt_utc.timetuple().tm_yday
    hour = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0
    g = 2.0 * math.pi / 365.0 * (doy - 1 + (hour - 12.0) / 24.0)
    eqtime = 229.18 * (
        0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
        - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g)
    )
    decl = (
        0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
        + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g)
    )
    tst = (hour * 60.0 + eqtime + 4.0 * lon_deg) % 1440.0
    ha = math.radians(tst / 4.0 - 180.0)
    lat = math.radians(lat_deg)
    cos_zen = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)
    zen = math.acos(max(-1.0, min(1.0, cos_zen)))
    az = math.atan2(math.sin(ha), math.cos(ha) * math.sin(lat) - math.tan(decl) * math.cos(lat)) + math.pi
    return 90.0 - math.degrees(zen), math.degrees(az) % 360.0


def cos_incidence(slope_deg, aspect_deg, sun_elev_deg: float, sun_az_deg: float) -> np.ndarray:
    """Cosine of the sun-to-surface-normal angle. Flat cells (NaN aspect) get ``sin(sun_elevation)``."""
    zen = math.radians(90.0 - sun_elev_deg)
    slope = np.radians(np.nan_to_num(np.asarray(slope_deg, dtype=np.float32), nan=0.0))
    aspect = np.radians(np.nan_to_num(np.asarray(aspect_deg, dtype=np.float32), nan=0.0))
    return (
        math.cos(zen) * np.cos(slope)
        + math.sin(zen) * np.sin(slope) * np.cos(aspect - math.radians(sun_az_deg))
    ).astype(np.float32)


def cast_shadow(dem: np.ndarray, res_m: float, sun_elev_deg: float, sun_az_deg: float,
                max_dist_m: float = CAST_SHADOW_MAX_DIST_M) -> np.ndarray:
    """Cells hidden from the sun by terrain further along the sun azimuth.

    Horizon test: a cell is shaded when some cell toward the sun rises above the
    sun ray, ``z_other - z > dist * tan(sun_elevation)``. Near the scene border
    only terrain inside the scene can shade a cell, so detection there is
    conservative (under-detects).
    """
    if sun_elev_deg <= 0.5:
        return np.isfinite(dem)  # sun at/below the horizon: everything is dark
    h, w = dem.shape
    east, north = math.sin(math.radians(sun_az_deg)), math.cos(math.radians(sun_az_deg))
    tan_e = math.tan(math.radians(sun_elev_deg))
    z = np.nan_to_num(dem.astype(np.float32), nan=-1e4)
    shaded = np.zeros(dem.shape, dtype=bool)
    seen: set[tuple[int, int]] = set()
    for k in range(1, int(max_dist_m / res_m) + 1):
        dx, dy = int(round(k * east)), int(round(-k * north))  # column / row offset toward the sun
        if (dx, dy) in seen or (dx == 0 and dy == 0):
            continue
        seen.add((dx, dy))
        if abs(dx) >= w or abs(dy) >= h:
            break
        r0, r1 = max(0, -dy), min(h, h - dy)
        c0, c1 = max(0, -dx), min(w, w - dx)
        other = z[r0 + dy : r1 + dy, c0 + dx : c1 + dx]
        here = z[r0:r1, c0:c1]
        shaded[r0:r1, c0:c1] |= (other - here) > math.hypot(dx, dy) * res_m * tan_e
    return shaded & np.isfinite(dem)


def illumination_stack(
    sensor: str,
    scene_id: str,
    transform,
    crs,
    elevation: np.ndarray,
    slope: np.ndarray,
    aspect: np.ndarray,
) -> np.ndarray:
    """``(2, H, W)`` float32 stack ``[illum_cos, terrain_shadow]`` for one scene.

    ``elevation`` (m), ``slope`` and ``aspect`` (degrees, NaN where flat) are the
    scene-grid DEM layers; ``transform``/``crs`` the scene grid georeferencing
    (``Affine`` or 6-list, as stored in the training cache).
    """
    from affine import Affine

    from landscape_change_detection_pipeline.features.spectral_indices import centroid_latlon_norm

    aff = transform if isinstance(transform, Affine) else Affine(*[float(v) for v in transform[:6]])
    shape = tuple(elevation.shape)
    lat_n, lon_n = centroid_latlon_norm(aff, crs, shape)
    lat, lon = float(lat_n) * 90.0, float(lon_n) * 180.0
    elev, az = sun_position(scene_datetime_utc(sensor, scene_id, lon), lat, lon)

    il = cos_incidence(slope, aspect, elev, az)
    shaded = cast_shadow(elevation, abs(float(aff.a)), elev, az) | (il < SELF_SHADOW_COS)
    out = np.stack([np.clip(il, -1.0, 1.0), shaded.astype(np.float32)], axis=0)
    # no elevation -> no geometry: leave the channels neutral instead of inventing shade
    out[:, ~np.isfinite(elevation)] = 0.0
    return out.astype(np.float32)
