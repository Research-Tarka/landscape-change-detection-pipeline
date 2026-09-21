"""DEM post-processing: masking to a tile's window, slope, and aspect.

Purpose
-------
Turn a raw source DEM window (MRDEM-30 DTM or Copernicus GLO-30 fallback,
:mod:`.sources`) into the per-tile products the pipeline stores: elevation
masked to the tile's buffered analysis window, oriented north-up, plus slope
and aspect derived from it.

Inputs
------
- A source DEM ``xarray.DataArray`` in the working CRS (EPSG:26910).
- The tile's buffered analysis-window bounds (also EPSG:26910).

Outputs
-------
- Elevation, slope (degrees), and aspect (degrees, 0=north, clockwise)
  ``DataArray``\\ s on one common grid.

Masking rules
-------------
Outside the window, NaN;
inside but non-finite in the source, NaN -- there is no reasonable 0.0 fill for
a mainland tile the way there might be for some other domain, so this stays
NaN rather than becoming a flagged zero-fill.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import rioxarray  # noqa: F401  -- registers the .rio accessor
import xarray as xr

from .sources import TARGET_EPSG, _wrap_array


def orient_north_up(dem: xr.DataArray) -> xr.DataArray:
    """Flip ``dem`` so the y axis decreases with row index (north-up).

    A windowed read of a source raster already stored north-up (as both
    MRDEM-30 and Copernicus GLO-30 are) comes out north-up already; this is a
    defensive no-op in that case and a real flip if a future source is not.
    """
    transform = dem.rio.transform(recalc=False)
    if transform.e < 0:
        return dem
    flipped = dem.isel(y=slice(None, None, -1))
    flipped.rio.write_transform(flipped.rio.transform(recalc=True), inplace=True)
    return flipped


def clip_to_bounds(
    dem: xr.DataArray, bounds: tuple[float, float, float, float]
) -> xr.DataArray:
    """Mask ``dem`` to ``bounds`` (in the array's own CRS): outside -> NaN."""
    minx, miny, maxx, maxy = bounds
    xs = dem["x"].values
    ys = dem["y"].values
    inside_x = (xs >= minx) & (xs <= maxx)
    inside_y = (ys >= miny) & (ys <= maxy)
    inside = inside_y[:, None] & inside_x[None, :]

    values = np.asarray(dem.values, dtype=np.float32)
    out = np.where(inside, values, np.nan).astype(np.float32)
    return _wrap_array(out, dem.rio.transform(recalc=False), TARGET_EPSG)


def compute_slope_aspect(dem: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Derive slope (degrees) and aspect (degrees, 0=N clockwise) from ``dem``.

    Uses the standard Horn (1981) 3x3 kernel (the same one GDAL's
    ``gdaldem slope``/``aspect`` use), computed directly with numpy gradients
    rather than shelling out, since the array is already in memory. NaN
    propagates: any 3x3 neighbourhood touching a NaN elevation produces NaN
    slope/aspect at that cell, consistent with the elevation mask's own edges.
    """
    transform = dem.rio.transform(recalc=False)
    res_x, res_y = abs(float(transform.a)), abs(float(transform.e))

    z = np.asarray(dem.values, dtype=np.float64)
    padded = np.pad(z, 1, mode="edge")
    nan_mask = ~np.isfinite(z)

    # Horn's weighted 3x3 kernel for dz/dx, dz/dy.
    z1, z2, z3 = padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:]
    z4, z6 = padded[1:-1, :-2], padded[1:-1, 2:]
    z7, z8, z9 = padded[2:, :-2], padded[2:, 1:-1], padded[2:, 2:]

    dzdx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8 * res_x)
    dzdy = ((z7 + 2 * z8 + z9) - (z1 + 2 * z2 + z3)) / (8 * res_y)

    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))

    aspect = np.degrees(np.arctan2(dzdy, -dzdx))
    aspect = np.where(aspect < 0, 90.0 - aspect, np.where(aspect > 90.0, 360.0 - aspect + 90.0, 90.0 - aspect))
    # Flat cells (both gradients ~0) have no meaningful aspect.
    flat = (np.abs(dzdx) < 1e-9) & (np.abs(dzdy) < 1e-9)
    aspect = np.where(flat, np.nan, aspect)

    # Any cell whose 3x3 neighbourhood touches a source NaN is invalid.
    invalid = _dilate_nan_mask(nan_mask)
    slope = np.where(invalid, np.nan, slope).astype(np.float32)
    aspect = np.where(invalid, np.nan, aspect).astype(np.float32)

    slope_da = _wrap_array(slope, transform, TARGET_EPSG)
    aspect_da = _wrap_array(aspect, transform, TARGET_EPSG)
    return slope_da, aspect_da


def _dilate_nan_mask(mask: np.ndarray) -> np.ndarray:
    """3x3 binary dilation without a scipy dependency (small, local kernel)."""
    padded = np.pad(mask, 1, mode="edge")
    out = np.zeros_like(mask)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def build_tile_dem_products(
    dem_raw: xr.DataArray,
    window_bounds: tuple[float, float, float, float],
) -> dict[str, xr.DataArray]:
    """Full per-tile pipeline: orient, mask to window, derive slope/aspect.

    Returns ``{"elevation": ..., "slope": ..., "aspect": ...}``, all on the
    same grid (shape/transform), float32, NaN nodata.
    """
    oriented = orient_north_up(dem_raw)
    masked = clip_to_bounds(oriented, window_bounds)
    slope, aspect = compute_slope_aspect(masked)
    return {"elevation": masked, "slope": slope, "aspect": aspect}


def valid_fraction_pct(da: xr.DataArray) -> float:
    """Percentage of finite pixels in ``da`` (re-exported for callers)."""
    values = np.asarray(da.values)
    return 100.0 * float(np.isfinite(values).mean()) if values.size else 0.0


def elevation_stats(da: xr.DataArray) -> dict[str, Optional[float]]:
    """Min/max/mean elevation over finite pixels, for sanity-check logging."""
    values = np.asarray(da.values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
    }
