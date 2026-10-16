"""DEM sources: MRDEM-30 DTM, ArcticDEM and Copernicus GLO-30, in a priority order.

Purpose
-------
Fetch an elevation model covering a tile's buffered analysis window, in the
tile's own CRS (the registry's ``crs`` column; ``TARGET_EPSG`` is only the
default), on the pipeline's 10 m reference grid, from the first usable source of
a configurable priority list (``dem.source_priority``, default
``arcticdem, mrdem, copernicus``). A source below ``min_valid_pct`` valid pixels
is completed pixel by pixel from the next ones:

1. **MRDEM-30 DTM** (NRCan CanElevation Series), read as a windowed COG read
   against its single nationwide mosaic COG on the public AWS bucket. This is
   a genuine bare-earth digital *terrain* model (lidar where NRCan has it,
   Copernicus GLO-30 terrain-corrected elsewhere), so it is not biased by
   forest canopy the way a raw InSAR DSM is.
2. **ArcticDEM** (PGC, 2 m DSM mosaic ``UMN/PGC/ArcticDEM/V4/2m_mosaic`` in
   Earth Engine, polar stereographic), averaged down to the 10 m grid by Earth
   Engine's pyramid. North of ~50 degN, with voids (cloud / featureless snow)
   that the next source fills.
3. **Copernicus GLO-30** (``COPERNICUS/DEM/GLO30_2024_1`` in Earth Engine,
   EGM2008 heights), global.

Vertical datums differ between sources (MRDEM CGVD2013, GLO-30 EGM2008,
ArcticDEM probably the WGS84 ellipsoid): when a source fills another, it is
shifted by the median difference over their common pixels so no step appears.

Inputs
------
- A bounding box in the tile's CRS (the tile's buffered analysis window).

Outputs
-------
- An ``xarray.DataArray`` (float32, dims ``y``/``x``, working CRS, NaN
  nodata), or ``None`` when the source has no usable coverage.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import rasterio
import rioxarray  # noqa: F401  -- registers the .rio accessor
import xarray as xr
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds as window_from_bounds

#: Working CRS for every raster product this pipeline stores.
TARGET_EPSG = 26910

#: MRDEM-30 is a single nationwide mosaic COG (not tiled), in EPSG:3979
#: (Canada Atlas Lambert), 30 m native resolution. Public bucket, no auth.
MRDEM_EPSG = 3979
MRDEM_RESOLUTION_M = 30
MRDEM_BASE_URL = "https://canelevation-dem.s3.ca-central-1.amazonaws.com/mrdem-30"
MRDEM_DTM_URL = f"{MRDEM_BASE_URL}/mrdem-30-dtm.tif"
MRDEM_DSM_URL = f"{MRDEM_BASE_URL}/mrdem-30-dsm.tif"

#: Copernicus GLO-30 fallback, via Earth Engine. GLO30 (undated) is
#: deprecated in favour of the dated collection as of 2026; using the dated
#: one directly avoids the deprecation warning on every fetch. GLO-30's own
#: *native* pixel spacing is still 30 m -- COPDEM_RESOLUTION_M is the
#: pipeline's output/reference-grid resolution (config-driven via
#: ``DemConfig.target_resolution_m``, default 10.0), not a claim that this
#: source resolves genuine 10 m detail; see
#: ``docs/decisions/unified_10m_grid.md``.
COPDEM_COLLECTION = "COPERNICUS/DEM/GLO30_2024_1"
COPDEM_RESOLUTION_M = 10.0

#: ArcticDEM 2 m mosaic (EPSG:3413 polar stereographic), via Earth Engine.
ARCTICDEM_ASSET = "UMN/PGC/ArcticDEM/V4/2m_mosaic"
ARCTICDEM_BAND = "elevation"

SOURCE_LABELS = {
    "mrdem": "MRDEM-30 DTM (NRCan CanElevation, AWS COG)",
    "arcticdem": f"ArcticDEM 2 m mosaic ({ARCTICDEM_ASSET}, GEE)",
    "copernicus": "Copernicus DEM GLO-30 (COPERNICUS/DEM/GLO30_2024_1, GEE)",
}
DEFAULT_SOURCE_PRIORITY = ("arcticdem", "mrdem", "copernicus")

#: Value masked pixels are unmasked to before an Earth Engine download, then
#: turned back into NaN (an elevation is never exactly this).
_EE_NODATA = -9999.0

#: Minimum fraction of finite pixels in the fetched window for a source to be
#: considered usable. Deliberately low: this only screens out windows that
#: are essentially empty (e.g. a window straddling the extent's edge or a gap
#: in coverage), not a quality gate on the values themselves.
MIN_VALID_PCT = 1.0


def configure_gdal_for_public_cogs() -> None:
    """Set the GDAL environment for anonymous, range-request COG access.

    ``GDAL_HTTP_TIMEOUT``/``GDAL_HTTP_CONNECTTIMEOUT`` are the critical ones:
    without them, GDAL's underlying curl handle has **no timeout at all** on
    a stalled or slow-to-respond HTTP range request, so a single flaky
    request against a public S3/AWS bucket hangs the whole process
    indefinitely -- confirmed live in a previous project (a Copernicus
    DEM fetch stalled with 0% CPU for several minutes, invisible to
    Python-level signal handling). ``GDAL_HTTP_MAX_RETRY``/
    ``GDAL_HTTP_RETRY_DELAY`` already existed but cannot help if the initial
    connection itself never times out to trigger a retry.
    """
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
    os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "10")
    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "30")
    os.environ.setdefault("VSI_CACHE", "TRUE")
    os.environ.setdefault("VSI_CACHE_SIZE", "50000000")


def _wrap_array(arr: np.ndarray, transform, epsg: int) -> xr.DataArray:
    """Wrap a numpy array + affine transform into a CRS-aware DataArray."""
    height, width = arr.shape
    xs = transform.c + transform.a * (np.arange(width) + 0.5)
    ys = transform.f + transform.e * (np.arange(height) + 0.5)
    da = xr.DataArray(arr, coords={"y": ys, "x": xs}, dims=("y", "x"))
    da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    da.rio.write_crs(epsg, inplace=True)
    da.rio.write_transform(transform, inplace=True)
    da.rio.write_nodata(np.nan, inplace=True)
    return da


def valid_fraction_pct(da: xr.DataArray) -> float:
    """Percentage of finite pixels in ``da``."""
    values = np.asarray(da.values)
    return 100.0 * float(np.isfinite(values).mean()) if values.size else 0.0


def _array_from_window(
    src: rasterio.DatasetReader,
    bounds: tuple[float, float, float, float],
) -> Optional[xr.DataArray]:
    """Read a windowed slice of an open dataset into a georeferenced DataArray."""
    window = window_from_bounds(*bounds, transform=src.transform)
    window = window.round_offsets().round_lengths()
    if window.width <= 0 or window.height <= 0:
        return None

    data = src.read(1, window=window, masked=True, boundless=True, fill_value=np.nan)
    arr = np.asarray(data.filled(np.nan), dtype=np.float32)
    if arr.size == 0:
        return None

    if src.nodata is not None and np.isfinite(src.nodata):
        arr[arr == np.float32(src.nodata)] = np.nan

    transform = src.window_transform(window)
    return _wrap_array(arr, transform, src.crs.to_epsg() or MRDEM_EPSG)


def load_mrdem_dtm(
    bounds: tuple[float, float, float, float],
    src_crs: int = TARGET_EPSG,
    url: str = MRDEM_DTM_URL,
    target_resolution_m: float = MRDEM_RESOLUTION_M,
) -> Optional[xr.DataArray]:
    """Load MRDEM-30's DTM asset over ``bounds`` (in ``src_crs``).

    MRDEM-30 is one nationwide mosaic COG in EPSG:3979, so this is a single
    windowed read -- no STAC search or per-tile mosaicking needed, unlike a
    genuinely tiled product. Returns ``None`` if the window has no usable
    coverage. The read result is reprojected to ``src_crs`` at
    ``target_resolution_m`` -- the pipeline's reference-grid resolution
    (``DemConfig.target_resolution_m``, default 10.0), not MRDEM-30's own
    native 30 m pixel spacing; genuine elevation detail is still only ~30 m,
    see ``docs/decisions/unified_10m_grid.md``.
    """
    configure_gdal_for_public_cogs()

    bounds_3979 = (
        transform_bounds(f"EPSG:{src_crs}", f"EPSG:{MRDEM_EPSG}", *bounds, densify_pts=21)
        if src_crs != MRDEM_EPSG
        else bounds
    )

    with rasterio.open(url) as src:
        dem = _array_from_window(src, bounds_3979)
    if dem is None:
        return None

    if dem.rio.crs.to_epsg() != src_crs or target_resolution_m != MRDEM_RESOLUTION_M:
        # An explicit dst transform/shape pins the reprojected grid to the
        # tile's own window bounds: passing only resolution/dst_crs sizes the
        # output to the *rotated* footprint of the source window (EPSG:3979
        # -> EPSG:26910 is not axis-aligned), which comes out several times
        # larger than the window on each side and leaves most of the array
        # outside it once masked -- confirmed live (a 501x501 window came
        # back ~32-56% valid before this fix, depending on rotation).
        dst_transform, dst_width, dst_height = _window_transform(bounds, target_resolution_m)
        dem = dem.rio.reproject(
            dst_crs=f"EPSG:{src_crs}",
            transform=dst_transform,
            shape=(dst_height, dst_width),
            resampling=Resampling.bilinear,
            nodata=np.nan,
        )
        dem.rio.write_nodata(np.nan, inplace=True)
    return dem.astype("float32")


def _window_transform(
    bounds: tuple[float, float, float, float], resolution_m: float
):
    """North-up affine transform + (width, height) covering ``bounds`` exactly."""
    from affine import Affine

    minx, miny, maxx, maxy = bounds
    width = max(1, round((maxx - minx) / resolution_m))
    height = max(1, round((maxy - miny) / resolution_m))
    transform = Affine(resolution_m, 0.0, minx, 0.0, -resolution_m, maxy)
    return transform, width, height


def _compute_pixels_window(
    image,
    bounds: tuple[float, float, float, float],
    src_crs: int,
    target_resolution_m: float,
) -> Optional[xr.DataArray]:
    """Download ``image`` over ``bounds`` on the exact ``src_crs`` grid at
    ``target_resolution_m`` with ``ee.data.computePixels``.

    The grid is built directly in ``src_crs`` (never a reprojected lon/lat
    rectangle handed to Earth Engine server-side) -- the same fix documented in
    ``tiles/registry.py`` for the ~1.5 km grid-shift bug at high latitude.
    ``Image.getDownloadURL`` returns an authenticated ``.../:getPixels``
    endpoint that GDAL's ``/vsicurl/`` cannot call (confirmed live: 404), so the
    request goes through ``computePixels``, which carries the session's
    credentials. Masked pixels come back as :data:`_EE_NODATA` -> NaN.
    """
    import io

    import ee

    dst_transform, dst_width, dst_height = _window_transform(bounds, target_resolution_m)
    try:
        tiff_bytes = ee.data.computePixels(
            {
                "expression": image.toFloat().unmask(_EE_NODATA),
                "fileFormat": "GEO_TIFF",
                "grid": {
                    "dimensions": {"width": dst_width, "height": dst_height},
                    "affineTransform": {
                        "scaleX": dst_transform.a,
                        "shearX": dst_transform.b,
                        "translateX": dst_transform.c,
                        "shearY": dst_transform.d,
                        "scaleY": dst_transform.e,
                        "translateY": dst_transform.f,
                    },
                    "crsCode": f"EPSG:{src_crs}",
                },
            }
        )
    except Exception:
        return None

    with rasterio.open(io.BytesIO(tiff_bytes)) as src:
        data = np.asarray(src.read(1), dtype=np.float32)
        data[data == np.float32(_EE_NODATA)] = np.nan
        if src.nodata is not None and np.isfinite(src.nodata):
            data[data == np.float32(src.nodata)] = np.nan
        result = _wrap_array(data, src.transform, src.crs.to_epsg() or src_crs)

    if result.rio.crs.to_epsg() != src_crs:
        result = result.rio.reproject(
            dst_crs=f"EPSG:{src_crs}",
            transform=dst_transform,
            shape=(dst_height, dst_width),
            resampling=Resampling.bilinear,
            nodata=np.nan,
        )
        result.rio.write_nodata(np.nan, inplace=True)
    return result.astype("float32")


def load_copernicus_dem_gee(
    bounds: tuple[float, float, float, float],
    src_crs: int = TARGET_EPSG,
    collection: str = COPDEM_COLLECTION,
    target_resolution_m: float = COPDEM_RESOLUTION_M,
) -> Optional[xr.DataArray]:
    """Load Copernicus GLO-30 over ``bounds`` (in ``src_crs``) via Earth Engine.

    ``target_resolution_m`` is the pipeline's reference-grid resolution
    (``DemConfig.target_resolution_m``); GLO-30's own native pixel spacing is
    still 30 m, so a finer target only changes pixel alignment, not genuine
    resolving power (see ``docs/decisions/unified_10m_grid.md``).
    """
    import ee

    image = ee.ImageCollection(collection).select("DEM").mosaic()
    return _compute_pixels_window(image, bounds, src_crs, target_resolution_m)


def load_arcticdem_gee(
    bounds: tuple[float, float, float, float],
    src_crs: int = TARGET_EPSG,
    asset: str = ARCTICDEM_ASSET,
    target_resolution_m: float = 10.0,
) -> Optional[xr.DataArray]:
    """Load the ArcticDEM 2 m mosaic over ``bounds`` (in ``src_crs``) via Earth Engine.

    The mosaic is requested directly on the ``target_resolution_m`` grid: Earth
    Engine serves the coarser request from the asset's MEAN pyramid, i.e. the
    mean of the 2 m pixels. Returns ``None`` outside ArcticDEM's footprint. A
    DSM (canopy and ice surface included).
    """
    import ee

    image = ee.Image(asset).select(ARCTICDEM_BAND)
    return _compute_pixels_window(image, bounds, src_crs, target_resolution_m)


def _load_source(
    name: str,
    bounds: tuple[float, float, float, float],
    src_crs: int,
    target_resolution_m: float,
) -> Optional[xr.DataArray]:
    """One named source, ``None`` on any failure or missing coverage."""
    if name not in SOURCE_LABELS:
        raise ValueError(f"Unknown DEM source {name!r}; expected one of {DEFAULT_SOURCE_PRIORITY}")
    try:
        if name == "mrdem":
            return load_mrdem_dtm(bounds, src_crs=src_crs, target_resolution_m=target_resolution_m)
        if name == "arcticdem":
            return load_arcticdem_gee(bounds, src_crs=src_crs, target_resolution_m=target_resolution_m)
        return load_copernicus_dem_gee(bounds, src_crs=src_crs, target_resolution_m=target_resolution_m)
    except Exception:
        return None


#: Fewest common pixels needed to estimate the vertical offset between two sources.
_MIN_OVERLAP_PX = 100


def fill_gaps(
    base: xr.DataArray, other: xr.DataArray
) -> tuple[xr.DataArray, int, Optional[float]]:
    """Fill ``base``'s NaN pixels from ``other`` (same grid).

    ``other`` is first shifted by the median of ``base - other`` over their
    common valid pixels (vertical datum / bias difference), when there are at
    least :data:`_MIN_OVERLAP_PX` of them. Returns ``(filled, n_filled,
    offset_m)``; ``offset_m`` is ``None`` if no shift could be estimated.
    """
    a = np.asarray(base.values, dtype=np.float32)
    b = np.asarray(other.values, dtype=np.float32)
    common = np.isfinite(a) & np.isfinite(b)
    offset = float(np.median(a[common] - b[common])) if common.sum() >= _MIN_OVERLAP_PX else None
    gap = ~np.isfinite(a) & np.isfinite(b)
    out = np.where(gap, b + np.float32(offset or 0.0), a).astype(np.float32)
    return base.copy(data=out), int(gap.sum()), offset


def load_best_dem(
    bounds: tuple[float, float, float, float],
    src_crs: int = TARGET_EPSG,
    use_copernicus_fallback: bool = True,
    force_copernicus: bool = False,
    target_resolution_m: float = 10.0,
    source_priority: Optional[Sequence[str]] = None,
    min_valid_pct: float = 90.0,
) -> tuple[Optional[xr.DataArray], str, int, float]:
    """Walk the sources in ``source_priority`` and return the assembled DEM.

    The first source with at least :data:`MIN_VALID_PCT` valid pixels is taken;
    while the result is below ``min_valid_pct`` valid, the next sources fill its
    gaps pixel by pixel (see :func:`fill_gaps`). Returns ``(dem, source_label,
    base_resolution_m, valid_pct)``; ``dem`` is ``None`` when no source yields a
    usable window. ``base_resolution_m`` is the *output* grid resolution
    (``target_resolution_m``); the sources' native spacing is 30 m (MRDEM,
    GLO-30) or 2 m (ArcticDEM), see ``docs/decisions/unified_10m_grid.md``.

    ``force_copernicus`` uses Copernicus alone; ``use_copernicus_fallback=False``
    drops it from the list (fail loud rather than substitute a canopy-biased
    source).
    """
    order = list(source_priority) if source_priority else list(DEFAULT_SOURCE_PRIORITY)
    if force_copernicus:
        order = ["copernicus"]
    elif not use_copernicus_fallback:
        order = [n for n in order if n != "copernicus"]

    merged: Optional[xr.DataArray] = None
    parts: list[str] = []
    for name in order:
        dem = _load_source(name, bounds, src_crs, target_resolution_m)
        if dem is None or valid_fraction_pct(dem) < MIN_VALID_PCT:
            continue
        if merged is None:
            merged, parts = dem, [SOURCE_LABELS[name]]
        else:
            merged, n_filled, offset = fill_gaps(merged, dem)
            if n_filled:
                shift = f"shifted {offset:+.1f} m" if offset is not None else "unshifted, little overlap"
                parts.append(f"{SOURCE_LABELS[name]} [gap fill: {n_filled} px, {shift}]")
        if valid_fraction_pct(merged) >= min_valid_pct:
            break

    if merged is None:
        return None, "", 0, 0.0
    return merged, " + ".join(parts), target_resolution_m, valid_fraction_pct(merged)
