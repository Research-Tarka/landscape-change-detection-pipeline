"""DEM sources: MRDEM-30 DTM (primary) with a Copernicus GLO-30 fallback.

Purpose
-------
Fetch an elevation model covering a tile's buffered analysis window, in the
pipeline's working CRS (``EPSG:26910``),
from one of two sources:

1. **MRDEM-30 DTM** (NRCan CanElevation Series), read as a windowed COG read
   against its single nationwide mosaic COG on the public AWS bucket. This is
   a genuine bare-earth digital *terrain* model (lidar where NRCan has it,
   Copernicus GLO-30 terrain-corrected elsewhere), so it is not biased by
   forest canopy the way a raw InSAR DSM is.
2. **Copernicus GLO-30** (``COPERNICUS/DEM/GLO30_2024_1`` in Earth Engine),
   used only if MRDEM has no usable coverage for the window.

Inputs
------
- A bounding box in the pipeline's working CRS (the tile's buffered analysis
  window).

Outputs
-------
- An ``xarray.DataArray`` (float32, dims ``y``/``x``, working CRS, NaN
  nodata), or ``None`` when the source has no usable coverage.
"""

from __future__ import annotations

import os
from typing import Optional

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
    coverage. The read result is reprojected to :data:`TARGET_EPSG` at
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

    if dem.rio.crs.to_epsg() != TARGET_EPSG or target_resolution_m != MRDEM_RESOLUTION_M:
        # An explicit dst transform/shape pins the reprojected grid to the
        # tile's own window bounds: passing only resolution/dst_crs sizes the
        # output to the *rotated* footprint of the source window (EPSG:3979
        # -> EPSG:26910 is not axis-aligned), which comes out several times
        # larger than the window on each side and leaves most of the array
        # outside it once masked -- confirmed live (a 501x501 window came
        # back ~32-56% valid before this fix, depending on rotation).
        dst_transform, dst_width, dst_height = _window_transform(bounds, target_resolution_m)
        dem = dem.rio.reproject(
            dst_crs=f"EPSG:{TARGET_EPSG}",
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


def load_copernicus_dem_gee(
    bounds: tuple[float, float, float, float],
    src_crs: int = TARGET_EPSG,
    collection: str = COPDEM_COLLECTION,
    target_resolution_m: float = COPDEM_RESOLUTION_M,
) -> Optional[xr.DataArray]:
    """Load Copernicus GLO-30 over ``bounds`` (in ``src_crs``) via Earth Engine.

    Built directly in ``src_crs`` (never as a reprojected lon/lat rectangle
    handed to Earth Engine server-side) -- the same fix documented in
    ``tiles/registry.py`` for the ~1.5 km grid-shift bug at this AOI's
    latitude applies equally here.

    ``target_resolution_m`` is the pipeline's reference-grid resolution
    (``DemConfig.target_resolution_m``); GLO-30's own native pixel spacing is
    still 30 m, so a finer target only changes pixel alignment, not genuine
    resolving power (see ``docs/decisions/unified_10m_grid.md``).

    ``Image.getDownloadURL`` returns an authenticated
    ``.../:getPixels`` endpoint, not a plain public URL: GDAL's
    ``/vsicurl/`` has no OAuth token to attach to it, so the request must go
    through ``ee.data.computePixels`` (which carries the session's
    credentials) rather than being opened directly with rasterio -- confirmed
    live (a bare ``rasterio.open("/vsicurl/" + url)`` 404s: the endpoint
    exists but rejects the unauthenticated GET).
    """
    import io

    import ee

    geom = ee.Geometry.Rectangle(
        list(bounds), proj=f"EPSG:{src_crs}", evenOdd=True, geodesic=False
    )
    dst_transform, dst_width, dst_height = _window_transform(bounds, target_resolution_m)
    image = ee.ImageCollection(collection).select("DEM").mosaic().clip(geom)

    try:
        tiff_bytes = ee.data.computePixels(
            {
                "expression": image,
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
        arr = src.read(1, masked=True)
        data = np.asarray(arr.filled(np.nan), dtype=np.float32)
        if src.nodata is not None and np.isfinite(src.nodata):
            data[data == np.float32(src.nodata)] = np.nan
        result = _wrap_array(data, src.transform, src.crs.to_epsg() or src_crs)

    if result.rio.crs.to_epsg() != TARGET_EPSG or target_resolution_m != COPDEM_RESOLUTION_M:
        dst_transform, dst_width, dst_height = _window_transform(bounds, target_resolution_m)
        result = result.rio.reproject(
            dst_crs=f"EPSG:{TARGET_EPSG}",
            transform=dst_transform,
            shape=(dst_height, dst_width),
            resampling=Resampling.bilinear,
            nodata=np.nan,
        )
        result.rio.write_nodata(np.nan, inplace=True)
    return result.astype("float32")


def load_best_dem(
    bounds: tuple[float, float, float, float],
    src_crs: int = TARGET_EPSG,
    use_copernicus_fallback: bool = True,
    force_copernicus: bool = False,
    target_resolution_m: float = 10.0,
) -> tuple[Optional[xr.DataArray], str, int, float]:
    """Try MRDEM-30 DTM first, then Copernicus GLO-30.

    Returns ``(dem, source_label, base_resolution_m, valid_pct)``. ``dem`` is
    ``None`` when neither source yields a window with at least
    :data:`MIN_VALID_PCT` finite pixels. ``base_resolution_m`` is the
    *output* grid resolution (``target_resolution_m``, normally
    ``DemConfig.target_resolution_m``), not necessarily the source's own
    native pixel spacing -- both MRDEM-30 and Copernicus GLO-30 are natively
    30 m; see ``docs/decisions/unified_10m_grid.md``.

    ``force_copernicus`` skips MRDEM entirely and goes straight to
    Copernicus GLO-30, so the fallback path can be exercised deliberately.
    """
    if not force_copernicus:
        try:
            dem = load_mrdem_dtm(bounds, src_crs=src_crs, target_resolution_m=target_resolution_m)
        except Exception:
            dem = None

        if dem is not None:
            pct = valid_fraction_pct(dem)
            if pct >= MIN_VALID_PCT:
                return dem, "MRDEM-30 DTM (NRCan CanElevation, AWS COG)", target_resolution_m, pct

    if not use_copernicus_fallback and not force_copernicus:
        return None, "", 0, 0.0

    try:
        dem = load_copernicus_dem_gee(bounds, src_crs=src_crs, target_resolution_m=target_resolution_m)
    except Exception:
        dem = None

    if dem is None:
        return None, "", 0, 0.0

    pct = valid_fraction_pct(dem)
    if pct < MIN_VALID_PCT:
        return None, "", 0, pct
    return dem, "Copernicus DEM GLO-30 (COPERNICUS/DEM/GLO30_2024_1, GEE)", target_resolution_m, pct
