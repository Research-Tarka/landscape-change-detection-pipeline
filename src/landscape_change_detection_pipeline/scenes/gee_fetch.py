"""Per-scene pixel fetch: in-memory GeoTIFF download, reprojected at request time.

Purpose
-------
For one scene already selected by discovery, fetch every band's
pixels over a tile's analysis window, already reprojected to the pipeline's
working CRS **at request time** -- ``region`` + ``scale`` + ``crs`` passed to
``getDownloadURL``, never ``crsTransform``/``dimensions`` (see
``tiles/registry.py``'s module docstring for the confirmed ~1.5 km
grid-shift bug this avoids at this AOI's latitude: requesting a fixed pixel
grid via ``crsTransform`` bypasses Earth Engine's own reprojection logic and
can silently return pixels aligned to the wrong grid when the source and
target CRS differ).

Bytes are fetched straight into a ``rasterio.io.MemoryFile`` (``/vsimem``) --
no temp files ever touch disk.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

#: requests' connect/read timeout for one band-group download. A single scene
#: fetch is a handful of these, not a hot loop, so a generous bound is fine;
#: it exists only to fail loudly on a stalled connection rather than hang.
DOWNLOAD_TIMEOUT_S = 120


def build_request_geometry(bbox: tuple[float, float, float, float], crs: str):
    """Build an ``ee.Geometry.Rectangle`` directly in ``crs`` (never lon/lat).

    Same fix as ``scenes/download.py::build_search_geometry`` and
    ``dem/sources.py::load_copernicus_dem_gee``: passing ``proj`` explicitly
    means Earth Engine treats ``bbox`` as already being in that CRS.
    """
    import ee

    return ee.Geometry.Rectangle(list(bbox), proj=crs, evenOdd=True, geodesic=False)


def _download_geotiff_bytes(image, geometry, scale: float, crs: str, bands: list[str]) -> bytes:
    """Fetch one image's pixels as GeoTIFF bytes via ``getDownloadURL``.

    ``region``/``scale``/``crs`` (not ``crsTransform``/``dimensions``) so
    Earth Engine performs the reprojection server-side using its own
    resampling, on a request geometry that is already in the target CRS --
    the combination documented as safe in ``tiles/registry.py``.
    """
    import requests

    url = image.select(bands).getDownloadURL(
        {
            "region": geometry,
            "scale": scale,
            "crs": crs,
            "format": "GEO_TIFF",
        }
    )
    response = requests.get(url, timeout=DOWNLOAD_TIMEOUT_S)
    response.raise_for_status()
    return response.content


def fetch_band_group(
    collection: str,
    scene_id: str,
    bands: list[str],
    window_bbox: tuple[float, float, float, float],
    crs: str,
    scale: float,
) -> tuple[np.ndarray, "object"]:
    """Fetch one group of bands (sharing one native resolution) for one scene.

    Returns ``(array, transform)`` where ``array`` has shape
    ``(len(bands), H, W)``, dtype float32, and ``transform`` is the affine
    transform of the returned grid (``rasterio``'s ``src.transform``).

    A "group" is bands that share a native resolution (e.g. Landsat's 30 m
    reflective bands vs. its 15 m pan band, or Sentinel-2's 10 m vs. 20 m
    bands) -- fetching them together in one request is both simpler and
    avoids issuing one HTTP round-trip per band.
    """
    import ee
    import rasterio
    from rasterio.io import MemoryFile

    image = ee.Image(f"{collection}/{scene_id}")
    geometry = build_request_geometry(window_bbox, crs)
    tiff_bytes = _download_geotiff_bytes(image, geometry, scale, crs, bands)

    with MemoryFile(tiff_bytes) as memfile:
        with memfile.open() as src:
            data = src.read(masked=True).astype(np.float32)
            arr = np.asarray(data.filled(np.nan), dtype=np.float32)
            transform = src.transform
    return arr, transform


def resample_band_to_grid(
    array: np.ndarray,
    src_transform,
    dst_transform,
    dst_shape: tuple[int, int],
    crs: str,
    method: str = "bilinear",
) -> np.ndarray:
    """Resample a ``(H, W)`` array onto an exact destination grid.

    Used for alignment-only upsampling: Landsat 7/8/9's SWIR1/SWIR2 (30 m ->
    15 m) and Sentinel-2's 20 m-native bands (20 m -> 10 m), the latter a plain
    resample rather than DSen2 super-resolution. Never used for the
    pansharpened bands (see :mod:`.pansharpen`).
    """
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    resampling = Resampling.bilinear if method == "bilinear" else Resampling.cubic

    dst = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=array,
        destination=dst,
        src_transform=src_transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs=crs,
        resampling=resampling,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst


def window_transform(
    bounds: tuple[float, float, float, float], resolution_m: float
):
    """North-up affine transform + (width, height) covering ``bounds`` exactly.

    Mirrors ``dem/sources.py::_window_transform``: pins the destination grid
    to the tile's own window bounds rather than letting a reprojection size
    itself to a rotated footprint.
    """
    from affine import Affine

    minx, miny, maxx, maxy = bounds
    width = max(1, round((maxx - minx) / resolution_m))
    height = max(1, round((maxy - miny) / resolution_m))
    transform = Affine(resolution_m, 0.0, minx, 0.0, -resolution_m, maxy)
    return transform, width, height


def reflectance_to_uint16(array: np.ndarray, scale: float = 10000.0, nodata: int = 65535) -> np.ndarray:
    """Quantize a float32 TOA reflectance array to uint16 (this pipeline's storage dtype).

    ``NaN`` (and any negative, which TOA reflectance should never be) maps to
    ``nodata``; finite values are scaled by ``scale`` and clipped to the
    representable range before the cast, so a rare over-bright pixel (e.g.
    cloud edge/specular glint) cannot wrap around into a valid-looking code.
    """
    out = np.full(array.shape, nodata, dtype=np.uint16)
    finite = np.isfinite(array) & (array >= 0)
    scaled = np.clip(array[finite] * scale, 0, nodata - 1)
    out[finite] = scaled.astype(np.uint16)
    return out


def uint16_to_reflectance(array: np.ndarray, scale: float = 10000.0, nodata: int = 65535) -> np.ndarray:
    """Inverse of :func:`reflectance_to_uint16`, for read-back/QA."""
    out = np.full(array.shape, np.nan, dtype=np.float32)
    valid = array != nodata
    out[valid] = array[valid].astype(np.float32) / scale
    return out
