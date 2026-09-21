"""AOI-wide mosaicking of per-tile monthly spectral-index composites.

Purpose
-------
The continuous-index counterpart of :mod:`landscape_change_detection_pipeline.mosaic.mosaic`
(Stage 8): crops every tile's monthly index composite
(:mod:`change.spectral_composites`) down to that tile's own unbuffered
``search_bbox``, then places the crops side by side onto one AOI-wide raster
per period. The crop-and-place geometry -- one AOI-anchored shared grid, so
every tile's crop lands on identical pixel boundaries regardless of that
tile's own composite's pixel-grid phase -- is exactly Stage 8's, reused
directly from :mod:`mosaic.mosaic` (``aoi_grid_origin``, ``build_mosaic_grid``,
``bbox_window``) rather than re-derived.

Why a separate module from ``mosaic.mosaic``, not a generalisation of it
--------------------------------------------------------------------------
Stage 8's mosaic is a single-band ``uint8`` categorical raster,
nearest-neighbour resampled (never interpolate between class ids). An index
mosaic is the opposite shape: up to 15 indices x 4 statistics (median/min/
max/n_obs -- see ``change.spectral_composites``) as float32 bands, bilinear
resampled where continuous (never for ``n_obs``, an integer count -- see
below). Threading both shapes through one function would mean a dtype/
resampling/band-count switch on every call, for two call sites that are
already fully exercised and tested independently; a second small module
sharing only the geometry helpers keeps each mosaic's own resampling
discipline explicit at its own call site instead.

Per-band resampling
----------------------
``median``/``min``/``max`` are continuous reflectance-derived values --
bilinear, matching ``change.spectral_composites``'s own per-scene
reprojection discipline. ``n_obs`` is an integer *count* of contributing
scenes -- nearest-neighbour, since interpolating an observation count
between neighbouring pixels would invent fractional scene counts that were
never actually observed.

Output
------
One AOI-wide ``.npz`` per period, matching every other intermediate array in
this pipeline -- not a GeoTIFF. Nothing downstream reads
this file directly in a GIS tool; the change-detection layers built on top
of it (NDVI regrowth, dNBR) read it as plain arrays. Use
``scripts/export_gui.py`` on demand for a GIS-readable file at a
specific period.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from landscape_change_detection_pipeline.change.spectral_composites import (
    STATS,
    index_composite_output_path,
    read_index_composite,
)
from landscape_change_detection_pipeline.mosaic.mosaic import aoi_grid_origin, bbox_window, build_mosaic_grid

#: n_obs is an integer count of contributing scenes -- never interpolated.
_NEAREST_NEIGHBOUR_STATS = frozenset({"n_obs"})


def discover_index_periods(index_composites_root: str | Path, tile_ids: list[str]) -> list[str]:
    """Every ``"{year:04d}-{month:02d}"`` period with at least one tile's
    index composite present, sorted."""
    periods: set[str] = set()
    for tile_id in tile_ids:
        tile_dir = Path(index_composites_root) / tile_id
        if not tile_dir.is_dir():
            continue
        for period_dir in tile_dir.iterdir():
            if period_dir.is_dir() and (period_dir / "indices.npz").is_file():
                periods.add(period_dir.name)
    return sorted(periods)


def load_period_tile_index_composites(
    index_composites_root: str | Path,
    registry: pd.DataFrame,
    period: str,
) -> list[dict]:
    """Every tile's index composite for ``period``, paired with that tile's
    ``search_bbox`` from the registry. Tiles with no composite for this
    period are skipped."""
    results: list[dict] = []
    for row in registry.itertuples(index=False):
        npz_path = index_composite_output_path(index_composites_root, row.tile_id, period)
        if not npz_path.is_file():
            continue
        composite = read_index_composite(npz_path)
        composite["tile_id"] = row.tile_id
        composite["search_bbox"] = (
            float(row.search_minx),
            float(row.search_miny),
            float(row.search_maxx),
            float(row.search_maxy),
        )
        results.append(composite)
    return results


def _reproject_bands_batch(
    stack: np.ndarray,
    src_transform,
    src_crs_wkt: str,
    dst_transform,
    dst_shape: tuple[int, int],
    dst_crs_wkt: str,
    nearest: bool,
    fill_value: float,
) -> np.ndarray:
    """Reproject a ``(n_bands, H, W)`` stack in one call, not one call per
    band -- profiling a real 4-tile-AOI period found GDAL's own per-call
    overhead, not actual pixel I/O, dominated a per-band reprojection loop
    (~1.8x measured slower than one batched call over the same 60 bands)."""
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    destination = np.full((stack.shape[0], *dst_shape), fill_value, dtype=np.float32)
    reproject(
        source=stack.astype(np.float32),
        destination=destination,
        src_transform=Affine(*src_transform[:6]),
        src_crs=CRS.from_user_input(src_crs_wkt),
        dst_transform=dst_transform,
        dst_crs=CRS.from_user_input(dst_crs_wkt),
        resampling=Resampling.nearest if nearest else Resampling.bilinear,
        src_nodata=np.nan if not nearest else None,
        dst_nodata=fill_value if not nearest else None,
    )
    return destination


def pick_index_mosaic_resolution(tiles: list[dict]) -> float:
    """The finest (smallest) ``resolution_m`` achieved by any tile this
    period -- same discipline as Stage 8's ``mosaic.pick_mosaic_resolution``."""
    if not tiles:
        raise ValueError("pick_index_mosaic_resolution requires at least one tile composite")
    return min(t["resolution_m"] for t in tiles)


def build_period_index_mosaic(tiles: list[dict], registry: pd.DataFrame) -> dict:
    """Crop-and-place one period's tile index composites onto one AOI-wide,
    multi-band raster. Returns a dict with ``bands`` (an ordered
    ``{"{index}__{stat}": (H, W) float32 array}``), the mosaic's
    transform/CRS, achieved resolution, and which tiles contributed.
    """
    if not tiles:
        raise ValueError("build_period_index_mosaic requires at least one tile composite")

    dst_crs_wkt = tiles[0]["crs_wkt"]
    resolution_m = pick_index_mosaic_resolution(tiles)
    aoi_bounds = aoi_grid_origin(registry)
    mosaic_transform, mosaic_shape = build_mosaic_grid(aoi_bounds, resolution_m)

    index_names = sorted({name for t in tiles for name in t["stats"]})
    band_keys = [f"{name}__{stat}" for name in index_names for stat in STATS]
    bands = {key: np.full(mosaic_shape, np.nan, dtype=np.float32) for key in band_keys}
    tiles_present: list[str] = []

    for tile in tiles:
        window = bbox_window(mosaic_transform, tile["search_bbox"])
        row_off, col_off, height, width = window
        row_off = max(row_off, 0)
        col_off = max(col_off, 0)
        height = min(row_off + height, mosaic_shape[0]) - row_off
        width = min(col_off + width, mosaic_shape[1]) - col_off
        if height <= 0 or width <= 0:
            continue

        from affine import Affine

        dst_transform = mosaic_transform @ Affine.translation(col_off, row_off)

        # Split into a bilinear batch (median/min/max -- continuous values)
        # and a nearest-neighbour batch (n_obs -- an integer count), each
        # reprojected in one call rather than one call per index x stat.
        bilinear_items = [
            (f"{name}__{stat}", array)
            for name, stats in tile["stats"].items()
            for stat, array in stats.items()
            if stat not in _NEAREST_NEIGHBOUR_STATS
        ]
        nearest_items = [
            (f"{name}__{stat}", array)
            for name, stats in tile["stats"].items()
            for stat, array in stats.items()
            if stat in _NEAREST_NEIGHBOUR_STATS
        ]

        for items, nearest, fill in ((bilinear_items, False, np.nan), (nearest_items, True, 0.0)):
            if not items:
                continue
            keys, arrays = zip(*items)
            crops = _reproject_bands_batch(
                np.stack(arrays, axis=0), tile["transform"], tile["crs_wkt"], dst_transform, (height, width),
                dst_crs_wkt, nearest=nearest, fill_value=fill,
            )
            for key, crop in zip(keys, crops):
                bands[key][row_off : row_off + height, col_off : col_off + width] = crop

        tiles_present.append(tile["tile_id"])

    return {
        "bands": bands,
        "transform": tuple(mosaic_transform)[:6],
        "crs_wkt": dst_crs_wkt,
        "resolution_m": resolution_m,
        "tiles_present": sorted(tiles_present),
    }


def index_mosaic_output_path(output_root: str | Path, period: str) -> Path:
    """``<output_root>/<period>/indices.npz``."""
    return Path(output_root) / period / "indices.npz"


def write_index_mosaic(path: str | Path, result: dict) -> Path:
    """Write one period's multi-band index mosaic as an uncompressed
    ``.npz`` -- matching every other intermediate array in this pipeline;
    nothing reads this file directly in a GIS tool, so it is never written as a GeoTIFF.
    Use ``scripts/export_gui.py`` on demand to get a GIS-readable
    file for a specific period."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    band_keys = list(result["bands"].keys())
    payload = {
        "transform": np.array(list(result["transform"])[:6], dtype=np.float64),
        "crs_wkt": np.array(result["crs_wkt"]),
        "resolution_m": np.array(result["resolution_m"], dtype=np.float64),
        "tiles_present": np.array(result["tiles_present"]),
        "band_keys": np.array(band_keys),
    }
    for key in band_keys:
        payload[f"band__{key}"] = result["bands"][key]
    np.savez(out, **payload)
    return out


def read_index_mosaic(path: str | Path) -> dict:
    """Read an index mosaic written by :func:`write_index_mosaic`."""
    with np.load(Path(path), allow_pickle=False) as data:
        band_keys = [str(k) for k in data["band_keys"]]
        return {
            "bands": {key: np.array(data[f"band__{key}"]) for key in band_keys},
            "transform": tuple(float(v) for v in data["transform"]),
            "crs_wkt": str(data["crs_wkt"]),
            "resolution_m": float(data["resolution_m"]),
            "tiles_present": [str(t) for t in data["tiles_present"]],
        }


def build_all_period_index_mosaics(
    index_composites_root: str | Path,
    output_root: str | Path,
    registry: pd.DataFrame,
    overwrite: bool = False,
) -> list[Path]:
    """Build every period's AOI-wide index mosaic available across
    ``registry``'s tiles. Returns the paths actually (re)written."""
    tile_ids = registry["tile_id"].tolist()
    periods = discover_index_periods(index_composites_root, tile_ids)

    written: list[Path] = []
    for period in periods:
        out_path = index_mosaic_output_path(output_root, period)
        if out_path.is_file() and not overwrite:
            continue
        tiles = load_period_tile_index_composites(index_composites_root, registry, period)
        if not tiles:
            continue
        result = build_period_index_mosaic(tiles, registry)
        write_index_mosaic(out_path, result)
        written.append(out_path)
    return written
