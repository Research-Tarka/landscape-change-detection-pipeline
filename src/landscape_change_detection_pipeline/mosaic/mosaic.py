"""AOI-wide mosaicking of per-tile monthly composites (the mosaicking stage).

Purpose
-------
Every stage through Stage 7 (:mod:`landscape_change_detection_pipeline.inference.composites`)
stays tile-scoped: each tile's stored composite covers its buffered
``window_bbox``, and neighbouring tiles' ``window_bbox``\\es overlap by
``2 * tiles.buffer_m`` by construction (the buffer is added to every tile's
edge independently -- see :mod:`landscape_change_detection_pipeline.tiles.registry`).
So tile composites cannot be stitched by placing them edge to edge as-is.
This module builds one continuous, georeferenced raster per period across the
whole AOI, which every stage after this one (change detection, a downstream
pipeline, or a plain GIS viewer) can read directly.

Crop-and-place, not blending
------------------------------
Each registry row also carries the tile's unbuffered ``search_bbox`` -- its
true, non-overlapping share of the AOI grid. Adjacent tiles' ``search_bbox``
boxes tile the AOI exactly, with zero gap and zero overlap, by construction
(``tiles/registry.py::build_tile_grid`` derives every tile's ``search_*`` from
one shared grid anchored at the AOI's own bounding-box minimum corner).
Mosaicking is therefore a plain crop-and-place: for each period, crop every
tile's composite down to its own ``search_bbox`` (discarding the buffer), then
place the crops side by side onto one AOI-wide raster. No feathering, no
seam-blending logic -- the crop step alone removes every source of overlap.

Why "crop" means "reproject onto a shared grid", not "array-slice"
---------------------------------------------------------------------
A tile's stored composite transform is **not** exactly anchored at that
tile's ``window_bbox``: it inherits its pixel-grid origin from whichever
sensor scene Stage 7 picked as that tile-month's reference grid (Earth
Engine's own per-scene pixel-grid snapping), which drifts from the registry's
``window_minx/window_maxy`` by a fraction of a pixel and even differs slightly
between two neighbouring tiles at the same resolution and month (confirmed on
the real 4-tile pilot registry: at 15 m, tile_0000_0000's 1999-08 composite is
568x568 px while its immediate neighbour tile_0000_0001's is 568x567 px, and
their origins are not an exact multiple of 15 m apart). A raw array slice at
each tile's own ``search_bbox`` would therefore not land on matching pixel
boundaries between neighbours, reintroducing a seam this stage exists to
remove.

The fix: define one AOI-wide output grid, anchored at the AOI's own bounding
box minimum corner (the same anchor ``build_tile_grid`` already uses for the
tile grid itself, read directly off the registry rather than re-derived), at
the period's chosen resolution (see below). Every tile's composite is
reprojected (nearest-neighbour -- these are categorical class labels, never
continuous-value resampling, matching
``inference.composites.reproject_class_map_nearest``) directly onto that
tile's ``search_bbox`` sub-window of the shared grid. Because the destination
grid is anchored once for the whole AOI rather than once per tile, every
tile's crop lands on exactly the same pixel boundaries as its neighbours', so
placing the crops side by side leaves zero gap and zero overlap regardless of
each source composite's own grid phase.

Per-period resolution
-----------------------
Neighbouring tiles can differ in which sensor covered them in a given month
(one tile gets Sentinel-2 that month, its neighbour only Landsat), so a
period's tiles can arrive at different resolutions. The mosaic's output
resolution for that period is the finest resolution achieved by *any* tile
that period (the minimum ``resolution_m`` across that period's composites);
every coarser tile is nearest-neighbour-resampled up to it during the same
reprojection step used for cropping -- one operation does both, matching the
resampling discipline Stage 7 already uses for cross-sensor compositing
within a single tile.

Output
------
One AOI-wide ``.npz`` per period (``<output_root>/<period>/mosaic.npz``),
matching every other intermediate array this pipeline stores (per-tile class
maps, per-tile monthly composites) -- not a GeoTIFF. Nothing downstream of
this stage needs a GIS-readable file directly; a GeoTIFF is only produced
on demand, at whichever specific stage/period actually needs to be opened in
a GIS tool, via the separate :mod:`landscape_change_detection_pipeline.export.geotiff`
converter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from landscape_change_detection_pipeline.inference.composites import composite_output_path


@dataclass(frozen=True)
class TileComposite:
    tile_id: str
    composite: np.ndarray
    transform: tuple[float, ...]
    crs_wkt: str
    resolution_m: float
    nodata: int
    search_bbox: tuple[float, float, float, float]  # (minx, miny, maxx, maxy)


def discover_periods(output_root: str | Path, tile_ids: list[str]) -> list[str]:
    """Every ``"{year:04d}-{month:02d}"`` period with at least one tile's
    composite present, sorted."""
    periods: set[str] = set()
    for tile_id in tile_ids:
        tile_dir = Path(output_root) / tile_id
        if not tile_dir.is_dir():
            continue
        for period_dir in tile_dir.iterdir():
            if period_dir.is_dir() and (period_dir / "composite.npz").is_file():
                periods.add(period_dir.name)
    return sorted(periods)


def load_period_tile_composites(
    composites_root: str | Path,
    registry: pd.DataFrame,
    period: str,
) -> list[TileComposite]:
    """Every tile's composite for ``period`` (``"YYYY-MM"``), paired with that
    tile's unbuffered ``search_bbox`` from the registry. Tiles with no
    composite for this period are skipped (a gap in coverage, not an error --
    a tile can simply have no usable scene that month)."""
    results: list[TileComposite] = []
    for row in registry.itertuples(index=False):
        npz_path = composite_output_path(composites_root, row.tile_id, period)
        if not npz_path.is_file():
            continue
        with np.load(npz_path, allow_pickle=False) as data:
            results.append(
                TileComposite(
                    tile_id=row.tile_id,
                    composite=np.array(data["composite"]),
                    transform=tuple(float(v) for v in data["transform"]),
                    crs_wkt=str(data["crs_wkt"]),
                    resolution_m=float(data["resolution_m"]),
                    nodata=int(data["nodata"]),
                    search_bbox=(
                        float(row.search_minx),
                        float(row.search_miny),
                        float(row.search_maxx),
                        float(row.search_maxy),
                    ),
                )
            )
    return results


def pick_mosaic_resolution(tiles: list[TileComposite]) -> float:
    """The finest (smallest) ``resolution_m`` achieved by any tile this
    period -- every coarser tile is resampled up to it (see module
    docstring)."""
    if not tiles:
        raise ValueError("pick_mosaic_resolution requires at least one tile composite")
    return min(t.resolution_m for t in tiles)


def aoi_grid_origin(registry: pd.DataFrame) -> tuple[float, float, float, float]:
    """The AOI's own bounding-box corners, in ``search_*`` terms: ``(minx,
    miny, maxx, maxy)`` across every tile in the registry. Anchoring the
    mosaic grid here -- the same corner ``tiles/registry.py::build_tile_grid``
    already anchors the tile grid itself at -- is what makes every tile's
    crop land on shared pixel boundaries with its neighbours (see module
    docstring)."""
    return (
        float(registry["search_minx"].min()),
        float(registry["search_miny"].min()),
        float(registry["search_maxx"].max()),
        float(registry["search_maxy"].max()),
    )


def build_mosaic_grid(
    aoi_bounds: tuple[float, float, float, float],
    resolution_m: float,
):
    """The AOI-wide output grid (transform + shape) at ``resolution_m``,
    anchored at ``aoi_bounds``'s minimum corner with north-up rows."""
    from affine import Affine

    minx, miny, maxx, maxy = aoi_bounds
    width = max(1, round((maxx - minx) / resolution_m))
    height = max(1, round((maxy - miny) / resolution_m))
    transform = Affine(resolution_m, 0.0, minx, 0.0, -resolution_m, maxy)
    return transform, (height, width)


def bbox_window(transform, bbox: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    """``bbox``'s pixel window ``(row_off, col_off, height, width)`` on a
    raster with north-up ``transform``, rounded to the nearest whole pixel
    (a tile's ``search_bbox`` corners are exact multiples of the mosaic
    resolution by construction: both the tile grid and the mosaic grid share
    the same AOI-anchored origin -- see module docstring)."""
    minx, miny, maxx, maxy = bbox
    col_off = round((minx - transform.c) / transform.a)
    row_off = round((maxy - transform.f) / transform.e)
    col_end = round((maxx - transform.c) / transform.a)
    row_end = round((miny - transform.f) / transform.e)
    return row_off, col_off, row_end - row_off, col_end - col_off


def crop_tile_to_search_bbox(
    tile: TileComposite,
    dst_crs_wkt: str,
    mosaic_transform,
    mosaic_shape: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Reproject one tile's composite onto the slice of the shared AOI
    mosaic grid covering that tile's own ``search_bbox``, nearest-neighbour.

    The destination window's pixel bounds are derived from
    ``mosaic_transform``/``mosaic_shape`` -- the *same* grid
    :func:`build_period_mosaic` places crops onto -- rather than
    independently re-deriving a width/height from ``search_bbox`` and
    ``resolution_m``. A tile's edge length (e.g. an 8000 m tile at 15 m/px =
    533.33 px) is not generally an exact pixel count, so rounding it
    independently in two places can disagree by a pixel between neighbours;
    computing the window once here and reusing it for both the reprojection
    target and the placement offset guarantees the two always agree, which
    is what keeps neighbouring crops seamless (see module docstring).

    Returns the cropped array and its placement window
    ``(row_off, col_off, height, width)`` on the mosaic grid.
    """
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.warp import reproject
    from rasterio.enums import Resampling

    window = bbox_window(mosaic_transform, tile.search_bbox)
    row_off, col_off, height, width = window
    row_off = max(row_off, 0)
    col_off = max(col_off, 0)
    height = min(row_off + height, mosaic_shape[0]) - row_off
    width = min(col_off + width, mosaic_shape[1]) - col_off
    window = (row_off, col_off, height, width)

    dst_transform = mosaic_transform @ Affine.translation(col_off, row_off)
    src_affine = Affine(*tile.transform[:6])

    destination = np.full((height, width), tile.nodata, dtype=np.uint8)
    reproject(
        source=tile.composite.astype(np.uint8),
        destination=destination,
        src_transform=src_affine,
        src_crs=CRS.from_user_input(tile.crs_wkt),
        dst_transform=dst_transform,
        dst_crs=CRS.from_user_input(dst_crs_wkt),
        resampling=Resampling.nearest,
        src_nodata=tile.nodata,
        dst_nodata=tile.nodata,
    )
    return destination, window


def build_period_mosaic(
    tiles: list[TileComposite],
    registry: pd.DataFrame,
    nodata: int = 255,
) -> dict:
    """Crop-and-place one period's tile composites onto one AOI-wide raster.

    Returns a dict with the ``(H, W)`` mosaic array, its transform/CRS, the
    achieved ``resolution_m``, and which tiles contributed.
    """
    if not tiles:
        raise ValueError("build_period_mosaic requires at least one tile composite")

    dst_crs_wkt = tiles[0].crs_wkt
    resolution_m = pick_mosaic_resolution(tiles)
    aoi_bounds = aoi_grid_origin(registry)
    mosaic_transform, mosaic_shape = build_mosaic_grid(aoi_bounds, resolution_m)

    mosaic = np.full(mosaic_shape, nodata, dtype=np.uint8)
    tiles_present: list[str] = []
    for tile in tiles:
        crop, (row_off, col_off, height, width) = crop_tile_to_search_bbox(
            tile, dst_crs_wkt, mosaic_transform, mosaic_shape
        )
        if height <= 0 or width <= 0:
            continue
        mosaic[row_off : row_off + height, col_off : col_off + width] = crop
        tiles_present.append(tile.tile_id)

    return {
        "mosaic": mosaic,
        "transform": tuple(mosaic_transform)[:6],
        "crs_wkt": dst_crs_wkt,
        "resolution_m": resolution_m,
        "tiles_present": sorted(tiles_present),
        "nodata": nodata,
    }


def mosaic_output_path(output_root: str | Path, period: str) -> Path:
    """``<output_root>/<period>/mosaic.npz``."""
    return Path(output_root) / period / "mosaic.npz"


def write_mosaic(path: str | Path, result: dict) -> Path:
    """Write one period's AOI-wide mosaic as an uncompressed ``.npz``
    (matching ``inference.composites.write_composite``'s own per-tile
    format, since compression is not worth its write-time cost for this pipeline's
    intermediate arrays -- categorical ``uint8`` mosaic data compresses fast
    with zlib, but there is no downstream reader here that benefits from a
    smaller file at the cost of slower writes, so plain ``np.savez`` is used
    uniformly rather than special-casing this one call site)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        mosaic=result["mosaic"],
        transform=np.array(list(result["transform"])[:6], dtype=np.float64),
        crs_wkt=np.array(result["crs_wkt"]),
        resolution_m=np.array(result["resolution_m"], dtype=np.float64),
        tiles_present=np.array(result["tiles_present"]),
        nodata=np.array(result["nodata"], dtype=np.uint8),
    )
    return out


def read_mosaic(path: str | Path) -> dict:
    """Read a mosaic written by :func:`write_mosaic`."""
    with np.load(Path(path), allow_pickle=False) as data:
        return {
            "mosaic": np.array(data["mosaic"]),
            "transform": tuple(float(v) for v in data["transform"]),
            "crs_wkt": str(data["crs_wkt"]),
            "resolution_m": float(data["resolution_m"]),
            "tiles_present": [str(t) for t in data["tiles_present"]],
            "nodata": int(data["nodata"]),
        }


def build_all_period_mosaics(
    composites_root: str | Path,
    output_root: str | Path,
    registry: pd.DataFrame,
    nodata: int = 255,
    overwrite: bool = False,
) -> list[Path]:
    """Build every period's AOI-wide mosaic available across ``registry``'s
    tiles. Returns the paths actually (re)written."""
    tile_ids = registry["tile_id"].tolist()
    periods = discover_periods(composites_root, tile_ids)

    written: list[Path] = []
    for period in periods:
        out_path = mosaic_output_path(output_root, period)
        if out_path.is_file() and not overwrite:
            continue
        tiles = load_period_tile_composites(composites_root, registry, period)
        if not tiles:
            continue
        result = build_period_mosaic(tiles, registry, nodata=nodata)
        write_mosaic(out_path, result)
        written.append(out_path)
    return written
