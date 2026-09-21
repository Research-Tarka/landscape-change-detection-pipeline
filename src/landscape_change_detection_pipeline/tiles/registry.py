"""AOI tiling and the tile registry.

Purpose
-------
Cover the study-area AOI (or an optional pilot sub-area) with a grid of
square tiles, and persist a **tile registry** (Parquet) that every later
pipeline stage reads: GEE collection search/filtering, scene download, and
inference all operate per-tile, keyed on this one table.

Inputs
------
- The AOI polygon, read from a GeoPackage layer (``config.tiles.aoi_path`` /
  ``aoi_layer``), already in a projected CRS (``config.tiles.aoi_crs``, default
  EPSG:26910 / UTM zone 10N -- appropriate for NE BC).
- Tiling parameters (``tile_size_m``, ``buffer_m``): pipeline parameters, not
  constants, so a pilot run can use smaller tiles without a code change.
- An optional ``pilot_bbox`` (``[minx, miny, maxx, maxy]`` in ``aoi_crs``) that
  restricts tiling to a sub-window of the AOI, for running the pipeline on a
  small pilot area.

Outputs
-------
- A **tile registry** written as Parquet (``tile_registry.parquet``), one row
  per tile intersecting the AOI (or pilot bbox): ``tile_id``, the unbuffered
  search bbox, the buffered square analysis window, and the tile's area (for
  the LPT splitter in :mod:`.splitter`).

Design notes
------------
* **Square tiles on a regular grid, not one square window per AOI.** The AOI
  here is large and needs to be *covered* by many tiles of a bounded
  size (GEE per-request pixel limits, manageable zarr chunk sizes). The grid
  origin is anchored at the AOI's bounding-box minimum corner so tile
  boundaries are deterministic and reproducible across runs.
* **Two bounding boxes per tile, computed directly in the AOI's projected
  CRS.** ``search_minx/miny/maxx/maxy`` is the raw, unbuffered grid cell --
  used to filter GEE collections by intersecting footprint. ``window_*`` is
  the buffered square analysis window actually used to fetch pixels. Both are
  computed in the AOI's own metric CRS and only reprojected to WGS84 for
  metadata.

  This matters because of a **real, confirmed bug**: requesting an
  ``ee.Geometry.Rectangle`` built from lon/lat bounds distorts significantly on
  reprojection at high latitude (verified live: ~1.5 km grid shift at ~60
  degN). NE BC sits well north (~55-60 degN) for the same distortion to matter.
  The fix -- build the rectangle directly in the working projected
  CRS, never let Earth Engine reproject a lon/lat rectangle server-side --
  is applied here: tile windows are computed and stored in
  ``aoi_crs`` (a projected CRS), and any GEE request must pass ``window_*``
  with ``aoi_crs`` as the geometry's CRS, not a WGS84 rectangle.
* **Buffer is a plain per-edge margin.** Tiles are already square by
  construction (from the grid), so there is no aspect ratio to square off;
  the buffer is simply added to every edge of the unbuffered cell.
* **Parquet, not JSON**: a flat,
  typed, columnar table read far more often than written.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

REGISTRY_FILENAME = "tile_registry.parquet"

#: Default tile side length, before buffering, in metres.
DEFAULT_TILE_SIZE_M = 8000.0
#: Default edge buffer added to each tile's analysis window, in metres.
DEFAULT_BUFFER_M = 250.0


class TilingError(Exception):
    """Raised when the AOI input is missing, unreadable, or malformed."""


def load_aoi(
    aoi_path: str | Path,
    layer: str,
    crs: str,
) -> gpd.GeoDataFrame:
    """Load the AOI polygon layer and reproject it to ``crs`` if needed."""
    path = Path(aoi_path)
    if not path.is_file():
        raise TilingError(f"AOI GeoPackage not found: {path}")
    gdf = gpd.read_file(path, layer=layer)
    if gdf.empty:
        raise TilingError(f"AOI layer '{layer}' in {path} is empty.")
    if gdf.crs is None:
        raise TilingError(f"AOI layer '{layer}' in {path} has no CRS defined.")
    if str(gdf.crs) != crs and gdf.crs.to_epsg() != _epsg_of(crs):
        gdf = gdf.to_crs(crs)
    return gdf


def _epsg_of(crs: str) -> Optional[int]:
    from pyproj import CRS

    try:
        return CRS.from_user_input(crs).to_epsg()
    except Exception:
        return None


def aoi_geometry(gdf: gpd.GeoDataFrame, pilot_bbox: Optional[Sequence[float]] = None):
    """Return the single geometry tiling is run against.

    Dissolves the AOI layer's rows into one geometry, then -- if
    ``pilot_bbox`` is given -- intersects it with that ``[minx, miny, maxx,
    maxy]`` window (in the same CRS as ``gdf``), so a pilot run covers only a
    small area without touching any code.
    """
    geom = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union
    if pilot_bbox is not None:
        if len(pilot_bbox) != 4:
            raise TilingError(f"pilot_bbox must have 4 values [minx, miny, maxx, maxy], got {pilot_bbox}")
        geom = geom.intersection(box(*pilot_bbox))
        if geom.is_empty:
            raise TilingError("pilot_bbox does not intersect the AOI geometry.")
    return geom


def build_tile_grid(
    geom,
    tile_size_m: float,
    buffer_m: float,
    tile_id_prefix: str = "",
) -> pd.DataFrame:
    """Build the tile registry table for a regular grid covering ``geom``.

    The grid is anchored at ``geom``'s bounding-box minimum corner: tile
    ``(i, j)`` covers
    ``[minx + i*tile_size_m, miny + j*tile_size_m, minx + (i+1)*tile_size_m, miny + (j+1)*tile_size_m]``,
    so the grid is deterministic given the AOI and tile size. Only tiles whose
    unbuffered cell intersects ``geom`` are kept -- a grid clipped to the AOI's
    bounding box would otherwise include tiles entirely outside the AOI, e.g.
    when the AOI itself is not rectangular.

    Both the unbuffered search bbox and the buffered analysis window are
    computed directly in ``geom``'s CRS units (metres); see the module
    docstring for why this must not be done via a reprojected lon/lat
    rectangle.

    ``tile_id_prefix``, when non-empty, is prepended as ``"{prefix}_tile_..."``
    to every generated ``tile_id``. ``tile_id`` is purely positional (row/col
    index into this contributor's own AOI grid, starting at 0,0), so two
    different AOIs independently produce the same ``tile_0000_0000`` --
    harmless in isolation, but a silent collision the moment two
    contributors' ``data/train_cache/`` trees are merged (one tile's
    features/mask overwriting the other's). Setting a distinct prefix per
    contributor/project up front is the only guard against that; there is no
    later stage that can detect or repair the collision after the fact.
    """
    if tile_size_m <= 0:
        raise ValueError(f"tile_size_m must be positive, got {tile_size_m}")
    if buffer_m < 0:
        raise ValueError(f"buffer_m must be non-negative, got {buffer_m}")

    minx, miny, maxx, maxy = geom.bounds
    n_cols = max(1, math.ceil((maxx - minx) / tile_size_m))
    n_rows = max(1, math.ceil((maxy - miny) / tile_size_m))

    rows = []
    for j in range(n_rows):
        cell_miny = miny + j * tile_size_m
        cell_maxy = cell_miny + tile_size_m
        for i in range(n_cols):
            cell_minx = minx + i * tile_size_m
            cell_maxx = cell_minx + tile_size_m
            cell = box(cell_minx, cell_miny, cell_maxx, cell_maxy)
            if not cell.intersects(geom):
                continue
            tile_id = f"tile_{j:04d}_{i:04d}"
            if tile_id_prefix:
                tile_id = f"{tile_id_prefix}_{tile_id}"
            rows.append(
                {
                    "tile_id": tile_id,
                    "row": j,
                    "col": i,
                    "search_minx": cell_minx,
                    "search_miny": cell_miny,
                    "search_maxx": cell_maxx,
                    "search_maxy": cell_maxy,
                    "window_minx": cell_minx - buffer_m,
                    "window_miny": cell_miny - buffer_m,
                    "window_maxx": cell_maxx + buffer_m,
                    "window_maxy": cell_maxy + buffer_m,
                    "tile_size_m": float(tile_size_m),
                    "buffer_m": float(buffer_m),
                    "area_km2": (tile_size_m * tile_size_m) / 1e6,
                }
            )

    if not rows:
        raise TilingError("No grid cell intersects the AOI geometry; check tile_size_m/pilot_bbox.")

    return pd.DataFrame(rows)


def build_registry(
    aoi_path: str | Path,
    aoi_layer: str,
    aoi_crs: str,
    tile_size_m: float = DEFAULT_TILE_SIZE_M,
    buffer_m: float = DEFAULT_BUFFER_M,
    pilot_bbox: Optional[Sequence[float]] = None,
    tile_id_prefix: str = "",
) -> pd.DataFrame:
    """Load the AOI and build its tile registry table. See module docstring."""
    gdf = load_aoi(aoi_path, aoi_layer, aoi_crs)
    geom = aoi_geometry(gdf, pilot_bbox)
    registry = build_tile_grid(geom, tile_size_m, buffer_m, tile_id_prefix)
    registry["crs"] = str(gdf.crs)
    return registry


def write_registry(registry: pd.DataFrame, path: str | Path) -> Path:
    """Write the tile registry to Parquet, creating parent directories as needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    registry.to_parquet(out, index=False)
    return out


def read_registry(path: str | Path) -> pd.DataFrame:
    """Read a tile registry written by :func:`write_registry`."""
    p = Path(path)
    if not p.is_file():
        raise TilingError(f"Tile registry not found at '{p}'. Build it first via build_registry().")
    return pd.read_parquet(p)
