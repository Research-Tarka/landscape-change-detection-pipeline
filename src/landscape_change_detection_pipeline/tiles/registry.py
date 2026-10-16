"""AOI tiling and the tile registry.

Purpose
-------
Cover one or several study-area inputs (AOI "sources") with square tiles, and
persist a **tile registry** (Parquet) that every later pipeline stage reads:
GEE collection search/filtering, scene download, and inference all operate
per-tile, keyed on this one table.

Inputs
------
- One or more AOI sources: a vector file (gpkg/shp/geojson) or a folder of
  them. Each source has its own tiling ``mode`` (see below), tile size limits
  and CRS (``"auto"`` or an explicit ``EPSG:xxxx``).
- ``buffer_m``: margin added on every edge of every tile's analysis window.
- An optional ``pilot_bbox`` that restricts tiling to a sub-window, for running
  the pipeline on a small pilot area.

Outputs
-------
- A **tile registry** written as Parquet (``tile_registry.parquet``), one row
  per tile: ``tile_id``, the unbuffered search bbox, the buffered square
  analysis window, the tile's area (for the LPT splitter in :mod:`.splitter`),
  its CRS, and the ``aoi_id`` it belongs to (a mosaic is built per ``aoi_id``,
  since each one is one grid in one CRS).

Tiling modes (per source)
-------------------------
* ``grid``: a regular grid of ``tile_size_m`` tiles over the dissolved source.
  Every cell that overlaps the AOI is kept, so the tiles cover all of it.
* ``single``: one tile per feature, sized to the feature (square, centred,
  rounded to the 10 m pixel grid, bounded by ``min_tile_size_m`` /
  ``max_tile_size_m``). A feature larger than ``max_tile_size_m`` is cut into a
  grid instead.
* ``auto``: ``single`` for features that fit under ``max_tile_size_m``, ``grid``
  for the others.

CRS ``"auto"``
--------------
The source is cut along UTM zone boundaries (6 degrees of longitude, ~330 km
wide at 60 degN) and each piece is tiled in its own zone (NAD83 269xx for zones
1-23 N, WGS84 326xx/327xx elsewhere); ``aoi_id`` carries the zone
(``<name>_utm06N``). A ``single`` tile is assigned to the zone of its feature's
centre. An explicit CRS keeps one CRS for the whole source (no cutting).

Design notes
------------
* **Global grid anchor** (``grid_anchor="global"``, default): tile edges sit at
  multiples of ``tile_size_m`` in the CRS, so ids are stable when the AOI
  changes, tiles of two sources line up, and edges fall on the 10 m pixel grid.
  ``"aoi_bbox"`` is the legacy origin at the AOI's bbox minimum corner.
* **Two bounding boxes per tile, computed directly in the tile's projected
  CRS.** ``search_minx/miny/maxx/maxy`` is the raw, unbuffered cell -- used to
  filter GEE collections by intersecting footprint. ``window_*`` is the
  buffered square analysis window actually used to fetch pixels.

  This matters because of a **real, confirmed bug**: requesting an
  ``ee.Geometry.Rectangle`` built from lon/lat bounds distorts significantly on
  reprojection at high latitude (verified live: ~1.5 km grid shift at ~60
  degN). The fix -- build the rectangle directly in the working projected
  CRS, never let Earth Engine reproject a lon/lat rectangle server-side --
  is applied here: tile windows are stored in the tile's own CRS, and any GEE
  request must pass ``window_*`` with that ``crs`` as the geometry's CRS.
* **Parquet, not JSON**: a flat, typed, columnar table read far more often than
  written.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box

REGISTRY_FILENAME = "tile_registry.parquet"

#: Default tile side length, before buffering, in metres.
DEFAULT_TILE_SIZE_M = 8000.0
#: Default edge buffer added to each tile's analysis window, in metres.
DEFAULT_BUFFER_M = 250.0
#: Pixel size the ``single`` tiles are rounded to (the pipeline's reference grid).
PIXEL_M = 10.0
#: Safety valve on the number of grid cells examined for one geometry.
MAX_GRID_CELLS = 5_000_000
#: Cells are shrunk by this much (m) for the overlap test so a cell that only
#: touches the AOI along an edge is not kept.
_TOUCH_EPS_M = 0.01

VECTOR_SUFFIXES = (".gpkg", ".shp", ".geojson", ".fgb")


class TilingError(Exception):
    """Raised when the AOI input is missing, unreadable, or malformed."""


@dataclass
class AoiSpec:
    """One resolved AOI source (config defaults already applied)."""

    path: str
    name: str
    layer: Optional[str] = None
    mode: str = "grid"
    crs: str = "auto"
    tile_size_m: float = DEFAULT_TILE_SIZE_M
    max_tile_size_m: float = DEFAULT_TILE_SIZE_M
    min_tile_size_m: Optional[float] = None
    min_aoi_cover_pct: float = 0.0
    tile_id_prefix: str = ""
    name_field: Optional[str] = None
    dem_source_priority: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# UTM helpers
# ---------------------------------------------------------------------------


def utm_zone(lon: float) -> int:
    """UTM zone number (1-60) of a longitude in degrees."""
    return int(min(60, max(1, math.floor((lon + 180.0) / 6.0) + 1)))


def utm_epsg(zone: int, north: bool = True) -> int:
    """EPSG of a UTM zone: NAD83 (269xx) for zones 1-23 N, else WGS84 (326xx/327xx)."""
    if north and zone <= 23:
        return 26900 + zone
    return (32600 if north else 32700) + zone


def _utm_label(zone: int, north: bool) -> str:
    return f"utm{zone:02d}{'N' if north else 'S'}"


def split_by_utm_zone(geom_lonlat) -> list[tuple[int, bool, object]]:
    """Cut a lon/lat geometry along UTM zone boundaries and the equator.

    Returns ``[(zone, north, piece), ...]`` for the non-empty pieces.
    """
    minlon, _, maxlon, _ = geom_lonlat.bounds
    pieces = []
    for zone in range(utm_zone(minlon), utm_zone(maxlon) + 1):
        lo = -180.0 + 6.0 * (zone - 1)
        for north in (True, False):
            band = box(lo, 0.0 if north else -90.0, lo + 6.0, 90.0 if north else 0.0)
            part = geom_lonlat.intersection(band)
            if not part.is_empty and part.area > 0:
                pieces.append((zone, north, part))
    return pieces


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def expand_source_paths(path: str | Path) -> list[Path]:
    """A vector file -> itself; a folder -> every vector file directly inside."""
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in VECTOR_SUFFIXES)
        if not files:
            raise TilingError(f"No vector file ({', '.join(VECTOR_SUFFIXES)}) found in folder: {p}")
        return files
    raise TilingError(f"AOI path not found: {p}")


def load_aoi(
    aoi_path: str | Path,
    layer: Optional[str],
    crs: str,
) -> gpd.GeoDataFrame:
    """Load the AOI polygon layer and reproject it to ``crs`` if needed."""
    path = Path(aoi_path)
    if not path.is_file():
        raise TilingError(f"AOI file not found: {path}")
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
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


def _union(geoms) -> object:
    series = gpd.GeoSeries(list(geoms))
    return series.union_all() if hasattr(series, "union_all") else series.unary_union


def aoi_geometry(
    gdf: gpd.GeoDataFrame,
    pilot_bbox: Optional[Sequence[float]] = None,
    pilot_bbox_requires_aoi_overlap: bool = True,
):
    """Return the single geometry tiling is run against.

    Dissolves the AOI layer's rows into one geometry, then -- if
    ``pilot_bbox`` is given -- intersects it with that ``[minx, miny, maxx,
    maxy]`` window (in the same CRS as ``gdf``), so a pilot run covers only a
    small area without touching any code.

    If ``pilot_bbox_requires_aoi_overlap`` is False, a ``pilot_bbox`` that
    doesn't intersect the AOI polygon is used as-is (as a standalone tiling
    window) instead of raising -- for pilot zones deliberately outside the
    tracked AOI.
    """
    geom = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union
    if pilot_bbox is not None:
        if len(pilot_bbox) != 4:
            raise TilingError(f"pilot_bbox must have 4 values [minx, miny, maxx, maxy], got {pilot_bbox}")
        bbox_geom = box(*pilot_bbox)
        intersection = geom.intersection(bbox_geom)
        if intersection.is_empty:
            if not pilot_bbox_requires_aoi_overlap:
                return bbox_geom
            raise TilingError("pilot_bbox does not intersect the AOI geometry.")
        geom = intersection
    return geom


# ---------------------------------------------------------------------------
# Tile rows
# ---------------------------------------------------------------------------


def _tile_row(
    tile_id: str,
    row: int,
    col: int,
    bounds: tuple[float, float, float, float],
    buffer_m: float,
    **extra,
) -> dict:
    minx, miny, maxx, maxy = bounds
    side = maxx - minx
    return {
        "tile_id": tile_id,
        "row": row,
        "col": col,
        "search_minx": minx,
        "search_miny": miny,
        "search_maxx": maxx,
        "search_maxy": maxy,
        "window_minx": minx - buffer_m,
        "window_miny": miny - buffer_m,
        "window_maxx": maxx + buffer_m,
        "window_maxy": maxy + buffer_m,
        "tile_size_m": float(side),
        "buffer_m": float(buffer_m),
        "area_km2": (side * (maxy - miny)) / 1e6,
        **extra,
    }


def _grid_rows(
    geom,
    tile_size_m: float,
    buffer_m: float,
    prefix: str,
    anchor: str,
    min_cover_pct: float = 0.0,
    local_index: bool = False,
    snap_origin: bool = False,
) -> list[dict]:
    """Rows of the grid cells that overlap ``geom`` (see :func:`build_tile_grid`).

    ``local_index``: row/col (and the ids) count from 0 at the geometry's own
    first cell instead of the absolute grid index. ``snap_origin``: with
    ``aoi_bbox``, round the origin down to the 10 m pixel grid.
    """
    minx, miny, maxx, maxy = geom.bounds
    if anchor == "global":
        ox = oy = 0.0
        i0, i1 = math.floor(minx / tile_size_m), math.ceil(maxx / tile_size_m)
        j0, j1 = math.floor(miny / tile_size_m), math.ceil(maxy / tile_size_m)
    elif anchor == "aoi_bbox":
        ox, oy = minx, miny
        if snap_origin:
            ox, oy = math.floor(minx / PIXEL_M) * PIXEL_M, math.floor(miny / PIXEL_M) * PIXEL_M
            maxx, maxy = max(maxx, ox + 1e-6), max(maxy, oy + 1e-6)
            minx, miny = ox, oy
        i0 = j0 = 0
        i1 = max(1, math.ceil((maxx - minx) / tile_size_m))
        j1 = max(1, math.ceil((maxy - miny) / tile_size_m))
    else:
        raise ValueError(f"grid_anchor must be 'global' or 'aoi_bbox', got {anchor!r}")
    i1, j1 = max(i1, i0 + 1), max(j1, j0 + 1)
    if (i1 - i0) * (j1 - j0) > MAX_GRID_CELLS:
        raise TilingError(
            f"Grid of {(i1 - i0) * (j1 - j0):,} cells over the geometry's bbox: "
            "tile_size_m is too small for this area."
        )

    ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1))
    ii, jj = ii.ravel(), jj.ravel()
    x0 = ox + ii * tile_size_m
    y0 = oy + jj * tile_size_m
    eps = _TOUCH_EPS_M
    cells = shapely.box(x0 + eps, y0 + eps, x0 + tile_size_m - eps, y0 + tile_size_m - eps)
    shapely.prepare(geom)
    keep = shapely.intersects(cells, geom)

    cover = np.full(len(cells), np.nan)
    if min_cover_pct > 0:
        full = shapely.box(x0, y0, x0 + tile_size_m, y0 + tile_size_m)
        idx = np.flatnonzero(keep)
        cover[idx] = shapely.area(shapely.intersection(full[idx], geom)) / (tile_size_m**2) * 100.0
        keep &= cover >= min_cover_pct

    rows = []
    for k in np.flatnonzero(keep):
        r_idx = int(jj[k]) - (j0 if local_index else 0)
        c_idx = int(ii[k]) - (i0 if local_index else 0)
        tile_id = f"tile_{r_idx:04d}_{c_idx:04d}"
        if prefix:
            tile_id = f"{prefix}_{tile_id}"
        rows.append(
            _tile_row(
                tile_id,
                r_idx,
                c_idx,
                (float(x0[k]), float(y0[k]), float(x0[k] + tile_size_m), float(y0[k] + tile_size_m)),
                buffer_m,
                aoi_cover_pct=float(cover[k]),
            )
        )
    return rows


def build_tile_grid(
    geom,
    tile_size_m: float,
    buffer_m: float,
    tile_id_prefix: str = "",
    grid_anchor: str = "global",
    min_aoi_cover_pct: float = 0.0,
) -> pd.DataFrame:
    """Build the tile registry table for a regular grid covering ``geom``.

    With ``grid_anchor="global"`` tile ``(i, j)`` covers
    ``[i*size, j*size, (i+1)*size, (j+1)*size]`` in ``geom``'s CRS; with
    ``"aoi_bbox"`` the origin is ``geom``'s bounding-box minimum corner. Only
    cells that overlap ``geom`` (not merely touch it) are kept, so the tiles
    cover the whole geometry without grid cells entirely outside it.

    Both the unbuffered search bbox and the buffered analysis window are
    computed directly in ``geom``'s CRS units (metres); see the module
    docstring for why this must not be done via a reprojected lon/lat
    rectangle.

    ``tile_id_prefix``, when non-empty, is prepended as ``"{prefix}_tile_..."``
    to every generated ``tile_id``. ``tile_id`` is purely positional, so two
    different AOIs independently produce the same ``tile_0000_0000`` --
    harmless in isolation, but a silent collision the moment two
    contributors' ``data/train_cache/`` trees are merged. Setting a distinct
    prefix per contributor/project up front is the only guard against that.
    """
    if tile_size_m <= 0:
        raise ValueError(f"tile_size_m must be positive, got {tile_size_m}")
    if buffer_m < 0:
        raise ValueError(f"buffer_m must be non-negative, got {buffer_m}")

    rows = _grid_rows(geom, tile_size_m, buffer_m, tile_id_prefix, grid_anchor, min_aoi_cover_pct)
    if not rows:
        raise TilingError("No grid cell intersects the AOI geometry; check tile_size_m/pilot_bbox.")
    return pd.DataFrame(rows)


def single_tile_bounds(
    geom,
    min_tile_size_m: Optional[float] = None,
    pixel_m: float = PIXEL_M,
) -> tuple[float, float, float, float]:
    """Square tile around ``geom``'s bbox, centred, on the ``pixel_m`` grid.

    Side = the bbox's longer side (at least ``min_tile_size_m``), rounded up to
    a whole number of pixels; the origin is snapped to the pixel grid and the
    side grown if needed so the tile always contains the bbox.
    """
    minx, miny, maxx, maxy = geom.bounds
    side = max(maxx - minx, maxy - miny)
    if min_tile_size_m:
        side = max(side, float(min_tile_size_m))
    side = math.ceil(side / pixel_m) * pixel_m
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    while True:
        x0 = math.floor((cx - side / 2.0) / pixel_m) * pixel_m
        y0 = math.floor((cy - side / 2.0) / pixel_m) * pixel_m
        if x0 <= minx and y0 <= miny and x0 + side >= maxx and y0 + side >= maxy:
            return x0, y0, x0 + side, y0 + side
        side += pixel_m


# ---------------------------------------------------------------------------
# One source -> rows
# ---------------------------------------------------------------------------


def _safe_label(value) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _pilot_polygon(
    pilot_bbox: Sequence[float], pilot_bbox_crs: Optional[str], source_crs: Optional[str], work_crs: str
):
    if len(pilot_bbox) != 4:
        raise TilingError(f"pilot_bbox must have 4 values [minx, miny, maxx, maxy], got {pilot_bbox}")
    bbox_crs = pilot_bbox_crs or source_crs
    if bbox_crs is None:
        raise TilingError(
            "pilot_bbox with crs 'auto' needs tiles.pilot_bbox_crs (e.g. EPSG:4326 for lon/lat, "
            "or the CRS the bbox is written in)."
        )
    poly = gpd.GeoSeries([box(*pilot_bbox)], crs=bbox_crs)
    return poly.to_crs(work_crs).iloc[0]


def _features(spec: AoiSpec, path: Path, work_crs: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, layer=spec.layer) if spec.layer else gpd.read_file(path)
    if gdf.empty:
        raise TilingError(f"AOI source {path} is empty.")
    if gdf.crs is None:
        raise TilingError(f"AOI source {path} has no CRS defined.")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["geometry"] = shapely.make_valid(gdf.geometry.values)
    return gdf.to_crs(work_crs)


def build_source_registry(
    spec: AoiSpec,
    path: Path,
    buffer_m: float = DEFAULT_BUFFER_M,
    grid_anchor: str = "global",
    pilot_bbox: Optional[Sequence[float]] = None,
    pilot_bbox_crs: Optional[str] = None,
    pilot_bbox_requires_aoi_overlap: bool = True,
) -> pd.DataFrame:
    """Tile one vector file according to ``spec``. See the module docstring."""
    if spec.tile_size_m <= 0:
        raise ValueError(f"tile_size_m must be positive, got {spec.tile_size_m}")
    if buffer_m < 0:
        raise ValueError(f"buffer_m must be non-negative, got {buffer_m}")
    if spec.tile_size_m > spec.max_tile_size_m:
        raise TilingError(
            f"[{spec.name}] tile_size_m ({spec.tile_size_m:g}) exceeds max_tile_size_m ({spec.max_tile_size_m:g})."
        )
    if spec.min_tile_size_m and spec.min_tile_size_m > spec.max_tile_size_m:
        raise TilingError(f"[{spec.name}] min_tile_size_m exceeds max_tile_size_m.")
    if spec.tile_size_m % PIXEL_M:
        print(f"[tiles] warning: [{spec.name}] tile_size_m {spec.tile_size_m:g} is not a multiple of {PIXEL_M:g} m; "
              "tile edges will not sit on the 10 m pixel grid.")

    auto_crs = spec.crs.lower() == "auto"
    work_crs = "EPSG:4326" if auto_crs else spec.crs
    gdf = _features(spec, path, work_crs)
    if not auto_crs and not gdf.crs.is_projected:
        raise TilingError(f"[{spec.name}] crs {spec.crs} is not projected (metres needed).")

    if pilot_bbox is not None:
        pilot = _pilot_polygon(pilot_bbox, pilot_bbox_crs, None if auto_crs else spec.crs, work_crs)
        clipped = gdf.geometry.intersection(pilot)
        gdf = gdf[~clipped.is_empty].copy()
        gdf["geometry"] = clipped[gdf.index]
        if gdf.empty:
            if not pilot_bbox_requires_aoi_overlap:
                gdf = gpd.GeoDataFrame({"geometry": [pilot]}, crs=work_crs)
            else:
                raise TilingError(f"[{spec.name}] pilot_bbox does not intersect the AOI geometry.")

    if spec.name_field and spec.name_field not in gdf.columns:
        raise TilingError(f"[{spec.name}] name_field '{spec.name_field}' not in {list(gdf.columns)}")
    priority = ",".join(spec.dem_source_priority)
    base_prefix = spec.tile_id_prefix or ""
    aoi_base = base_prefix or spec.name

    def meta(aoi_id: str, crs_epsg: str, mode: str) -> dict:
        return {"aoi_id": aoi_id, "aoi_name": spec.name, "mode": mode, "crs": crs_epsg,
                "dem_source_priority": priority}

    def compose(*parts: Optional[str]) -> str:
        return "_".join(p for p in parts if p)

    rows: list[dict] = []

    # -- name_field: every feature (e.g. a glacier) is its own AOI ------------
    # aoi_id and tile_id start with the feature's value of that column; the CRS
    # (auto) is the UTM zone of the feature's centre, the whole feature in it.
    # A feature that fits under max_tile_size_m gets one tile sized to it
    # (modes single/auto); a bigger one, or any feature in mode grid, is gridded
    # from its own bbox corner so it gets the fewest tiles.
    if spec.name_field:
        raw = gdf[spec.name_field]
        blank = (raw.isna() | (raw.astype(str).str.strip() == "")).to_numpy()
        label = [
            f"{spec.name}_{k:05d}" if b else _safe_label(v) for k, (v, b) in enumerate(zip(raw.tolist(), blank))
        ]
        if blank.any():
            print(f"[tiles] warning: [{spec.name}] {int(blank.sum())} feature(s) with an empty '{spec.name_field}': "
                  f"named '{spec.name}_<index>'.")
        merged = gpd.GeoDataFrame({"label": label}, geometry=gdf.geometry.values, crs=gdf.crs).dissolve(
            by="label", sort=False
        )
        if len(merged) < len(gdf):
            print(f"[tiles] [{spec.name}] {len(gdf) - len(merged)} feature(s) share a '{spec.name_field}' with another: "
                  "merged into one AOI per value.")
        names = merged.index.tolist()
        geoms = merged.geometry.values

        if auto_crs:
            b = merged.geometry.bounds
            lons = ((b["minx"] + b["maxx"]) / 2.0).to_numpy()
            lats = ((b["miny"] + b["maxy"]) / 2.0).to_numpy()
            groups: dict[tuple[int, bool], list[int]] = {}
            for k, (lon, lat) in enumerate(zip(lons, lats)):
                groups.setdefault((utm_zone(lon), lat >= 0), []).append(k)
            batches = []
            for (zone, north), members in groups.items():
                epsg_int = utm_epsg(zone, north)
                proj = gpd.GeoSeries([geoms[k] for k in members], crs="EPSG:4326").to_crs(epsg_int)
                batches.append((members, list(proj.values), f"EPSG:{epsg_int}"))
        else:
            epsg = f"EPSG:{gdf.crs.to_epsg()}" if gdf.crs.to_epsg() else str(gdf.crs)
            batches = [(list(range(len(geoms))), list(geoms), epsg)]

        for members, projected, crs_epsg in batches:
            for k, geom in zip(members, projected):
                name = names[k]
                minx, miny, maxx, maxy = geom.bounds
                if spec.mode != "grid" and max(maxx - minx, maxy - miny) <= spec.max_tile_size_m:
                    r = _tile_row(f"{name}_tile_0000_0000", 0, 0, single_tile_bounds(geom, spec.min_tile_size_m),
                                  buffer_m, aoi_cover_pct=float("nan"))
                    r.update(meta(name, crs_epsg, "single"))
                    rows.append(r)
                else:
                    for r in _grid_rows(geom, spec.tile_size_m, buffer_m, name, "aoi_bbox",
                                        spec.min_aoi_cover_pct, local_index=True, snap_origin=True):
                        r.update(meta(name, crs_epsg, "grid"))
                        rows.append(r)
        if not rows:
            raise TilingError(f"[{spec.name}] no tile intersects the AOI; check tile sizes / pilot_bbox.")
        return pd.DataFrame(rows)

    # -- no name_field: the source is one AOI ----------------------------------
    labels = [f"{i:05d}" for i in range(len(gdf))]
    geoms = list(gdf.geometry.values)

    def add_grid(geom, zone: Optional[str], crs_epsg: str) -> None:
        for r in _grid_rows(geom, spec.tile_size_m, buffer_m, "", grid_anchor, spec.min_aoi_cover_pct):
            r.update(meta("", crs_epsg, "grid"), _zone=zone)
            rows.append(r)

    def add_single(geom, name: str, zone: Optional[str], crs_epsg: str) -> None:
        r = _tile_row(f"obj_{name}", -1, -1, single_tile_bounds(geom, spec.min_tile_size_m), buffer_m,
                      aoi_cover_pct=float("nan"))
        r.update(meta("", crs_epsg, "single"), _zone=zone)
        rows.append(r)

    # explicit CRS: a single working CRS for the whole source
    if not auto_crs:
        epsg = f"EPSG:{gdf.crs.to_epsg()}" if gdf.crs.to_epsg() else str(gdf.crs)
        grid_geoms = []
        for geom, name in zip(geoms, labels):
            if spec.mode == "grid":
                grid_geoms.append(geom)
                continue
            minx, miny, maxx, maxy = geom.bounds
            if max(maxx - minx, maxy - miny) <= spec.max_tile_size_m:
                add_single(geom, name, None, epsg)
            else:
                grid_geoms.append(geom)
        if grid_geoms:
            add_grid(_union(grid_geoms), None, epsg)

    # auto CRS: single tiles take the zone of the feature's centre, grids are cut by zone
    else:
        grid_geoms_ll = []
        if spec.mode == "grid":
            grid_geoms_ll = geoms
        else:
            b = gdf.geometry.bounds
            lons = ((b["minx"] + b["maxx"]) / 2.0).to_numpy()
            lats = ((b["miny"] + b["maxy"]) / 2.0).to_numpy()
            zgroups: dict[tuple[int, bool], list[int]] = {}
            for k, (lon, lat) in enumerate(zip(lons, lats)):
                zgroups.setdefault((utm_zone(lon), lat >= 0), []).append(k)
            for (zone, north), members in zgroups.items():
                epsg_int = utm_epsg(zone, north)
                projected = gpd.GeoSeries([geoms[k] for k in members], crs="EPSG:4326").to_crs(epsg_int)
                for k, geom in zip(members, projected.values):
                    minx, miny, maxx, maxy = geom.bounds
                    if max(maxx - minx, maxy - miny) <= spec.max_tile_size_m:
                        add_single(geom, labels[k], _utm_label(zone, north), f"EPSG:{epsg_int}")
                    else:
                        grid_geoms_ll.append(geoms[k])
        if grid_geoms_ll:
            for zone, north, piece in split_by_utm_zone(_union(grid_geoms_ll)):
                epsg_int = utm_epsg(zone, north)
                projected = gpd.GeoSeries([piece], crs="EPSG:4326").to_crs(epsg_int).iloc[0]
                add_grid(projected, _utm_label(zone, north), f"EPSG:{epsg_int}")

    if not rows:
        raise TilingError(f"[{spec.name}] no tile intersects the AOI; check tile sizes / pilot_bbox.")

    # The id is the name given; the UTM zone is appended only when the source had to be
    # cut into several zones (one aoi_id = one grid in one CRS).
    zones = {r["_zone"] for r in rows} - {None}
    for r in rows:
        zone = r.pop("_zone") if len(zones) > 1 else (r.pop("_zone"), None)[1]
        r["tile_id"] = compose(base_prefix, zone, r["tile_id"])
        r["aoi_id"] = compose(aoi_base, zone)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Registry assembly
# ---------------------------------------------------------------------------


def specs_from_config(tiles_cfg) -> list[tuple[AoiSpec, Path]]:
    """Resolve ``config.tiles`` into ``(AoiSpec, file)`` pairs (folders expanded).

    ``tiles.aoi_sources`` wins when non-empty; otherwise the legacy
    ``aoi_path``/``aoi_layer``/``aoi_crs``/``tile_id_prefix`` describe one
    implicit ``grid`` source.
    """
    sources = list(tiles_cfg.aoi_sources)
    if not sources:
        sources = [
            type("LegacySource", (), dict(
                path=tiles_cfg.aoi_path, layer=tiles_cfg.aoi_layer, name=None, mode="grid",
                crs=tiles_cfg.aoi_crs, tile_size_m=None, max_tile_size_m=None, min_tile_size_m=None,
                min_aoi_cover_pct=None, tile_id_prefix=tiles_cfg.tile_id_prefix, name_field=None,
                dem_source_priority=None,
            ))()
        ]
        legacy = True
    else:
        legacy = False

    resolved: list[tuple[AoiSpec, Path]] = []
    for src in sources:
        files = expand_source_paths(src.path)
        for f in files:
            name = f.stem if (src.name is None or len(files) > 1) else src.name
            if src.name and len(files) > 1:
                name = f"{src.name}_{f.stem}"
            if legacy:
                prefix = src.tile_id_prefix or ""
            else:
                prefix = src.tile_id_prefix if src.tile_id_prefix is not None else name
                if src.tile_id_prefix is not None and len(files) > 1:
                    prefix = f"{src.tile_id_prefix}_{f.stem}"
            spec = AoiSpec(
                path=str(f),
                name=name,
                layer=src.layer,
                mode=src.mode,
                crs=src.crs,
                tile_size_m=src.tile_size_m if src.tile_size_m is not None else tiles_cfg.tile_size_m,
                max_tile_size_m=src.max_tile_size_m if src.max_tile_size_m is not None else tiles_cfg.max_tile_size_m,
                min_tile_size_m=src.min_tile_size_m if src.min_tile_size_m is not None else tiles_cfg.min_tile_size_m,
                min_aoi_cover_pct=(src.min_aoi_cover_pct if src.min_aoi_cover_pct is not None
                                   else tiles_cfg.min_aoi_cover_pct),
                tile_id_prefix=prefix,
                name_field=src.name_field,
                dem_source_priority=tuple(src.dem_source_priority or ()),
            )
            resolved.append((spec, f))
    return resolved


def build_registry_from_specs(
    specs: Sequence[tuple[AoiSpec, Path]],
    buffer_m: float = DEFAULT_BUFFER_M,
    grid_anchor: str = "global",
    pilot_bbox: Optional[Sequence[float]] = None,
    pilot_bbox_crs: Optional[str] = None,
    pilot_bbox_requires_aoi_overlap: bool = True,
) -> pd.DataFrame:
    """Tile every source and concatenate. Tiles shared by two sources of the
    same grid (same id) are an error: give the sources distinct prefixes."""
    frames = [
        build_source_registry(
            spec, path, buffer_m, grid_anchor, pilot_bbox, pilot_bbox_crs, pilot_bbox_requires_aoi_overlap
        )
        for spec, path in specs
    ]
    if not frames:
        raise TilingError("No AOI source given.")
    registry = pd.concat(frames, ignore_index=True)
    dup = registry["tile_id"][registry["tile_id"].duplicated()]
    if len(dup):
        raise TilingError(
            f"{dup.nunique()} duplicate tile_id(s) across sources (e.g. {dup.iloc[0]}): "
            "give each source a distinct tile_id_prefix/name."
        )
    return registry


def build_registry(
    aoi_path: str | Path,
    aoi_layer: Optional[str],
    aoi_crs: str,
    tile_size_m: float = DEFAULT_TILE_SIZE_M,
    buffer_m: float = DEFAULT_BUFFER_M,
    pilot_bbox: Optional[Sequence[float]] = None,
    pilot_bbox_requires_aoi_overlap: bool = True,
    tile_id_prefix: str = "",
    grid_anchor: str = "global",
) -> pd.DataFrame:
    """Single-source ``grid`` registry (the original entry point). See module docstring."""
    path = Path(aoi_path)
    if not path.is_file():
        raise TilingError(f"AOI file not found: {path}")
    spec = AoiSpec(
        path=str(path),
        name=tile_id_prefix or path.stem,
        layer=aoi_layer,
        mode="grid",
        crs=aoi_crs,
        tile_size_m=tile_size_m,
        max_tile_size_m=max(tile_size_m, DEFAULT_TILE_SIZE_M),
        tile_id_prefix=tile_id_prefix,
    )
    return build_registry_from_specs(
        [(spec, path)],
        buffer_m=buffer_m,
        grid_anchor=grid_anchor,
        pilot_bbox=pilot_bbox,
        pilot_bbox_requires_aoi_overlap=pilot_bbox_requires_aoi_overlap,
    )


def summarize_registry(registry: pd.DataFrame, max_aoi_rows: int = 50) -> pd.DataFrame:
    """Per-``aoi_id`` tile count, total km², CRS and mode (for the build log).

    With more than ``max_aoi_rows`` AOIs (one per glacier, say) it is summarised
    per CRS instead: number of AOIs, of tiles, of AOIs with several tiles, km².
    """
    if registry["aoi_id"].nunique() <= max_aoi_rows:
        return (
            registry.groupby("aoi_id", sort=True)
            .agg(n_tiles=("tile_id", "size"), area_km2=("area_km2", "sum"), crs=("crs", "first"), mode=("mode", "first"))
            .reset_index()
        )
    per_aoi = registry.groupby("aoi_id").agg(n=("tile_id", "size"), crs=("crs", "first"), km2=("area_km2", "sum"))
    return (
        per_aoi.groupby("crs")
        .agg(n_aoi=("n", "size"), n_tiles=("n", "sum"), n_aoi_multi_tile=("n", lambda x: int((x > 1).sum())), area_km2=("km2", "sum"))
        .reset_index()
    )


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
