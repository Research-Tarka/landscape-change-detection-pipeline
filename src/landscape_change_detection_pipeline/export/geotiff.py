"""Convert any of this pipeline's stored ``.npz`` rasters to a GeoTIFF.

Purpose
-------
Every stage in this
pipeline stores its raster output as ``.npz`` (or zarr for the per-tile
scene/DEM stores), never GeoTIFF -- GeoTIFF write cost (COG overviews, in
particular) is not worth paying for outputs nothing downstream reads back as
a GIS file. This module is the one place that cost *is* paid: on demand, for
one specific file a person actually wants to open in QGIS or hand to someone
outside this codebase, never as a side effect of a pipeline stage running.

Supported inputs
-------------------
Auto-detected from which array keys are present in the ``.npz`` (each
pipeline stage's writer uses a distinct, fixed key set, so this is
unambiguous, never a guess):

- **Single-band categorical** (``inference.engine.write_class_map``,
  ``inference.composites.write_composite``, ``mosaic.mosaic.write_mosaic``):
  keys ``class_map``/``composite``/``mosaic`` (whichever is present) +
  ``transform`` + ``crs_wkt`` (+ ``nodata`` if present). Written as a
  single-band ``uint8`` GeoTIFF, nearest-neighbour overviews (categorical
  labels).
- **Multi-band continuous index composite/mosaic**
  (``change.spectral_composites.write_index_composite``,
  ``change.index_mosaic.write_index_mosaic``): keys following the
  ``{name}__{stat}`` or ``band__{key}`` convention (both are tried) +
  ``transform`` + ``crs_wkt``. Written as a multi-band ``float32`` GeoTIFF,
  ``nan`` nodata, band descriptions set to each key so a GIS consumer does
  not have to remember band ordering positionally.
- **Named heterogeneous multi-band** (``change.change_maps.write_change_map``):
  a fixed set of named dense ``(H, W)`` arrays, each with its own natural
  dtype (categorical ``uint8``, boolean masks, continuous ``float32``) --
  every array is written as its own band, cast to whichever of ``uint8``/
  ``float32`` fits it, in :data:`CHANGE_MAP_KEYS` order, with band
  descriptions set to each key.
- **Dense regrowth-severity dNBR** (``change.regrowth_severity.write_regrowth_severity_result``):
  keys ``dnbr`` + ``has_dnbr`` + ``transform`` + ``crs_wkt`` -- the dense
  ``(H, W)`` dNBR grid and its validity mask (the file's other keys,
  ``trajectory_*``, are a sparse per-(pixel, month) table, not a grid, and
  are never part of a GeoTIFF export).
- **Generic dense grids** (``change_events``, ``snow_water_dynamics``,
  ``segments`` maps, and everything written by ``scripts/14``: vegetation
  dynamics, recovery analysis, monthly anomalies): every 2-D array on the
  file's grid is a band, every 3-D array a band per slice
  (``<key>__<year|water year|month>``), each with its own dtype.
- **Landcover persistence** (``change.landcover_persistence``): the
  interval table summarised per tracked class (count, first start, last end,
  still active).
- **Sparse per-pixel table rasterised onto its own reference grid**
  (``change.bfast.write_bfast_result``): keys ``row`` + ``col`` + ``shape``
  (the reference grid every ``(row, col)`` indexes into) + one or more
  per-pixel value arrays (``has_break``, ``break_date``, ``magnitude``).
  Scattered onto a dense ``(H, W)`` grid per value array (pixels never
  covered by a row are left at that band's nodata) before being written
  like any other multi-band GeoTIFF.

Extending this converter
---------------------------
Every ``.npz`` writer in this pipeline stores the same two georeferencing
keys (``transform``, ``crs_wkt``), so adding support for another stage's
file is a matter of adding its raster-array key name to
:data:`SINGLE_BAND_KEYS` (categorical) or teaching :func:`_is_multiband`
its key convention (continuous, multi-statistic) -- not a new code path per
format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

#: Single-band categorical raster key, per writer -- checked in this order;
#: the first one present in the ``.npz`` wins.
SINGLE_BAND_KEYS: tuple[str, ...] = ("class_map", "composite", "mosaic")

#: ``change.change_maps.write_change_map`` -- fixed key set, each its own
#: band, checked before every other format (this writer's key set is
#: unambiguous: nothing else in the pipeline stores ``from_class``).
CHANGE_MAP_KEYS: tuple[str, ...] = (
    "from_class",
    "to_class",
    "changed_raw",
    "changed_persistent",
    "changed_corroborated",
    "dnbr",
)

#: ``change.regrowth_severity.write_regrowth_severity_result`` -- the dense
#: subset of that file's keys (``trajectory_*`` is a sparse table, skipped).
REGROWTH_SEVERITY_DENSE_KEYS: tuple[str, ...] = ("dnbr", "has_dnbr")

#: ``change.bfast.write_bfast_result`` -- per-pixel value arrays to scatter
#: onto the ``(row, col)`` reference grid.
BFAST_VALUE_KEYS: tuple[str, ...] = ("has_break", "break_date", "magnitude", "n_history_obs", "n_monitoring_obs")

#: ``dem.zarr_store.write_tile_dem`` -- product arrays inside the
#: ``<tile_id>.zarr/dem`` group (not ``.npz`` -- read via zarr, not
#: :func:`_load_npz`; see :func:`build_dem_export_payload`).
DEM_PRODUCT_KEYS: tuple[str, ...] = ("elevation", "slope", "aspect")


class GeoTiffExportError(Exception):
    """Raised when an ``.npz`` file's contents cannot be recognised as one
    of this converter's supported formats."""


def _load_npz(path: str | Path) -> dict:
    """Load every array in the ``.npz`` that does not require unpickling.
    Some writers (e.g. ``regrowth_severity``) store ``object``-dtype string
    arrays (``pre_month``/``post_month``) or ragged tables
    (``trajectory_*``) alongside the dense grids this converter cares
    about; those are never rasterisable, so they are skipped here rather
    than failing the whole load (``allow_pickle`` is deliberately kept
    ``False`` -- this converter never executes untrusted pickle data)."""
    with np.load(Path(path), allow_pickle=False) as data:
        loaded = {}
        for key in data.files:
            try:
                loaded[key] = np.array(data[key])
            except ValueError:
                continue
        return loaded


def _is_multiband_index_format(data: dict) -> bool:
    return "band_keys" in data or any("__" in key for key in data if key not in ("transform", "crs_wkt"))


def export_single_band(data: dict, band_key: str) -> dict:
    """Build the GeoTIFF-write payload for a single-band categorical raster."""
    array = data[band_key]
    nodata = int(data["nodata"]) if "nodata" in data else None
    return {
        "bands": {band_key: array.astype(np.uint8)},
        "transform": tuple(float(v) for v in data["transform"]),
        "crs_wkt": str(data["crs_wkt"]),
        "dtype": "uint8",
        "nodata": nodata,
        "resampling": "nearest",
    }


def export_multiband_index(data: dict) -> dict:
    """Build the GeoTIFF-write payload for a multi-band continuous index
    composite/mosaic (``{name}__{stat}`` or ``band__{key}`` convention)."""
    if "band_keys" in data:
        # index_mosaic.write_index_mosaic: band_keys + band__{key} arrays.
        band_keys = [str(k) for k in data["band_keys"]]
        bands = {key: data[f"band__{key}"] for key in band_keys}
    else:
        # spectral_composites.write_index_composite: index_names + {name}__{stat} arrays.
        index_names = [str(n) for n in data["index_names"]]
        stat_names = ("median", "min", "max", "n_obs")
        bands = {
            f"{name}__{stat}": data[f"{name}__{stat}"]
            for name in index_names
            for stat in stat_names
            if f"{name}__{stat}" in data
        }
    return {
        "bands": {key: array.astype(np.float32) for key, array in bands.items()},
        "transform": tuple(float(v) for v in data["transform"]),
        "crs_wkt": str(data["crs_wkt"]),
        "dtype": "float32",
        "nodata": np.nan,
        "resampling": "bilinear",
    }


#: Per-band dtype/nodata for each :data:`CHANGE_MAP_KEYS` band -- categorical
#: class labels stay ``uint8`` (nodata 255, the pipeline's usual "no class"
#: sentinel), boolean change masks are stored as ``uint8`` 0/1 (GeoTIFF has
#: no native boolean type), and ``dnbr`` stays continuous ``float32``.
_CHANGE_MAP_BAND_SPEC: dict[str, tuple[str, object]] = {
    "from_class": ("uint8", 255),
    "to_class": ("uint8", 255),
    "changed_raw": ("uint8", None),
    "changed_persistent": ("uint8", None),
    "changed_corroborated": ("uint8", None),
    "dnbr": ("float32", np.nan),
}


def export_named_multiband(data: dict, band_keys: tuple[str, ...], band_spec: dict[str, tuple[str, object]]) -> dict:
    """Build the GeoTIFF-write payload for a fixed set of named dense bands
    that do not share one dtype (``change_map.npz``: categorical labels +
    boolean masks + a continuous band, all in one file). Unlike
    :func:`export_single_band`/:func:`export_multiband_index`, each band
    keeps its own dtype/nodata (see ``per_band_dtype``/``per_band_nodata``
    in the payload) -- :func:`write_geotiff` writes one dtype per band
    rather than casting everything to a shared one."""
    present = [key for key in band_keys if key in data]
    bands = {}
    per_band_dtype = {}
    per_band_nodata = {}
    for key in present:
        dtype, nodata = band_spec[key]
        array = data[key]
        if dtype == "uint8" and array.dtype == np.bool_:
            array = array.astype(np.uint8)
        bands[key] = array.astype(dtype)
        per_band_dtype[key] = dtype
        per_band_nodata[key] = nodata
    return {
        "bands": bands,
        "transform": tuple(float(v) for v in data["transform"]),
        "crs_wkt": str(data["crs_wkt"]),
        "per_band_dtype": per_band_dtype,
        "per_band_nodata": per_band_nodata,
        "resampling": "nearest",
    }


def export_regrowth_severity_dnbr(data: dict) -> dict:
    """Build the GeoTIFF-write payload for the dense subset
    (``dnbr``/``has_dnbr``) of a ``regrowth_severity_<source>.npz`` file --
    the ``trajectory_*`` keys are a sparse per-(pixel, month) table, never
    part of this export."""
    band_spec = {"dnbr": ("float32", np.nan), "has_dnbr": ("uint8", None)}
    return export_named_multiband(data, REGROWTH_SEVERITY_DENSE_KEYS, band_spec)


def export_bfast_result(data: dict) -> dict:
    """Build the GeoTIFF-write payload for ``bfast.npz`` by scattering its
    per-pixel value arrays (indexed by ``row``/``col``) onto the file's own
    reference grid (``shape``). Pixels never tested (absent from the
    ``row``/``col`` table) are left at each band's nodata."""
    height, width = (int(v) for v in data["shape"])
    rows = data["row"]
    cols = data["col"]

    band_spec = {
        "has_break": ("uint8", 255),
        "break_date": ("float32", np.nan),
        "magnitude": ("float32", np.nan),
        "n_history_obs": ("float32", np.nan),
        "n_monitoring_obs": ("float32", np.nan),
    }
    bands = {}
    per_band_dtype = {}
    per_band_nodata = {}
    for key in BFAST_VALUE_KEYS:
        if key not in data:
            continue
        dtype, nodata = band_spec[key]
        fill = 255 if dtype == "uint8" else np.nan
        grid = np.full((height, width), fill, dtype=dtype)
        values = data[key]
        if dtype == "uint8" and values.dtype == np.bool_:
            values = values.astype(np.uint8)
        grid[rows, cols] = values.astype(dtype)
        bands[key] = grid
        per_band_dtype[key] = dtype
        per_band_nodata[key] = nodata

    return {
        "bands": bands,
        "transform": tuple(float(v) for v in data["transform"]),
        "crs_wkt": str(data["crs_wkt"]),
        "per_band_dtype": per_band_dtype,
        "per_band_nodata": per_band_nodata,
        "resampling": "nearest",
    }


def export_ccdc_result(data: dict) -> dict:
    """Build the GeoTIFF-write payload for ``ccdc.npz`` by deriving a dense
    per-pixel summary grid from its ragged segment table (0..N segments per
    pixel, unlike bfast's exactly-one-row-per-pixel table -- not directly a
    grid). Per pixel, across every one of its segments:

    - ``n_segments``: how many segments were fit (0 where the pixel was
      never fit at all).
    - ``latest_t_break``: the most recent segment's break date (ordinal
      day), the single most GIS-useful summary of "when did this pixel
      last change" -- NaN where no segment ever recorded a break
      (``t_break <= 0``, this module's own "no break" sentinel).
    - ``latest_change_prob``/``latest_num_obs``: that same latest segment's
      own change probability and observation count, for a quick confidence
      read alongside the break date.

    The full segment table (every segment, every band's magnitude) is not
    rasterisable even after this derivation -- use
    :mod:`landscape_change_detection_pipeline.export.csv_export` for that."""
    height, width = (int(v) for v in data["shape"])
    rows = data["row"]
    cols = data["col"]
    t_break = data["t_break"]
    change_prob = data["change_prob"]
    num_obs = data["num_obs"]

    n_segments = np.zeros((height, width), dtype=np.float32)
    latest_t_break = np.full((height, width), np.nan, dtype=np.float32)
    latest_change_prob = np.full((height, width), np.nan, dtype=np.float32)
    latest_num_obs = np.full((height, width), np.nan, dtype=np.float32)

    for row, col, tb, cp, obs in zip(rows, cols, t_break, change_prob, num_obs):
        n_segments[row, col] += 1
        has_break = tb > 0
        if has_break and (np.isnan(latest_t_break[row, col]) or tb > latest_t_break[row, col]):
            latest_t_break[row, col] = tb
            latest_change_prob[row, col] = cp
            latest_num_obs[row, col] = obs

    bands = {
        "n_segments": n_segments,
        "latest_t_break": latest_t_break,
        "latest_change_prob": latest_change_prob,
        "latest_num_obs": latest_num_obs,
    }
    return {
        "bands": bands,
        "transform": tuple(float(v) for v in data["transform"]),
        "crs_wkt": str(data["crs_wkt"]),
        "dtype": "float32",
        "nodata": np.nan,
        "resampling": "nearest",
    }


#: Keys that are georeferencing / labelling metadata, never a band.
_META_KEYS = ("transform", "crs_wkt", "resolution_m", "shape", "nodata", "product", "year0")


def _axis_labels(data: dict, key: str, n: int) -> list[str]:
    """Band labels for the leading axis of a 3-D array: ``labels__<key>`` if
    stored, else a ``water_years`` / ``years`` array of matching length, else
    the positional index."""
    for candidate in (f"labels__{key}", "water_years", "years"):
        labels = data.get(candidate)
        if labels is not None and np.ndim(labels) == 1 and len(labels) == n:
            return [str(v) for v in labels]
    return [str(i) for i in range(n)]


def _band_spec_for(array: np.ndarray) -> tuple[np.ndarray, str, object]:
    """``(array cast for GeoTIFF, dtype name, nodata)`` for one band."""
    kind = array.dtype.kind
    if kind == "b":
        return array.astype(np.uint8), "uint8", None
    if kind == "f":
        return array.astype(np.float32), "float32", np.nan
    if kind == "u":
        return (array, "uint8", None) if array.dtype.itemsize == 1 else (array.astype(np.int32), "int32", None)
    if kind == "i":
        if array.dtype.itemsize <= 2:
            return array.astype(np.int16), "int16", None
        return array.astype(np.int32), "int32", None
    raise GeoTiffExportError(f"unsupported band dtype {array.dtype}")


def _is_band_candidate(key: str, array: np.ndarray) -> bool:
    return not (key in _META_KEYS or key.startswith(("tbl_", "labels__", "grid_meta")) or array.dtype.kind not in "biuf")


def _grid_shape_of(data: dict) -> Optional[tuple[int, int]]:
    """The raster grid a generic ``.npz`` lives on: the trailing ``(H, W)`` shared by the most
    numeric 2-D / 3-D arrays (table columns, ``tbl_*`` and ``labels__*`` never count)."""
    votes: dict[tuple[int, int], int] = {}
    for key, array in data.items():
        if not _is_band_candidate(key, array):
            continue
        if array.ndim in (2, 3) and min(array.shape[-2:]) >= 16:  # (N, n_features) tables are not grids
            shape = tuple(int(v) for v in array.shape[-2:])
            votes[shape] = votes.get(shape, 0) + (array.shape[0] if array.ndim == 3 else 1)
    if not votes:
        return None
    if "shape" in data and tuple(int(v) for v in data["shape"]) in votes:
        return tuple(int(v) for v in data["shape"])
    return max(votes, key=votes.get)


def export_generic_dense(data: dict) -> dict:
    """Payload for any ``.npz`` that carries georeferencing plus dense grids: every
    2-D array on the file's grid is a band, every 3-D array a band per slice named
    ``<key>__<label>`` (year, water year, month ...). Each band keeps its own dtype (a
    file mixing dtypes is written as one GeoTIFF per dtype). Sparse tables are not
    part of it -- they are CSV exports."""
    if "transform" not in data or "crs_wkt" not in data:
        raise GeoTiffExportError("no 'transform'/'crs_wkt' in the file: not a georeferenced raster product")
    grid = _grid_shape_of(data)
    if grid is None:
        raise GeoTiffExportError("no dense (H, W) grid found in the file (a sparse table? export it as CSV)")
    bands: dict[str, np.ndarray] = {}
    per_band_dtype: dict[str, str] = {}
    per_band_nodata: dict[str, object] = {}
    for key in sorted(data):
        array = data[key]
        if not _is_band_candidate(key, array):
            continue
        if array.ndim == 2 and tuple(array.shape) == grid:
            slices = [(key, array)]
        elif array.ndim == 3 and tuple(array.shape[1:]) == grid:
            labels = _axis_labels(data, key, array.shape[0])
            slices = [(f"{key}__{lab}", array[i]) for i, lab in enumerate(labels)]
        else:
            continue
        for name, arr in slices:
            cast, dtype, nodata = _band_spec_for(arr)
            bands[name] = cast
            per_band_dtype[name] = dtype
            per_band_nodata[name] = nodata
    if not bands:
        raise GeoTiffExportError("no dense band found on the file's grid")
    return {
        "bands": bands,
        "transform": tuple(float(v) for v in data["transform"]),
        "crs_wkt": str(data["crs_wkt"]),
        "per_band_dtype": per_band_dtype,
        "per_band_nodata": per_band_nodata,
        "resampling": "nearest",
    }


def _month_code(values: np.ndarray) -> np.ndarray:
    """``"YYYY-MM"`` strings -> ``YYYYMM`` ints (0 for empty)."""
    out = np.zeros(len(values), dtype=np.int32)
    for i, v in enumerate(values):
        v = str(v)
        if len(v) >= 7:
            out[i] = int(v[:4]) * 100 + int(v[5:7])
    return out


def export_landcover_persistence(data: dict) -> dict:
    """Dense summary of ``landcover_persistence.npz``'s interval table: per tracked class,
    the number of intervals, the first start and last end (``YYYYMM``; 0 = none / still
    active) and whether the class is still active at the end of the series."""
    height, width = (int(v) for v in data["shape"])
    rows, cols = data["row"], data["col"]
    names = np.array([str(v) for v in data["class_name"]])
    start, end = _month_code(data["start_month"]), _month_code(data["end_month"])
    bands, dtypes, nodata = {}, {}, {}

    def put(key, grid, dtype, nd):
        bands[key], dtypes[key], nodata[key] = grid, dtype, nd

    for cls in sorted(set(names.tolist())):
        sel = names == cls
        r, c, st, en = rows[sel], cols[sel], start[sel], end[sel]
        n_int = np.zeros((height, width), dtype=np.uint8)
        np.add.at(n_int, (r, c), 1)
        first = np.full((height, width), 999999, dtype=np.int32)
        np.minimum.at(first, (r, c), st)
        first[first == 999999] = 0
        last = np.zeros((height, width), dtype=np.int32)
        np.maximum.at(last, (r, c), en)
        active = np.zeros((height, width), dtype=np.uint8)
        active[r[en == 0], c[en == 0]] = 1
        put(f"{cls}__n_intervals", n_int, "uint8", None)
        put(f"{cls}__first_start_yyyymm", first, "int32", None)
        put(f"{cls}__last_end_yyyymm", last, "int32", None)
        put(f"{cls}__still_active", active, "uint8", None)
    return {
        "bands": bands, "transform": tuple(float(v) for v in data["transform"]), "crs_wkt": str(data["crs_wkt"]),
        "per_band_dtype": dtypes, "per_band_nodata": nodata, "resampling": "nearest",
    }


def build_export_payload(npz_path: str | Path) -> dict:
    """Load an ``.npz`` file and return its GeoTIFF-write payload, detecting
    which of this pipeline's stored formats it is."""
    data = _load_npz(npz_path)

    if "product" in data or "latest_event_type" in data:  # scripts/14 products, change events (row/col/break_date look like bfast)
        return export_generic_dense(data)

    if "class_name" in data and "start_month" in data and "end_month" in data and "shape" in data:
        return export_landcover_persistence(data)

    if "from_class" in data:
        return export_named_multiband(data, CHANGE_MAP_KEYS, _CHANGE_MAP_BAND_SPEC)

    if "band_order" in data:
        # change.ccdc.write_ccdc_result -- checked before the bfast/row/col
        # branch below, since ccdc.npz also has row/col/shape but a ragged
        # (multi-segment-per-pixel) table, not bfast's one-row-per-pixel one.
        return export_ccdc_result(data)

    if "row" in data and "col" in data and "shape" in data and any(k in data for k in BFAST_VALUE_KEYS):
        return export_bfast_result(data)

    if "dnbr" in data and "has_dnbr" in data:
        return export_regrowth_severity_dnbr(data)

    if _is_multiband_index_format(data):
        return export_multiband_index(data)

    for key in SINGLE_BAND_KEYS:
        if key in data:
            return export_single_band(data, key)

    try:  # change events, snow/water dynamics, segments ...: any georeferenced file with dense grids
        return export_generic_dense(data)
    except GeoTiffExportError:
        pass

    raise GeoTiffExportError(
        f"'{npz_path}' does not match any recognised format (expected one of "
        f"{SINGLE_BAND_KEYS} for a single-band raster, a change-map, "
        f"regrowth-severity, ccdc, or bfast key set, or an index-composite/"
        f"index-mosaic key set for a multi-band raster). Found keys: {sorted(data)}"
    )


def build_dem_export_payload(tile_dir: str | Path, tile_id: str) -> dict:
    """Build the GeoTIFF-write payload for a tile's DEM products
    (``<tile_id>.zarr/dem``) -- a zarr group, not an ``.npz``, so this is a
    separate entry point from :func:`build_export_payload` rather than a
    branch inside it. Every product present (``elevation``/``slope``/
    ``aspect``, any subset) becomes its own band."""
    from landscape_change_detection_pipeline.dem.zarr_store import read_tile_dem

    bands = {}
    transform = None
    crs_wkt = None
    for product in DEM_PRODUCT_KEYS:
        try:
            array, product_transform, product_crs_wkt = read_tile_dem(tile_dir, tile_id, product)
        except KeyError:
            continue
        bands[product] = array.astype(np.float32)
        transform = transform or product_transform
        crs_wkt = crs_wkt or product_crs_wkt

    if not bands:
        raise GeoTiffExportError(f"No DEM products found for tile '{tile_id}' under '{tile_dir}'.")

    return {
        "bands": bands,
        "transform": (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f),
        "crs_wkt": str(crs_wkt),
        "dtype": "float32",
        "nodata": np.nan,
        "resampling": "bilinear",
    }


def _write_single_geotiff(
    bands: dict[str, np.ndarray],
    band_keys: list[str],
    dtype: str,
    nodata,
    transform: tuple[float, ...],
    crs_wkt: str,
    resampling: str,
    out_path: Path,
) -> Path:
    from affine import Affine
    import rasterio
    from rasterio.crs import CRS

    out_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = bands[band_keys[0]].shape
    stack = np.stack([bands[key] for key in band_keys], axis=0)

    profile = dict(
        driver="COG",
        height=height,
        width=width,
        count=len(band_keys),
        dtype=dtype,
        crs=CRS.from_user_input(crs_wkt),
        transform=Affine(*transform),
        compress="zstd",
        blocksize=512,
        overview_resampling=resampling,
    )
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(stack)
        if len(band_keys) > 1:
            dst.descriptions = band_keys
    return out_path


def write_geotiff(payload: dict, out_path: str | Path) -> Path:
    """Write a GeoTIFF (COG) from a payload built by
    :func:`build_export_payload`. This is the only place in this pipeline
    that pays GeoTIFF/COG write cost -- see module docstring.

    A payload with ``per_band_dtype`` (:func:`export_named_multiband` and
    callers) mixes dtypes across bands (categorical + boolean + continuous
    in one source file); since a single GeoTIFF cannot mix dtypes across
    bands, that case is written as one GeoTIFF per distinct dtype, each
    named ``<out_path stem>__<dtype><out_path suffix>``, and this function
    then returns the list of paths actually written instead of one path."""
    out = Path(out_path)

    if "per_band_dtype" in payload:
        per_band_dtype = payload["per_band_dtype"]
        per_band_nodata = payload["per_band_nodata"]
        dtypes_present = list(dict.fromkeys(per_band_dtype.values()))
        written = []
        for dtype in dtypes_present:
            keys_for_dtype = [key for key in payload["bands"] if per_band_dtype[key] == dtype]
            nodata_values = {per_band_nodata[key] for key in keys_for_dtype}
            nodata = nodata_values.pop() if len(nodata_values) == 1 else None
            dtype_out = out if len(dtypes_present) == 1 else out.with_stem(f"{out.stem}__{dtype}")
            written.append(
                _write_single_geotiff(
                    payload["bands"], keys_for_dtype, dtype, nodata,
                    payload["transform"], payload["crs_wkt"], payload["resampling"], dtype_out,
                )
            )
        return written if len(written) > 1 else written[0]

    band_keys = list(payload["bands"].keys())
    return _write_single_geotiff(
        payload["bands"], band_keys, payload["dtype"], payload["nodata"],
        payload["transform"], payload["crs_wkt"], payload["resampling"], out,
    )


def export_to_geotiff(npz_path: str | Path, out_path: str | Path) -> Path:
    """Convert one stored ``.npz`` raster to a GeoTIFF. The single entry
    point ``scripts/export_gui.py`` calls. Returns the written path, or
    a list of paths when the source mixes dtypes across bands (see
    :func:`write_geotiff`)."""
    payload = build_export_payload(npz_path)
    return write_geotiff(payload, out_path)
