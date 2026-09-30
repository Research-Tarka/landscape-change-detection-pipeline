"""NDVI regrowth trajectory and dNBR burn severity.

Purpose
-------
Two index-based, per-pixel change layers built on top of the per-pixel
break dates :mod:`change.change_detection` detects, and NBR/NDVI monthly
composites computed on demand (:func:`build_month_composite_cache`, reusing
:mod:`change.spectral_composites`'s per-scene index computation directly --
never a separate persisted index-composite store, see that function's own
docstring for why):

- **dNBR burn severity**: the NBR difference between the monthly composite
  immediately before a detected break and the one immediately after it --
  the standard remote-sensing burn-severity index, computed at the specific
  dates a break was actually detected rather than a fixed
  before/after-year pair. This is deliberately kept alongside the trained
  classifier's own ``burned_disturbed`` class rather than replaced by
  it: the class is a categorical "burned or not," while dNBR is a
  continuous severity measure (a light surface fire and a stand-replacing
  crown fire both land in the same class, but at very different dNBR
  values) -- literature severity classes (unburned/low/moderate/high) are
  themselves dNBR thresholds, not something a single categorical class
  label can express.
- **NDVI regrowth trajectory**: for every pixel with a detected break, its
  NDVI (median statistic) trajectory across every monthly composite after
  that break date -- a direct satellite-observed disturbance-age signal, in
  place of the static inventory-age fields this kind of analysis usually
  depends on. No class from the trained classifier can substitute
  for this: "regrowth" is a trajectory over time, not a state at one
  instant, so it is not something any single-scene classification (however
  accurate) could express on its own.

An NDSI-based snow filter was deliberately **not** added here: snow/ice/
water dynamics are tracked separately by
:mod:`change.snow_water_dynamics`, directly from the trained classifier's
own monthly class -- a second, index-threshold reading of "is this pixel
snow" here was judged not to add enough value to justify a second
representation of the same thing (project decision -- unlike dNBR, where
the classifier's categorical answer and the continuous severity question
are genuinely different things).

One shared grid per tile: the segmentation grid, not the composite grid
------------------------------------------------------------------------
:mod:`change.change_detection` operates on one fixed grid per tile -- the
tile's own unified 10 m grid -- while each monthly index composite
(:mod:`change.spectral_composites`) sits on *that month's* own
finest-available-sensor grid, which varies month to month (usually also
10 m, but not guaranteed for a Landsat-only month before the unified-grid
revision was applied). Combining a break date with a monthly index value
therefore requires reprojecting one onto the other; this module always
reprojects the monthly composite onto the segmentation grid (bilinear --
continuous index values), never the reverse, so that comparing "the pixel at
row r, col c" across different months always means the same real-world
location. This was an explicit project decision, favouring one fixed grid
per tile for straightforward before/after comparison over preserving each
month's own native detail.

Which break counts as "the" break for a pixel
------------------------------------------------
A pixel can carry several segments (several breaks) across the 1984-present
series. This module reads each pixel's *most recent* break
(``latest_break_dates``) -- the disturbance regrowth/severity layers care
about the latest known change, not every historical one the segment table
records.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from landscape_change_detection_pipeline.classes.class_config import ClassConfig


def build_month_composite_cache(
    tile_dir: str | Path,
    inference_root: str | Path,
    tile_id: str,
    class_config: ClassConfig,
    index_names: tuple[str, ...] = ("NBR", "NDVI"),
):
    """A ``month -> Optional[composite dict]`` accessor, computing each
    requested month's index composite directly from this tile's stored
    scenes (:mod:`change.spectral_composites`) on every call, never from a
    persisted per-month store on disk.

    A prior revision of this module read from a separate
    ``index_composites`` stage that pre-computed and stored *every* index
    (NBR, NDVI, NDSI, Tasseled Cap, ...) x 4 statistics x every tile-month
    unconditionally -- at real-AOI scale this reached hundreds of GB on disk
    for a superset this module (the only real consumer) never used more than
    two indices of (see ``docs`` history / project decision). Computing only
    ``index_names`` on demand avoids storing anything at all: the cost is
    recomputing a month's composite (reading its scenes' reflectance,
    masking cloud/shadow, reducing to median/min/max) from scratch on every
    fresh run, instead of amortizing it once to disk.

    Deliberately not memoized across months: this module's own callers
    (:func:`compute_dnbr_for_breaks`, :func:`compute_ndvi_regrowth_trajectory`)
    each visit a given month once and cache only the one reprojected
    statistic band they need, so keeping every month's full composite (both
    indices x every statistic, full tile grid) alive here too would just
    duplicate that memory for no benefit -- and across a tile's whole
    1984-present month range, previously exhausted host RAM.
    """
    from landscape_change_detection_pipeline.change.spectral_composites import (
        build_month_index_composite,
        compute_scene_indices,
        discover_tile_scenes,
        month_key,
    )
    from landscape_change_detection_pipeline.features.training_cache import scene_year
    from landscape_change_detection_pipeline.inference.composites import scene_month

    scenes_by_month: dict[str, list[tuple[str, str]]] = {}
    for sensor, scene_id in discover_tile_scenes(tile_dir, tile_id):
        key = month_key(scene_year(sensor, scene_id), scene_month(sensor, scene_id))
        scenes_by_month.setdefault(key, []).append((sensor, scene_id))

    def get_month_composite(month: str) -> Optional[dict]:
        # Not memoized across months: callers (compute_dnbr_for_breaks,
        # compute_ndvi_regrowth_trajectory) each visit a given month once and
        # immediately reproject+cache only the one statistic band they need
        # (see their own nbr_cache/ndvi_cache). Keeping every month's full
        # composite (all indices x all stats, full tile grid) alive here too
        # was pure duplication and, across a tile's whole 1984-present month
        # range, the actual cause of this stage exhausting host RAM.
        scene_indices = []
        for sensor, scene_id in scenes_by_month.get(month, []):
            try:
                scene_indices.append(
                    compute_scene_indices(tile_dir, inference_root, tile_id, sensor, scene_id, class_config, index_names)
                )
            except FileNotFoundError:
                continue  # a scene with no Stage 6 classification yet -- a coverage gap, not an error
        return build_month_index_composite(scene_indices, index_names) if scene_indices else None

    return get_month_composite


def _reproject_index_band(
    array: np.ndarray,
    src_transform,
    src_crs_wkt: str,
    dst_transform,
    dst_crs_wkt: str,
    dst_shape: tuple[int, int],
) -> np.ndarray:
    """Bilinear-reproject one monthly composite statistic band onto the
    break-detection grid (see module docstring: always the composite onto
    the fixed break grid, never the reverse)."""
    if (
        tuple(array.shape) == tuple(dst_shape)
        and tuple(src_transform)[:6] == tuple(dst_transform)[:6]
        and src_crs_wkt == dst_crs_wkt
    ):
        return array.astype(np.float32)  # already on the break grid (the usual 10 m case): nothing to resample

    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    destination = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=array.astype(np.float32),
        destination=destination,
        src_transform=Affine(*src_transform[:6]),
        src_crs=CRS.from_user_input(src_crs_wkt),
        dst_transform=Affine(*dst_transform[:6]),
        dst_crs=CRS.from_user_input(dst_crs_wkt),
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return destination


def load_index_on_grid(
    get_month_composite,
    month: str,
    index_name: str,
    stat: str,
    dst_transform,
    dst_crs_wkt: str,
    dst_shape: tuple[int, int],
) -> Optional[np.ndarray]:
    """One index/statistic for one tile-month, reprojected onto the given
    (break-detection) grid. ``get_month_composite`` is a
    :func:`build_month_composite_cache` accessor. Returns ``None`` if that
    tile-month has no usable scene at all."""
    composite = get_month_composite(month)
    if composite is None or index_name not in composite["stats"]:
        return None
    array = composite["stats"][index_name][stat]
    return _reproject_index_band(
        array, composite["transform"], composite["crs_wkt"], dst_transform, dst_crs_wkt, dst_shape
    )


@dataclass(frozen=True)
class PixelBreakDate:
    row: int
    col: int
    break_date_ordinal: int
    magnitude: Optional[np.ndarray] = None  # the segment's per-feature break magnitude


@dataclass(frozen=True)
class BreakArrays:
    """Every break pixel's latest break as parallel arrays, sorted by break
    date ascending (so the pixels whose break is on/before any given date are
    a prefix -- see :func:`sweep_months`)."""

    rows: np.ndarray  # int64
    cols: np.ndarray  # int64
    ordinals: np.ndarray  # int64, ascending

    def __len__(self) -> int:
        return len(self.rows)


def latest_break_arrays(segments_result: dict) -> tuple[BreakArrays, np.ndarray]:
    """Vectorised :func:`latest_break_dates`: returns the arrays plus, for each
    entry, the index of the segment row it came from."""
    t_break = np.asarray(segments_result["t_break"])
    seg_idx = np.flatnonzero(t_break > 0)
    if len(seg_idx) == 0:
        empty = np.empty(0, dtype=np.int64)
        return BreakArrays(empty, empty, empty), empty
    rows = np.asarray(segments_result["row"])[seg_idx].astype(np.int64)
    cols = np.asarray(segments_result["col"])[seg_idx].astype(np.int64)
    ords = t_break[seg_idx].astype(np.int64)
    width = int(cols.max()) + 1
    key = rows * width + cols
    # latest break per pixel: sort by (pixel, date) and keep each pixel's last entry
    order = np.lexsort((ords, key))
    key_o = key[order]
    last = np.ones(len(order), dtype=bool)
    last[:-1] = key_o[1:] != key_o[:-1]
    keep = order[last]
    by_date = keep[np.argsort(ords[keep], kind="stable")]
    return BreakArrays(rows[by_date], cols[by_date], ords[by_date]), seg_idx[by_date]


def latest_break_dates(segments_result: dict) -> dict[tuple[int, int], PixelBreakDate]:
    """For every pixel with at least one detected break in a
    :mod:`change.change_detection` segment table (``t_break > 0``), its
    *most recent* break date (see module docstring)."""
    arrays, seg_idx = latest_break_arrays(segments_result)
    magnitude = segments_result.get("magnitude")
    return {
        (int(r), int(c)): PixelBreakDate(
            row=int(r), col=int(c), break_date_ordinal=int(o),
            magnitude=None if magnitude is None else magnitude[s],
        )
        for r, c, o, s in zip(arrays.rows, arrays.cols, arrays.ordinals, seg_idx)
    }


def month_bracketing(ordinal_date: int) -> tuple[str, str]:
    """The ``"YYYY-MM"`` month containing ``ordinal_date`` and the one
    immediately after it -- dNBR compares the composite immediately before
    a break to the one immediately after, so the "after" side starts at the
    month right after the break's own month."""
    from datetime import date

    d = date.fromordinal(ordinal_date)
    this_month = f"{d.year:04d}-{d.month:02d}"
    if d.month == 12:
        next_month = f"{d.year + 1:04d}-01"
    else:
        next_month = f"{d.year:04d}-{d.month + 1:02d}"
    return this_month, next_month


_EPOCH_ORDINAL = 719163  # date(1970, 1, 1).toordinal()


def _month_index_of_ordinals(ordinals: np.ndarray) -> np.ndarray:
    """Months since 1970-01 of each proleptic-Gregorian ordinal date."""
    days = (np.asarray(ordinals, dtype=np.int64) - _EPOCH_ORDINAL).astype("datetime64[D]")
    return days.astype("datetime64[M]").astype(np.int64)


def _month_name(index: int) -> str:
    return f"{1970 + index // 12:04d}-{index % 12 + 1:02d}"


def _month_parse(month: str) -> tuple[int, int]:
    """``"YYYY-MM"`` -> (months since 1970-01, ordinal of its first day)."""
    from datetime import date

    year, mon = (int(v) for v in month.split("-"))
    return (year - 1970) * 12 + mon - 1, date(year, mon, 1).toordinal()


def _iter_prefetched(months: list[str], fetch, workers: int):
    """Yield ``(month, fetch(month))`` in order, computing up to ``workers``
    months ahead in threads (a month's composite is I/O + numpy work that
    releases the GIL, and the consumer is comparatively instant)."""
    if workers <= 1 or len(months) <= 1:
        for month in months:
            yield month, fetch(month)
        return
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: deque = deque()
        it = iter(months)
        for month in it:
            pending.append((month, pool.submit(fetch, month)))
            if len(pending) >= workers + 1:
                break
        while pending:
            month, future = pending.popleft()
            nxt = next(it, None)
            if nxt is not None:
                pending.append((nxt, pool.submit(fetch, nxt)))
            yield month, future.result()


def sweep_months(
    get_month_composite,
    breaks: BreakArrays,
    trajectory_months: list[str],
    dst_transform,
    dst_crs_wkt: str,
    dst_shape: tuple[int, int],
    want_dnbr: bool = True,
    want_trajectory: bool = True,
    workers: int = 1,
    tile_id: str = "",
) -> tuple[Optional[dict], Optional[dict]]:
    """One pass over the months, everything vectorised.

    Each month's composite is built and reprojected exactly once and serves
    both outputs (the previous two-pass design rebuilt every month's
    composite -- both indices, all scenes -- once for dNBR and again for the
    trajectory, and looped over pixels in Python).

    - dNBR: a pixel's pre-break value is the NBR median of the month holding
      its break, the post value that of the following month (no outward
      search; both must be finite).
    - trajectory: NDVI median of every month in ``trajectory_months`` whose
      first day is on/after the pixel's break date.

    Returns ``(dnbr_result, trajectory_arrays)`` (``None`` for what was not
    asked for).
    """
    n = len(breaks)
    height, width = dst_shape
    prefix = f"[regrowth_severity] {tile_id}: " if tile_id else "[regrowth_severity] "

    break_month = _month_index_of_ordinals(breaks.ordinals) if n else np.empty(0, dtype=np.int64)
    pre_value = np.full(n, np.nan, dtype=np.float32)
    post_value = np.full(n, np.nan, dtype=np.float32)

    months: dict[str, None] = {}
    if want_dnbr and n:
        for idx in np.unique(np.concatenate([break_month, break_month + 1])):
            months[_month_name(int(idx))] = None
    if want_trajectory and n:
        earliest = int(breaks.ordinals[0])
        for m in trajectory_months:
            if _month_parse(m)[1] >= earliest:
                months[m] = None
    ordered = sorted(months)

    need_nbr = want_dnbr
    need_ndvi = want_trajectory

    def fetch(month: str):
        nbr = load_index_on_grid(get_month_composite, month, "NBR", "median", dst_transform, dst_crs_wkt, dst_shape) if need_nbr else None
        ndvi = load_index_on_grid(get_month_composite, month, "NDVI", "median", dst_transform, dst_crs_wkt, dst_shape) if need_ndvi else None
        return nbr, ndvi

    traj_pix: list[np.ndarray] = []
    traj_val: list[np.ndarray] = []
    traj_month: list[np.ndarray] = []
    month_names: list[str] = []

    for k, (month, (nbr, ndvi)) in enumerate(_iter_prefetched(ordered, fetch, workers), start=1):
        if k == 1 or k % 25 == 0 or k == len(ordered):
            print(f"{prefix}month {k}/{len(ordered)} ({month})", flush=True)
        m_idx, m_start = _month_parse(month)
        if nbr is not None:
            sel = np.flatnonzero(break_month == m_idx)
            if len(sel):
                pre_value[sel] = nbr[breaks.rows[sel], breaks.cols[sel]]
            sel = np.flatnonzero(break_month + 1 == m_idx)
            if len(sel):
                post_value[sel] = nbr[breaks.rows[sel], breaks.cols[sel]]
        if ndvi is not None:
            n_eligible = int(np.searchsorted(breaks.ordinals, m_start, side="right"))  # break on/before month start
            if n_eligible:
                values = ndvi[breaks.rows[:n_eligible], breaks.cols[:n_eligible]]
                finite = np.isfinite(values)
                pix = np.flatnonzero(finite)
                if len(pix):
                    traj_pix.append(pix.astype(np.int32))
                    traj_val.append(values[pix].astype(np.float32))
                    traj_month.append(np.full(len(pix), len(month_names), dtype=np.int16))
                    month_names.append(month)
                    continue
    dnbr_result = None
    if want_dnbr:
        dnbr = np.full((height, width), np.nan, dtype=np.float32)
        has = np.zeros((height, width), dtype=bool)
        pre_name = np.full((height, width), "", dtype="<U7")
        post_name = np.full((height, width), "", dtype="<U7")
        ok = np.flatnonzero(np.isfinite(pre_value) & np.isfinite(post_value))
        if len(ok):
            r, c = breaks.rows[ok], breaks.cols[ok]
            dnbr[r, c] = pre_value[ok] - post_value[ok]
            has[r, c] = True
            names = np.array([_month_name(int(i)) for i in range(int(break_month.min()), int(break_month.max()) + 2)])
            base = int(break_month.min())
            pre_name[r, c] = names[break_month[ok] - base]
            post_name[r, c] = names[break_month[ok] + 1 - base]
        dnbr_result = {"dnbr": dnbr, "has_dnbr": has, "pre_month": pre_name, "post_month": post_name}

    trajectory = None
    if want_trajectory:
        if traj_pix:
            pix = np.concatenate(traj_pix)
            val = np.concatenate(traj_val)
            mon = np.concatenate(traj_month)
            order = np.argsort(pix, kind="stable")  # pixel-major, months already ascending within a pixel
            pix, val, mon = pix[order], val[order], mon[order]
        else:
            pix = np.empty(0, dtype=np.int32)
            val = np.empty(0, dtype=np.float32)
            mon = np.empty(0, dtype=np.int16)
        trajectory = {
            "row": breaks.rows[pix].astype(np.int32),
            "col": breaks.cols[pix].astype(np.int32),
            "month_index": mon,
            "month_names": np.array(month_names, dtype="<U7"),
            "ndvi": val,
        }
    return dnbr_result, trajectory


def _breaks_from_dict(break_dates: dict[tuple[int, int], PixelBreakDate]) -> BreakArrays:
    if not break_dates:
        empty = np.empty(0, dtype=np.int64)
        return BreakArrays(empty, empty, empty)
    rows = np.array([k[0] for k in break_dates], dtype=np.int64)
    cols = np.array([k[1] for k in break_dates], dtype=np.int64)
    ords = np.array([pb.break_date_ordinal for pb in break_dates.values()], dtype=np.int64)
    order = np.argsort(ords, kind="stable")
    return BreakArrays(rows[order], cols[order], ords[order])


def compute_dnbr_for_breaks(
    get_month_composite,
    break_dates: dict[tuple[int, int], PixelBreakDate],
    dst_transform,
    dst_crs_wkt: str,
    dst_shape: tuple[int, int],
    tile_id: str = "",
) -> dict:
    """dNBR at every pixel with a detected break: the NBR median composite of
    the break's own month minus that of the following month (both must exist).

    Returns per-pixel arrays (on the break-detection grid) of ``dnbr``,
    ``pre_month``/``post_month`` (which composites were used), and
    ``has_dnbr`` (whether both sides were found at all).
    """
    result, _ = sweep_months(
        get_month_composite, _breaks_from_dict(break_dates), [], dst_transform, dst_crs_wkt, dst_shape,
        want_dnbr=True, want_trajectory=False, tile_id=tile_id,
    )
    return result


def compute_ndvi_regrowth_trajectory(
    get_month_composite,
    break_dates: dict[tuple[int, int], PixelBreakDate],
    all_months: list[str],
    dst_transform,
    dst_crs_wkt: str,
    dst_shape: tuple[int, int],
    tile_id: str = "",
) -> dict[tuple[int, int], list[tuple[str, float]]]:
    """For every pixel with a detected break, its NDVI-median trajectory
    across every monthly composite from the break's own month onward (months
    starting before the pixel's own break are skipped), as
    ``{(row, col): [(month, ndvi), ...]}``."""
    _, traj = sweep_months(
        get_month_composite, _breaks_from_dict(break_dates), sorted(all_months), dst_transform, dst_crs_wkt,
        dst_shape, want_dnbr=False, want_trajectory=True, tile_id=tile_id,
    )
    out: dict[tuple[int, int], list[tuple[str, float]]] = {key: [] for key in break_dates}
    names = traj["month_names"]
    for r, c, m, v in zip(traj["row"], traj["col"], traj["month_index"], traj["ndvi"]):
        out[(int(r), int(c))].append((str(names[m]), float(v)))
    return out


def regrowth_severity_output_path(output_root: str | Path, tile_id: str) -> Path:
    """``<output_root>/<tile_id>/regrowth_severity.npz``."""
    return Path(output_root) / tile_id / "regrowth_severity.npz"


def write_regrowth_severity_result(
    path: str | Path,
    dnbr_result: dict,
    trajectories,
    transform: tuple[float, ...],
    crs_wkt: str,
    resolution_m: float,
) -> Path:
    """dNBR as dense per-pixel arrays (one value per pixel of the tile grid,
    NaN where not applicable); NDVI regrowth trajectories flattened into
    parallel arrays (one row per (pixel, month) observation, months stored as
    an int16 index into ``trajectory_month_names``). ``trajectories`` is
    either :func:`sweep_months`'s array dict or the legacy
    ``{(row, col): [(month, ndvi), ...]}`` mapping."""
    if isinstance(trajectories, dict) and "month_index" in trajectories:
        traj = trajectories
    else:
        rows, cols, months, values = [], [], [], []
        for (row, col), points in trajectories.items():
            for month, value in points:
                rows.append(row)
                cols.append(col)
                months.append(month)
                values.append(value)
        names = sorted(set(months))
        lookup = {m: i for i, m in enumerate(names)}
        traj = {
            "row": np.array(rows, dtype=np.int32),
            "col": np.array(cols, dtype=np.int32),
            "month_index": np.array([lookup[m] for m in months], dtype=np.int16),
            "month_names": np.array(names, dtype="<U7"),
            "ndvi": np.array(values, dtype=np.float32),
        }

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        dnbr=dnbr_result["dnbr"],
        has_dnbr=dnbr_result["has_dnbr"],
        pre_month=np.asarray(dnbr_result["pre_month"]).astype(str),
        post_month=np.asarray(dnbr_result["post_month"]).astype(str),
        trajectory_row=traj["row"],
        trajectory_col=traj["col"],
        trajectory_month_index=traj["month_index"],
        trajectory_month_names=traj["month_names"],
        trajectory_ndvi=traj["ndvi"],
        transform=np.array(list(transform)[:6], dtype=np.float64),
        crs_wkt=np.array(crs_wkt),
        resolution_m=np.array(resolution_m, dtype=np.float64),
    )
    return out


def read_regrowth_severity_result(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as data:
        if "trajectory_month_index" in data.files:
            names = np.array(data["trajectory_month_names"])
            month_index = np.array(data["trajectory_month_index"])
            trajectory_month = names[month_index] if len(month_index) else np.array([], dtype="<U7")
        else:  # files written before months were stored as an index
            trajectory_month = np.array([str(m) for m in data["trajectory_month"]])
        return {
            "dnbr": np.array(data["dnbr"]),
            "has_dnbr": np.array(data["has_dnbr"]),
            "pre_month": np.array(data["pre_month"]),
            "post_month": np.array(data["post_month"]),
            "trajectory_row": np.array(data["trajectory_row"]),
            "trajectory_col": np.array(data["trajectory_col"]),
            "trajectory_month": trajectory_month,
            "trajectory_ndvi": np.array(data["trajectory_ndvi"]),
            "transform": tuple(float(v) for v in data["transform"]),
            "crs_wkt": str(data["crs_wkt"]),
            "resolution_m": float(data["resolution_m"]),
        }


def build_tile_regrowth_severity(
    tile_dir: str | Path,
    inference_root: str | Path,
    output_root: str | Path,
    tile_id: str,
    class_config: ClassConfig,
    segments_result: dict,
    transform: tuple[float, ...],
    crs_wkt: str,
    resolution_m: float,
    dst_shape: tuple[int, int],
    all_months: list[str],
    overwrite: bool = False,
    month_threads: int = 1,
) -> Optional[Path]:
    """Build one tile's dNBR + NDVI-regrowth result from a
    :mod:`change.change_detection` segment table, computing every month's
    NBR/NDVI composite it needs directly from stored scenes (see
    :func:`build_month_composite_cache`) -- no separate index-composite
    store is read or written, and every month is built once for both
    outputs (:func:`sweep_months`). Returns the written path, or ``None`` if
    already built and ``overwrite`` is false.
    """
    from affine import Affine

    out_path = regrowth_severity_output_path(output_root, tile_id)
    if out_path.is_file() and not overwrite:
        return None

    breaks, _ = latest_break_arrays(segments_result)
    print(f"[regrowth_severity] {tile_id}: {len(breaks)} pixels with a break", flush=True)
    dst_transform = Affine(*transform[:6])
    get_month_composite = build_month_composite_cache(tile_dir, inference_root, tile_id, class_config)
    dnbr_result, trajectory = sweep_months(
        get_month_composite, breaks, sorted(all_months), dst_transform, crs_wkt, dst_shape,
        workers=month_threads, tile_id=tile_id,
    )
    print(f"[regrowth_severity] {tile_id}: writing result to '{out_path}'...", flush=True)
    write_regrowth_severity_result(out_path, dnbr_result, trajectory, transform, crs_wkt, resolution_m)
    return out_path
