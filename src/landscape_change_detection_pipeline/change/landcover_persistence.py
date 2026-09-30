"""Persistent land-cover object tracking: appearance/disappearance dates.

Purpose
-------
Stage 7's monthly composites answer "what class is this pixel this month" --
a state, not an object with a lifetime. Several classes in
``configs/classes.yaml`` (``cultivated_agriculture``, ``cutblock_harvest``,
``built_up_infrastructure``, ``burned_disturbed``) are conceptually
persistent objects rather than instantaneous states: a field is planted in a
given month and (maybe, eventually) abandoned back to another cover; a
cutblock appears the month harvest starts and (maybe, eventually) regrows
back to forest; a burn appears at its detected break date and may recover.
This module walks each pixel's monthly classification history for one
tracked class at a time and turns it into a small number of dated intervals
("active from month X" / "ended at month Y, still active" / "" ), never a
per-pixel value.

Noise handling: a sliding-window majority vote, not a raw month-to-month flip
--------------------------------------------------------------------------------
A single misclassified month (residual cloud, a transitional/ambiguous
scene) must not register as a fake appearance or disappearance. Each
pixel's raw monthly boolean series ("is this month's composite class == the
tracked class") is smoothed with a sliding window over *observed* months
only (a tile-month with no stored composite at all is skipped, never treated
as "class absent that month" -- an unobserved month must not count as
evidence either way): within any window of ``window_size`` observed months,
at least ``min_fraction`` of them must show the tracked class for the
window's last month to be marked "confirmed present". This is a proportional
majority vote, not a strict N-consecutive-months rule, precisely so one
aberrant month inside an otherwise-consistent run does not reset the count
the way a consecutive-run requirement would, while still not reacting to a
single flipped month the way a threshold of 1 would.

Unobserved months, hysteresis, minimum duration
------------------------------------------------
Cloud, shadow, snow and ice months are *unobserved* (``masked_classes``):
they are skipped by the vote, neither for nor against the class -- a cutblock
under snow has not disappeared and cloud must not dilute the vote. The window
is the last ``window_size`` *observed* months of each pixel. A class is
confirmed when at least ``min_fraction`` of the window shows it and ends when
at most ``end_fraction`` does (hysteresis: one odd month never closes an
interval); intervals shorter than ``min_duration_months`` are dropped.

``burned_disturbed`` is a special case: reuses Stage 09/11 instead of
tracking independently
-------------------------------------------------------------------------
Every other tracked class is detected purely from Stage 7's monthly class
composites. ``burned_disturbed`` is different: Stage 09 (per-pixel
segmentation) already detects the disturbance date precisely (a real break
in the spectral time series, not a monthly-composite majority vote), and
Stage 11 (regrowth_severity) already tracks the NDVI trajectory afterward.
Re-running the generic class-persistence detector on ``burned_disturbed``
would produce a second, cruder date estimate for the same event from a
strictly weaker signal (monthly class majority vs. per-scene break
detection) -- so this module's ``burned_disturbed`` handling instead reads
a Stage 09 break (an NBR/NDVI fall of at least ``min_break_drop`` within
``break_tolerance_months`` before / ``break_after_months`` after the vote's
start) as the required evidence and its month as the appearance date; the end
(recovery) comes from the vote. ``require_break_classes`` lists the classes
handled that way (default burned and cutblock: abrupt events).

Output
------
Per pixel per tracked class, zero or more intervals
``(start_month, end_month_or_None)`` -- ``end_month=None`` means still
active as of the last observed month. Written as flattened parallel arrays
(one row per (pixel, class, interval)), the same pattern
``change.change_detection``'s segment table and
``change.regrowth_severity``'s trajectory table use for one-to-many
per-pixel records. Grouping into connected-component polygons (one cutblock
= one object with one age) is left to a downstream GIS step, not done here --
this module's own unit is still the pixel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numba import njit, prange

from landscape_change_detection_pipeline.classes.class_config import ClassConfig
from landscape_change_detection_pipeline.inference.composites import composite_output_path, read_composite

#: Classes tracked as persistent objects by default -- see module docstring
#: for why ``burned_disturbed`` is handled differently within this module
#: rather than being excluded.
DEFAULT_TRACKED_CLASSES: tuple[str, ...] = ("burned_disturbed",)

#: Sliding-window majority vote defaults (see module docstring). Both are
#: exposed as config so real-data results (too much/too little noise) can be
#: tuned without a code change.
DEFAULT_WINDOW_SIZE = 5
DEFAULT_MIN_FRACTION = 0.8


def _reproject_nearest_uint8(array, src_transform, src_crs_wkt, dst_transform, dst_crs_wkt, dst_shape, nodata):
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    destination = np.full(dst_shape, nodata, dtype=np.uint8)
    reproject(
        source=array.astype(np.uint8),
        destination=destination,
        src_transform=Affine(*src_transform[:6]),
        src_crs=CRS.from_user_input(src_crs_wkt),
        dst_transform=Affine(*dst_transform[:6]),
        dst_crs=CRS.from_user_input(dst_crs_wkt),
        resampling=Resampling.nearest,
        src_nodata=nodata,
        dst_nodata=nodata,
    )
    return destination


@dataclass(frozen=True)
class MonthlyClassStack:
    """Every tile-month's classification composite, reprojected onto one
    shared reference grid and stacked in chronological order."""

    months: tuple[str, ...]  # sorted "YYYY-MM", one per stack layer
    classes: np.ndarray  # (n_months, H, W) uint8, nodata where unobserved
    nodata: int
    transform: tuple[float, ...]
    crs_wkt: str
    shape: tuple[int, int]


def build_monthly_class_stack(
    composites_root: str | Path,
    tile_id: str,
    all_months: list[str],
    dst_transform,
    dst_crs_wkt: str,
    dst_shape: tuple[int, int],
    nodata: int = 255,
) -> MonthlyClassStack:
    """Read every stored monthly composite for this tile and reproject each
    onto ``(dst_transform, dst_crs_wkt, dst_shape)`` -- the same grid Stage
    09's segmentation and Stage 13's change maps compare on, so a pixel at
    ``(row, col)`` means the same real-world location across every module.
    A month with no stored composite is simply absent from the stack (never
    filled with nodata as a synthetic observation) -- see module docstring
    on why an unobserved month must not count as evidence."""
    months_sorted = sorted(all_months)
    layers: list[np.ndarray] = []
    kept_months: list[str] = []

    for i, month in enumerate(months_sorted, start=1):
        print(f"[landcover_persistence] {tile_id}: reading month {i}/{len(months_sorted)} ({month})", flush=True)
        npz_path = composite_output_path(composites_root, tile_id, month)
        if not npz_path.is_file():
            continue
        composite = read_composite(npz_path)
        reprojected = _reproject_nearest_uint8(
            composite["composite"],
            composite["transform"],
            composite["crs_wkt"],
            dst_transform,
            dst_crs_wkt,
            dst_shape,
            composite["nodata"],
        )
        # Normalize this month's own nodata sentinel to the stack's shared one.
        if composite["nodata"] != nodata:
            reprojected = np.where(reprojected == composite["nodata"], nodata, reprojected).astype(np.uint8)
        layers.append(reprojected)
        kept_months.append(month)

    if not layers:
        raise ValueError(f"No stored monthly composites found for tile '{tile_id}'")

    return MonthlyClassStack(
        months=tuple(kept_months),
        classes=np.stack(layers, axis=0),
        nodata=nodata,
        transform=tuple(dst_transform)[:6],
        crs_wkt=dst_crs_wkt,
        shape=dst_shape,
    )


@dataclass(frozen=True)
class PixelInterval:
    row: int
    col: int
    start_month: str
    end_month: Optional[str]  # None = still active as of the last observed month


@njit(parallel=True, cache=True)
def _track_kernel(series, target, masked, window, min_hits, end_hits, out_start, out_end):
    """Per-pixel interval tracker over ``series`` ``(n_pix, n_months)`` uint8.

    Only *observed* months count (``masked[value]`` False): a month that is
    cloud, shadow, snow, ice or no-data is neither for nor against the class,
    it is skipped. Over the last ``window`` observed months, the class is
    confirmed once at least ``min_hits`` of them show it (the interval starts
    at the earliest hit of that window) and ends once at most ``end_hits`` do
    (the interval ends at the first observed month after the last hit). The
    gap between ``min_hits`` and ``end_hits`` is a hysteresis: an interval is
    not closed by a single odd month. ``out_end`` is -1 for an interval still
    active at the end of the series."""
    n_pix, n_m = series.shape
    max_i = out_start.shape[1]
    for p in prange(n_pix):
        ring_hit = np.zeros(window, np.int64)
        ring_m = np.zeros(window, np.int64)
        cnt = 0
        hits = 0
        last_hit = -1
        start_m = -1
        state = False
        n_int = 0
        for m in range(n_m):
            v = series[p, m]
            if masked[v]:
                continue
            h = 1 if v == target else 0
            slot = cnt % window
            if cnt >= window:
                hits -= ring_hit[slot]
            ring_hit[slot] = h
            ring_m[slot] = m
            hits += h
            cnt += 1
            if h:
                last_hit = m
            if cnt < window:
                continue
            if not state:
                if hits >= min_hits:
                    for j in range(window):
                        idx = (cnt - window + j) % window
                        if ring_hit[idx]:
                            start_m = ring_m[idx]
                            break
                    state = True
            elif hits <= end_hits:
                end_m = m
                for j in range(window):
                    idx = (cnt - window + j) % window
                    if ring_m[idx] > last_hit:
                        end_m = ring_m[idx]
                        break
                if n_int < max_i:
                    out_start[p, n_int] = start_m
                    out_end[p, n_int] = end_m
                    n_int += 1
                state = False
        if state and n_int < max_i:
            out_start[p, n_int] = start_m
            out_end[p, n_int] = -1
            n_int += 1


def _abs_month(month: str) -> int:
    year, mon = (int(v) for v in month.split("-"))
    return (year - 1970) * 12 + mon - 1


def _name_of_abs_month(index: int) -> str:
    return f"{1970 + index // 12:04d}-{index % 12 + 1:02d}"


def _break_table(segments_result: dict, min_drop_features=("NBR", "NDVI")):
    """Every Stage 09 break as sorted parallel arrays ``(pixel_key, month,
    drop)``: ``month`` is months since 1970-01 of the break date, ``drop`` the
    largest fall of NBR/NDVI across it (model-residual magnitude when the
    segment table has it, the raw before/after difference otherwise)."""
    from landscape_change_detection_pipeline.change.regrowth_severity import _month_index_of_ordinals

    t_break = np.asarray(segments_result["t_break"])
    sel = np.flatnonzero(t_break > 0)
    height, width = segments_result["shape"]
    key = (np.asarray(segments_result["row"])[sel].astype(np.int64) * width + np.asarray(segments_result["col"])[sel])
    month = _month_index_of_ordinals(t_break[sel])
    names = list(segments_result["feature_names"])
    mag = segments_result.get("resid_magnitude", segments_result["magnitude"])
    cols = [names.index(n) for n in min_drop_features if n in names]
    drop = np.max(-np.asarray(mag)[sel][:, cols], axis=1) if len(sel) and cols else np.zeros(len(sel))
    order = np.argsort(key, kind="stable")
    return key[order], month[order], drop[order]


def detect_intervals_for_class(
    stack: MonthlyClassStack,
    class_id: int,
    window_size: int = DEFAULT_WINDOW_SIZE,
    min_fraction: float = DEFAULT_MIN_FRACTION,
    *,
    masked_ids: tuple[int, ...] = (),
    end_fraction: float = 0.2,
    min_duration_months: int = 0,
    segments_result: Optional[dict] = None,
    require_break: bool = False,
    break_tolerance_months: int = 18,
    break_after_months: int = 6,
    min_break_drop: float = 0.1,
    max_intervals: int = 8,
) -> list[PixelInterval]:
    """Confirmed-presence intervals of ``class_id`` for every pixel of
    ``stack`` (see :func:`_track_kernel` for the vote and its hysteresis).

    ``masked_ids`` are classes whose months are *unobserved* (cloud, shadow,
    snow, ice): they never count as evidence for or against the class, so a
    cutblock under snow is not "gone" and cloud does not dilute the vote.
    Intervals shorter than ``min_duration_months`` are dropped. With
    ``require_break`` (and a Stage 09 ``segments_result``) an interval is kept
    only if the pixel has a break between ``break_tolerance_months`` before
    and ``break_after_months`` after its start with a NBR/NDVI fall of at
    least ``min_break_drop``; its start date is then that break's month. A
    class flash without a spectral break behind it is not a disturbance."""
    n_months, height, width = stack.classes.shape
    if n_months < window_size:
        return []
    masked = np.zeros(256, dtype=np.bool_)
    masked[stack.nodata] = True
    for c in masked_ids:
        masked[c] = True
    min_hits = int(np.ceil(min_fraction * window_size - 1e-9))
    end_hits = int(np.floor(end_fraction * window_size + 1e-9))
    if end_hits >= min_hits:
        raise ValueError(f"end_fraction ({end_fraction}) must give fewer hits than min_fraction ({min_fraction}) for window_size={window_size}")

    flat = np.ascontiguousarray(stack.classes.reshape(n_months, -1).T)  # (n_pix, n_months)
    candidates = np.flatnonzero((flat == class_id).any(axis=1))
    if len(candidates) == 0:
        return []
    series = np.ascontiguousarray(flat[candidates])
    out_start = np.full((len(candidates), max_intervals), -2, dtype=np.int32)
    out_end = np.full((len(candidates), max_intervals), -2, dtype=np.int32)
    _track_kernel(series, int(class_id), masked, int(window_size), min_hits, end_hits, out_start, out_end)

    pix_i, slot = np.nonzero(out_start >= 0)
    if len(pix_i) == 0:
        return []
    s_idx = out_start[pix_i, slot]
    e_idx = out_end[pix_i, slot]
    pixel = candidates[pix_i]
    abs_months = np.array([_abs_month(m) for m in stack.months], dtype=np.int64)
    start_abs = abs_months[s_idx]
    end_abs = np.where(e_idx >= 0, abs_months[np.maximum(e_idx, 0)], -1)
    last_abs = abs_months[-1]
    duration = np.where(e_idx >= 0, end_abs, last_abs + 1) - start_abs
    keep = duration >= min_duration_months

    if require_break and segments_result is not None:
        b_key, b_month, b_drop = _break_table(segments_result)
        if len(b_key) == 0:
            keep[:] = False
        else:
            lo = np.searchsorted(b_key, pixel, side="left")
            hi = np.searchsorted(b_key, pixel, side="right")
            best_dist = np.full(len(pixel), np.iinfo(np.int64).max, dtype=np.int64)
            best_month = np.zeros(len(pixel), dtype=np.int64)
            for j in range(int((hi - lo).max()) if len(pixel) else 0):
                idx = lo + j
                ok = idx < hi
                safe = np.minimum(idx, len(b_key) - 1)
                bm, dr = b_month[safe], b_drop[safe]
                gap = bm - start_abs
                match = ok & (gap >= -break_tolerance_months) & (gap <= break_after_months) & (dr >= min_break_drop)
                better = match & (np.abs(gap) < best_dist)
                best_dist = np.where(better, np.abs(gap), best_dist)
                best_month = np.where(better, bm, best_month)
            supported = best_dist < np.iinfo(np.int64).max
            keep &= supported
            start_abs = np.where(supported, best_month, start_abs)

    width_ = width
    rows = (pixel // width_).astype(np.int64)
    cols = (pixel % width_).astype(np.int64)
    out: list[PixelInterval] = []
    for k in np.flatnonzero(keep):
        out.append(PixelInterval(
            row=int(rows[k]), col=int(cols[k]),
            start_month=_name_of_abs_month(int(start_abs[k])),
            end_month=None if e_idx[k] < 0 else stack.months[int(e_idx[k])],
        ))
    return out


def detect_burned_disturbed_intervals(
    stack: MonthlyClassStack,
    class_id: int,
    segments_result: dict,
    window_size: int = DEFAULT_WINDOW_SIZE,
    min_fraction: float = DEFAULT_MIN_FRACTION,
    **options,
) -> list[PixelInterval]:
    """``burned_disturbed`` intervals: the classifier's monthly class vote
    *and* a Stage 09 break with a real NBR/NDVI fall at its start (whose month
    becomes the interval start). See :func:`detect_intervals_for_class`."""
    return detect_intervals_for_class(
        stack, class_id, window_size, min_fraction, segments_result=segments_result, require_break=True, **options
    )


def _month_ordinal(month: str) -> int:
    year, mon = (int(v) for v in month.split("-"))
    return year * 12 + mon


def landcover_persistence_output_path(output_root: str | Path, tile_id: str) -> Path:
    """``<output_root>/<tile_id>/landcover_persistence.npz``."""
    return Path(output_root) / tile_id / "landcover_persistence.npz"


def write_landcover_persistence_result(
    path: str | Path,
    intervals_by_class: dict[str, list[PixelInterval]],
    transform: tuple[float, ...],
    crs_wkt: str,
    resolution_m: float,
    shape: tuple[int, int],
) -> Path:
    """Every tracked class's intervals flattened into parallel arrays (one
    row per (class, pixel, interval)), the same one-to-many pattern used by
    ``change.change_detection``'s segment table."""
    class_names: list[str] = []
    rows: list[int] = []
    cols: list[int] = []
    start_months: list[str] = []
    end_months: list[str] = []  # "" sentinel for still-active (None)

    for class_name, intervals in intervals_by_class.items():
        for interval in intervals:
            class_names.append(class_name)
            rows.append(interval.row)
            cols.append(interval.col)
            start_months.append(interval.start_month)
            end_months.append(interval.end_month if interval.end_month is not None else "")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        class_name=np.array(class_names),
        row=np.array(rows, dtype=np.int32),
        col=np.array(cols, dtype=np.int32),
        start_month=np.array(start_months),
        end_month=np.array(end_months),
        transform=np.array(list(transform)[:6], dtype=np.float64),
        crs_wkt=np.array(crs_wkt),
        resolution_m=np.array(resolution_m, dtype=np.float64),
        shape=np.array(shape, dtype=np.int32),
    )
    return out


def read_landcover_persistence_result(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as data:
        return {
            "class_name": np.array([str(v) for v in data["class_name"]]),
            "row": np.array(data["row"]),
            "col": np.array(data["col"]),
            "start_month": np.array([str(v) for v in data["start_month"]]),
            "end_month": np.array([str(v) for v in data["end_month"]]),
            "transform": tuple(float(v) for v in data["transform"]),
            "crs_wkt": str(data["crs_wkt"]),
            "resolution_m": float(data["resolution_m"]),
            "shape": tuple(int(v) for v in data["shape"]),
        }


def build_tile_landcover_persistence(
    composites_root: str | Path,
    output_root: str | Path,
    tile_id: str,
    class_config: ClassConfig,
    all_months: list[str],
    dst_transform,
    dst_crs_wkt: str,
    dst_shape: tuple[int, int],
    resolution_m: float,
    segments_result: Optional[dict] = None,
    tracked_classes: tuple[str, ...] = DEFAULT_TRACKED_CLASSES,
    window_size: int = DEFAULT_WINDOW_SIZE,
    min_fraction: float = DEFAULT_MIN_FRACTION,
    overwrite: bool = False,
    masked_classes: tuple[str, ...] = ("cloud", "shadow", "snow_cover", "ice_cover"),
    end_fraction: float = 0.2,
    min_duration_months: Optional[dict] = None,
    default_min_duration_months: int = 12,
    require_break_classes: tuple[str, ...] = ("burned_disturbed",),
    break_tolerance_months: int = 18,
    break_after_months: int = 6,
    min_break_drop: float = 0.1,
) -> Optional[Path]:
    """Build one tile's persistent land-cover object intervals for every
    class in ``tracked_classes``. Returns the written path, or ``None`` if
    already built and ``overwrite`` is false."""
    out_path = landcover_persistence_output_path(output_root, tile_id)
    if out_path.is_file() and not overwrite:
        return None

    print(f"[landcover_persistence] {tile_id}: building monthly class stack ({len(all_months)} months available)", flush=True)
    stack = build_monthly_class_stack(composites_root, tile_id, all_months, dst_transform, dst_crs_wkt, dst_shape)
    print(f"[landcover_persistence] {tile_id}: stack built ({len(stack.months)} months kept)", flush=True)

    masked_ids = tuple(class_config.by_name(n).id for n in masked_classes)
    intervals_by_class: dict[str, list[PixelInterval]] = {}
    for i, class_name in enumerate(tracked_classes, start=1):
        print(f"[landcover_persistence] {tile_id}: tracking class {i}/{len(tracked_classes)} ({class_name})", flush=True)
        class_id = class_config.by_name(class_name).id
        if not (stack.classes == class_id).any():
            print(
                f"[landcover_persistence] {tile_id}: WARNING '{class_name}' is never predicted in this tile's "
                "monthly composites (the classifier has no training samples for it?) -- no interval possible",
                flush=True,
            )
        need_break = class_name in require_break_classes and segments_result is not None
        intervals_by_class[class_name] = detect_intervals_for_class(
            stack, class_id, window_size, min_fraction,
            masked_ids=masked_ids, end_fraction=end_fraction,
            min_duration_months=int((min_duration_months or {}).get(class_name, default_min_duration_months)),
            segments_result=segments_result, require_break=need_break,
            break_tolerance_months=break_tolerance_months, break_after_months=break_after_months,
            min_break_drop=min_break_drop,
        )
        print(f"[landcover_persistence] {tile_id}: {class_name}: {len(intervals_by_class[class_name])} intervals found", flush=True)

    print(f"[landcover_persistence] {tile_id}: writing result to '{out_path}'...", flush=True)
    write_landcover_persistence_result(out_path, intervals_by_class, dst_transform, dst_crs_wkt, resolution_m, dst_shape)
    return out_path
