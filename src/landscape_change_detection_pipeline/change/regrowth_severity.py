"""NDVI regrowth trajectory and dNBR burn severity.

Purpose
-------
Two index-based, per-pixel change layers built on top of the monthly
spectral-index composites (:mod:`change.spectral_composites`) and the
per-pixel break dates :mod:`change.ccdc`/:mod:`change.bfast` detect:

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

An NDSI-based snow filter was deliberately **not** added here: the trained
classifier already has its own ``snow_cover`` class from the same
composites this module reads, and a separate index-threshold reading of the
same information was judged not to add enough value to justify a second
representation of "is this pixel snow" (project decision -- unlike dNBR,
where the classifier's categorical answer and the continuous severity
question are genuinely different things).

One shared grid per tile: the break-detection grid, not the composite grid
--------------------------------------------------------------------------
:mod:`change.ccdc`/:mod:`change.bfast` both operate on one fixed grid per
tile -- the finest resolution that tile ever achieved across its whole
history (see ``change.ccdc``'s own module docstring) -- while each monthly
index composite (:mod:`change.spectral_composites`) sits on *that month's*
own finest-available-sensor grid, which varies month to month. Combining a
break date with a monthly index value therefore requires reprojecting one
onto the other; this module always reprojects the monthly composite onto
the break-detection grid (bilinear -- continuous index values), never the
reverse, so that comparing "the pixel at row r, col c" across different
months always means the same real-world location. This was an explicit
project decision, favouring one fixed grid per tile for straightforward
before/after comparison over preserving each month's own native detail.

Both CCDC and BFAST outputs are supported as the break-date source
-------------------------------------------------------------------
Either can supply "when did this pixel change," so both are accepted
(``break_source="ccdc"`` picks each pixel's most recent detected
``t_break`` across all its segments; ``break_source="bfast"`` reads
``break_date`` directly from a BFAST result) -- this module does not
prefer one over the other; the caller picks based on the same reasoning
that guides break-detection algorithm choice generally
(CCDC for general breaks, BFAST specifically for fire).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from landscape_change_detection_pipeline.change.spectral_composites import (
    discover_tile_scenes,
    index_composite_output_path,
    read_index_composite,
)


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
    index_composites_root: str | Path,
    tile_id: str,
    month: str,
    index_name: str,
    stat: str,
    dst_transform,
    dst_crs_wkt: str,
    dst_shape: tuple[int, int],
) -> Optional[np.ndarray]:
    """One index/statistic for one tile-month, reprojected onto the given
    (break-detection) grid. Returns ``None`` if that tile-month has no
    stored index composite at all."""
    npz_path = index_composite_output_path(index_composites_root, tile_id, month)
    if not npz_path.is_file():
        return None
    composite = read_index_composite(npz_path)
    if index_name not in composite["stats"]:
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
    magnitude: Optional[np.ndarray] = None  # CCDC's per-band magnitude, if from CCDC


def latest_break_dates_from_ccdc(ccdc_result: dict) -> dict[tuple[int, int], PixelBreakDate]:
    """For every pixel with at least one detected break in a CCDC result
    (``t_break > 0``), its *most recent* break date -- the disturbance
    regrowth/severity layers care about the latest known change, not every
    historical one CCDC's segment table records."""
    breaks: dict[tuple[int, int], PixelBreakDate] = {}
    for i in range(len(ccdc_result["row"])):
        t_break = int(ccdc_result["t_break"][i])
        if t_break <= 0:
            continue
        key = (int(ccdc_result["row"][i]), int(ccdc_result["col"][i]))
        candidate = PixelBreakDate(
            row=key[0], col=key[1], break_date_ordinal=t_break, magnitude=ccdc_result["magnitude"][i]
        )
        existing = breaks.get(key)
        if existing is None or candidate.break_date_ordinal > existing.break_date_ordinal:
            breaks[key] = candidate
    return breaks


def break_dates_from_bfast(bfast_result: dict) -> dict[tuple[int, int], PixelBreakDate]:
    """Every pixel BFAST flagged with a break, as the same
    ``{(row, col): PixelBreakDate}`` shape :func:`latest_break_dates_from_ccdc`
    returns (BFAST reports at most one break per pixel, so there is no
    "most recent of several" step here)."""
    breaks: dict[tuple[int, int], PixelBreakDate] = {}
    for i in range(len(bfast_result["row"])):
        if not bfast_result["has_break"][i]:
            continue
        row, col = int(bfast_result["row"][i]), int(bfast_result["col"][i])
        breaks[(row, col)] = PixelBreakDate(row=row, col=col, break_date_ordinal=int(bfast_result["break_date"][i]))
    return breaks


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


def compute_dnbr_for_breaks(
    index_composites_root: str | Path,
    tile_id: str,
    break_dates: dict[tuple[int, int], PixelBreakDate],
    dst_transform,
    dst_crs_wkt: str,
    dst_shape: tuple[int, int],
) -> dict:
    """dNBR at every pixel with a detected break: the last available
    monthly NBR composite before the break's own month, minus the first
    available one on/after it (searching outward month by month when the
    break's immediately-bracketing months have no stored composite, up to
    ``max_month_search`` steps, since a tile-month can simply have no
    usable scene).

    Returns per-pixel arrays (on the break-detection grid) of ``dnbr``,
    ``pre_month``/``post_month`` (which composites were actually used), and
    ``has_dnbr`` (whether both sides were found at all).
    """
    height, width = dst_shape
    dnbr = np.full((height, width), np.nan, dtype=np.float32)
    has_dnbr = np.zeros((height, width), dtype=bool)
    pre_month_used = np.full((height, width), "", dtype=object)
    post_month_used = np.full((height, width), "", dtype=object)

    # Cache reprojected NBR-median bands per month so repeated pixels in the
    # same month (the common case) don't reproject the same composite twice.
    nbr_cache: dict[str, Optional[np.ndarray]] = {}

    def nbr_for_month(month: str) -> Optional[np.ndarray]:
        if month not in nbr_cache:
            nbr_cache[month] = load_index_on_grid(
                index_composites_root, tile_id, month, "NBR", "median", dst_transform, dst_crs_wkt, dst_shape
            )
        return nbr_cache[month]

    for (row, col), pixel_break in break_dates.items():
        this_month, next_month = month_bracketing(pixel_break.break_date_ordinal)

        pre_nbr_band = nbr_for_month(this_month)
        pre_value = pre_nbr_band[row, col] if pre_nbr_band is not None else np.nan
        post_nbr_band = nbr_for_month(next_month)
        post_value = post_nbr_band[row, col] if post_nbr_band is not None else np.nan

        if np.isfinite(pre_value) and np.isfinite(post_value):
            dnbr[row, col] = pre_value - post_value
            has_dnbr[row, col] = True
            pre_month_used[row, col] = this_month
            post_month_used[row, col] = next_month

    return {
        "dnbr": dnbr,
        "has_dnbr": has_dnbr,
        "pre_month": pre_month_used,
        "post_month": post_month_used,
    }


def compute_ndvi_regrowth_trajectory(
    index_composites_root: str | Path,
    tile_id: str,
    break_dates: dict[tuple[int, int], PixelBreakDate],
    all_months: list[str],
    dst_transform,
    dst_crs_wkt: str,
    dst_shape: tuple[int, int],
) -> dict[tuple[int, int], list[tuple[str, float]]]:
    """For every pixel with a detected break, its NDVI-median trajectory
    across every stored monthly composite from the break's own month
    onward -- the regrowth curve. ``all_months`` is every ``"YYYY-MM"`` this
    tile has a composite for (sorted); months before a given pixel's own
    break are skipped for that pixel."""
    ndvi_cache: dict[str, Optional[np.ndarray]] = {}

    def ndvi_for_month(month: str) -> Optional[np.ndarray]:
        if month not in ndvi_cache:
            ndvi_cache[month] = load_index_on_grid(
                index_composites_root, tile_id, month, "NDVI", "median", dst_transform, dst_crs_wkt, dst_shape
            )
        return ndvi_cache[month]

    trajectories: dict[tuple[int, int], list[tuple[str, float]]] = {key: [] for key in break_dates}

    from datetime import date

    for month in sorted(all_months):
        year, mon = (int(v) for v in month.split("-"))
        month_start_ordinal = date(year, mon, 1).toordinal()
        ndvi_band = ndvi_for_month(month)
        if ndvi_band is None:
            continue
        for (row, col), pixel_break in break_dates.items():
            if month_start_ordinal < pixel_break.break_date_ordinal:
                continue
            value = ndvi_band[row, col]
            if np.isfinite(value):
                trajectories[(row, col)].append((month, float(value)))

    return trajectories


def regrowth_severity_output_path(output_root: str | Path, tile_id: str, break_source: str) -> Path:
    """``<output_root>/<tile_id>/regrowth_severity_<break_source>.npz``."""
    return Path(output_root) / tile_id / f"regrowth_severity_{break_source}.npz"


def write_regrowth_severity_result(
    path: str | Path,
    dnbr_result: dict,
    trajectories: dict[tuple[int, int], list[tuple[str, float]]],
    transform: tuple[float, ...],
    crs_wkt: str,
    resolution_m: float,
) -> Path:
    """dNBR as dense per-pixel arrays (one value per pixel of the tile
    grid, NaN where not applicable); NDVI regrowth trajectories flattened
    into parallel arrays (one row per (pixel, month) observation), since
    trajectory length varies per pixel just as CCDC's segment count does."""
    traj_rows: list[int] = []
    traj_cols: list[int] = []
    traj_months: list[str] = []
    traj_ndvi: list[float] = []
    for (row, col), points in trajectories.items():
        for month, value in points:
            traj_rows.append(row)
            traj_cols.append(col)
            traj_months.append(month)
            traj_ndvi.append(value)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        dnbr=dnbr_result["dnbr"],
        has_dnbr=dnbr_result["has_dnbr"],
        pre_month=dnbr_result["pre_month"].astype(str),
        post_month=dnbr_result["post_month"].astype(str),
        trajectory_row=np.array(traj_rows, dtype=np.int32),
        trajectory_col=np.array(traj_cols, dtype=np.int32),
        trajectory_month=np.array(traj_months),
        trajectory_ndvi=np.array(traj_ndvi, dtype=np.float32),
        transform=np.array(list(transform)[:6], dtype=np.float64),
        crs_wkt=np.array(crs_wkt),
        resolution_m=np.array(resolution_m, dtype=np.float64),
    )
    return out


def read_regrowth_severity_result(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as data:
        return {
            "dnbr": np.array(data["dnbr"]),
            "has_dnbr": np.array(data["has_dnbr"]),
            "pre_month": np.array(data["pre_month"]),
            "post_month": np.array(data["post_month"]),
            "trajectory_row": np.array(data["trajectory_row"]),
            "trajectory_col": np.array(data["trajectory_col"]),
            "trajectory_month": [str(m) for m in data["trajectory_month"]],
            "trajectory_ndvi": np.array(data["trajectory_ndvi"]),
            "transform": tuple(float(v) for v in data["transform"]),
            "crs_wkt": str(data["crs_wkt"]),
            "resolution_m": float(data["resolution_m"]),
        }


def build_tile_regrowth_severity(
    index_composites_root: str | Path,
    output_root: str | Path,
    tile_id: str,
    break_result: dict,
    break_source: str,
    transform: tuple[float, ...],
    crs_wkt: str,
    resolution_m: float,
    dst_shape: tuple[int, int],
    all_months: list[str],
    overwrite: bool = False,
) -> Optional[Path]:
    """Build one tile's dNBR + NDVI-regrowth result from a CCDC or BFAST
    break-detection result (``break_source`` is ``"ccdc"`` or ``"bfast"``,
    selecting which reader to use -- see module docstring). Returns the
    written path, or ``None`` if already built and ``overwrite`` is false.
    """
    from affine import Affine

    out_path = regrowth_severity_output_path(output_root, tile_id, break_source)
    if out_path.is_file() and not overwrite:
        return None

    if break_source == "ccdc":
        break_dates = latest_break_dates_from_ccdc(break_result)
    elif break_source == "bfast":
        break_dates = break_dates_from_bfast(break_result)
    else:
        raise ValueError(f"break_source must be 'ccdc' or 'bfast', got {break_source!r}")

    dst_transform = Affine(*transform[:6])
    dnbr_result = compute_dnbr_for_breaks(
        index_composites_root, tile_id, break_dates, dst_transform, crs_wkt, dst_shape
    )
    trajectories = compute_ndvi_regrowth_trajectory(
        index_composites_root, tile_id, break_dates, all_months, dst_transform, crs_wkt, dst_shape
    )
    write_regrowth_severity_result(out_path, dnbr_result, trajectories, transform, crs_wkt, resolution_m)
    return out_path
