"""Combined annual/month-to-month change maps.

Purpose
-------
The final integration layer: combines classification-based change (Stage 7's
monthly composites, compared across periods), CCDC/BFAST break detection,
and dNBR into one landscape-change output per tile per period pair -- as
separate bands, never collapsed into one opaque "changed" bit, so a
downstream consumer (an external pipeline, or a human analyst) can
weight each signal independently. Per the project's explicit direction,
three classification-change bands at increasing confidence are kept side by
side rather than choosing one:

- **raw**: a plain two-period classification comparison
  (``from_class``/``to_class``/``changed``). Cheap, immediate, but noisy --
  a pixel sitting near a classification decision boundary can flip classes
  between two periods with no real disturbance behind it at all.
- **persistent**: the same comparison, but ``changed`` only fires when the
  *to*-period class also holds for at least one more consecutive period
  after it (``persistence_periods``, default 2) -- filters single-period
  classification noise while still catching short-lived real events (a
  short persistence window was chosen deliberately, per the project's
  explicit direction, precisely so a real-but-brief disturbance is not
  filtered out the way a long persistence requirement would).
- **corroborated**: ``changed`` (raw) narrowed to pixels where CCDC or
  BFAST *also* detected a break near the same date -- the strongest
  available confidence signal, at the cost of only covering pixels/periods
  where break detection actually ran successfully.

Cloud/shadow handling
------------------------
Per explicit project requirement: a pixel flagged ``cloud``/``shadow``
(``class_config.ClassDef.change_eligible=False``) in *either* period being
compared is treated as no-data for that comparison, never as evidence of
change -- comparing a real class against an imaging artifact would produce
a meaningless "changed" flag purely from cloud cover moving between
periods, not because the land cover itself changed.

Two comparison modes
------------------------
- **Annual** (:func:`build_annual_change_map`): the same calendar month
  across two consecutive years (e.g. August 2023 vs. August 2024) -- shows
  what changes most year-over-year for a fixed seasonal reference point,
  avoiding the seasonal-signal confound a month-to-month comparison across
  different seasons would introduce.
- **Month-to-month** (:func:`build_month_to_month_change_map`): any two
  consecutive stored months within one time span, regardless of season --
  catches faster within-year change (e.g. a summer fire) an annual-only
  comparison would only notice a year later.

Grid
----
All comparisons happen on the break-detection grid
(:mod:`change.ccdc`/:mod:`change.bfast`'s fixed per-tile grid), for the same
reason :mod:`change.regrowth_severity` reprojects onto it rather than the
reverse: this module reads the same monthly
composites and break results that module does.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from landscape_change_detection_pipeline.classes.class_config import ClassConfig
from landscape_change_detection_pipeline.inference.composites import composite_output_path, read_composite

#: How close (in days) a break-detection date must fall to a classification
#: change's own period boundary to count as corroborating it -- a break
#: detected weeks before/after the compared periods' exact boundary is still
#: describing the same real-world event, given monthly-composite granularity.
DEFAULT_CORROBORATION_WINDOW_DAYS = 45


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


def load_composite_on_grid(
    composites_root: str | Path,
    tile_id: str,
    month: str,
    dst_transform,
    dst_crs_wkt: str,
    dst_shape: tuple[int, int],
) -> Optional[tuple[np.ndarray, int]]:
    """One tile-month's classification composite, reprojected (nearest-
    neighbour -- categorical labels) onto the break-detection grid. Returns
    ``(class_array, nodata)``, or ``None`` if this tile-month has no stored
    composite."""
    npz_path = composite_output_path(composites_root, tile_id, month)
    if not npz_path.is_file():
        return None
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
    return reprojected, composite["nodata"]


def _valid_mask(class_array: np.ndarray, nodata: int, class_config: ClassConfig) -> np.ndarray:
    """True wherever a pixel's class is both present (not nodata) and
    change-eligible (not cloud/shadow) -- see module docstring."""
    change_eligible_ids = set(class_config.change_eligible_ids())
    return (class_array != nodata) & np.isin(class_array, list(change_eligible_ids))


@dataclass(frozen=True)
class RawChangeResult:
    from_class: np.ndarray
    to_class: np.ndarray
    changed: np.ndarray
    valid: np.ndarray


def compute_raw_change(
    composites_root: str | Path,
    tile_id: str,
    from_month: str,
    to_month: str,
    dst_transform,
    dst_crs_wkt: str,
    dst_shape: tuple[int, int],
    class_config: ClassConfig,
    nodata: int = 255,
) -> Optional[RawChangeResult]:
    """The raw two-period classification comparison (see module docstring).
    Returns ``None`` if either month has no stored composite at all."""
    from_result = load_composite_on_grid(composites_root, tile_id, from_month, dst_transform, dst_crs_wkt, dst_shape)
    to_result = load_composite_on_grid(composites_root, tile_id, to_month, dst_transform, dst_crs_wkt, dst_shape)
    if from_result is None or to_result is None:
        return None

    from_class, from_nodata = from_result
    to_class, to_nodata = to_result

    valid = _valid_mask(from_class, from_nodata, class_config) & _valid_mask(to_class, to_nodata, class_config)
    changed = valid & (from_class != to_class)

    from_out = np.where(valid, from_class, nodata).astype(np.uint8)
    to_out = np.where(valid, to_class, nodata).astype(np.uint8)

    return RawChangeResult(from_class=from_out, to_class=to_out, changed=changed, valid=valid)


def compute_persistent_change(
    composites_root: str | Path,
    tile_id: str,
    from_month: str,
    to_month: str,
    next_months: list[str],
    dst_transform,
    dst_crs_wkt: str,
    dst_shape: tuple[int, int],
    class_config: ClassConfig,
    persistence_periods: int = 1,
) -> Optional[np.ndarray]:
    """The raw change, narrowed to pixels where the *to*-period class also
    holds for at least ``persistence_periods`` more stored months
    immediately following ``to_month`` (``next_months``, in order --
    typically the next 1-2 available monthly composites for this tile, not
    necessarily calendar-consecutive since a tile can have gaps). A pixel
    with fewer than ``persistence_periods`` follow-up composites available
    at all cannot be confirmed and is excluded (``False``), not assumed
    persistent by default -- absence of contradicting evidence is not
    evidence of persistence."""
    raw = compute_raw_change(
        composites_root, tile_id, from_month, to_month, dst_transform, dst_crs_wkt, dst_shape, class_config
    )
    if raw is None:
        return None
    if len(next_months) < persistence_periods:
        return np.zeros(dst_shape, dtype=bool)

    persists = np.ones(dst_shape, dtype=bool)
    for month in next_months[:persistence_periods]:
        follow_up = load_composite_on_grid(composites_root, tile_id, month, dst_transform, dst_crs_wkt, dst_shape)
        if follow_up is None:
            return np.zeros(dst_shape, dtype=bool)
        follow_class, follow_nodata = follow_up
        follow_valid = _valid_mask(follow_class, follow_nodata, class_config)
        persists &= follow_valid & (follow_class == raw.to_class)

    return raw.changed & persists


def _month_start_ordinal(month: str) -> int:
    from datetime import date

    year, mon = (int(v) for v in month.split("-"))
    return date(year, mon, 1).toordinal()


def compute_corroborated_change(
    raw_changed: np.ndarray,
    from_month: str,
    to_month: str,
    ccdc_result: Optional[dict],
    bfast_result: Optional[dict],
    corroboration_window_days: int = DEFAULT_CORROBORATION_WINDOW_DAYS,
) -> np.ndarray:
    """Raw change narrowed to pixels where CCDC and/or BFAST also detected a
    break within ``corroboration_window_days`` of the ``[from_month,
    to_month]`` boundary -- the strongest-confidence band (see module
    docstring). ``ccdc_result``/``bfast_result`` are the dicts
    ``change.ccdc.read_ccdc_result``/``change.bfast.read_bfast_result``
    return; either may be ``None`` if that detector's output is not
    available for this tile, in which case it simply contributes nothing
    (this band still reflects whichever detector(s) *are* available, rather
    than requiring both)."""
    boundary_ordinal = _month_start_ordinal(to_month)
    window_start = boundary_ordinal - corroboration_window_days
    window_end = boundary_ordinal + corroboration_window_days

    corroborated = np.zeros_like(raw_changed, dtype=bool)

    if ccdc_result is not None:
        in_window = (ccdc_result["t_break"] > 0) & (ccdc_result["t_break"] >= window_start) & (ccdc_result["t_break"] <= window_end)
        for row, col in zip(ccdc_result["row"][in_window], ccdc_result["col"][in_window]):
            if raw_changed[row, col]:
                corroborated[row, col] = True

    if bfast_result is not None:
        in_window = (
            bfast_result["has_break"]
            & (bfast_result["break_date"] >= window_start)
            & (bfast_result["break_date"] <= window_end)
        )
        for row, col in zip(bfast_result["row"][in_window], bfast_result["col"][in_window]):
            if raw_changed[row, col]:
                corroborated[row, col] = True

    return corroborated


def change_map_output_path(output_root: str | Path, tile_id: str, comparison_key: str) -> Path:
    """``<output_root>/<tile_id>/<comparison_key>/change_map.npz`` --
    ``comparison_key`` is e.g. ``"2023-08_vs_2024-08"`` (annual) or
    ``"2023-08_vs_2023-09"`` (month-to-month)."""
    return Path(output_root) / tile_id / comparison_key / "change_map.npz"


def write_change_map(path: str | Path, result: dict) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        from_class=result["from_class"],
        to_class=result["to_class"],
        changed_raw=result["changed_raw"],
        changed_persistent=result["changed_persistent"],
        changed_corroborated=result["changed_corroborated"],
        dnbr=result["dnbr"],
        transform=np.array(list(result["transform"])[:6], dtype=np.float64),
        crs_wkt=np.array(result["crs_wkt"]),
        resolution_m=np.array(result["resolution_m"], dtype=np.float64),
        from_month=np.array(result["from_month"]),
        to_month=np.array(result["to_month"]),
    )
    return out


def read_change_map(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as data:
        return {
            "from_class": np.array(data["from_class"]),
            "to_class": np.array(data["to_class"]),
            "changed_raw": np.array(data["changed_raw"]),
            "changed_persistent": np.array(data["changed_persistent"]),
            "changed_corroborated": np.array(data["changed_corroborated"]),
            "dnbr": np.array(data["dnbr"]),
            "transform": tuple(float(v) for v in data["transform"]),
            "crs_wkt": str(data["crs_wkt"]),
            "resolution_m": float(data["resolution_m"]),
            "from_month": str(data["from_month"]),
            "to_month": str(data["to_month"]),
        }


def build_change_map(
    composites_root: str | Path,
    tile_id: str,
    from_month: str,
    to_month: str,
    next_months: list[str],
    dst_transform,
    dst_crs_wkt: str,
    dst_shape: tuple[int, int],
    class_config: ClassConfig,
    resolution_m: float,
    ccdc_result: Optional[dict] = None,
    bfast_result: Optional[dict] = None,
    dnbr: Optional[np.ndarray] = None,
    persistence_periods: int = 1,
    corroboration_window_days: int = DEFAULT_CORROBORATION_WINDOW_DAYS,
) -> Optional[dict]:
    """Build one ``[from_month, to_month]`` comparison's full change map
    (all three classification-change confidence bands, plus dNBR passed
    through if available). Returns ``None`` if the raw comparison itself
    cannot be built (either month missing a stored composite)."""
    raw = compute_raw_change(
        composites_root, tile_id, from_month, to_month, dst_transform, dst_crs_wkt, dst_shape, class_config
    )
    if raw is None:
        return None

    persistent = compute_persistent_change(
        composites_root, tile_id, from_month, to_month, next_months, dst_transform, dst_crs_wkt, dst_shape,
        class_config, persistence_periods=persistence_periods,
    )
    corroborated = compute_corroborated_change(
        raw.changed, from_month, to_month, ccdc_result, bfast_result, corroboration_window_days
    )

    dnbr_band = dnbr if dnbr is not None else np.full(dst_shape, np.nan, dtype=np.float32)

    return {
        "from_class": raw.from_class,
        "to_class": raw.to_class,
        "changed_raw": raw.changed,
        "changed_persistent": persistent if persistent is not None else np.zeros(dst_shape, dtype=bool),
        "changed_corroborated": corroborated,
        "dnbr": dnbr_band,
        "transform": tuple(dst_transform)[:6],
        "crs_wkt": dst_crs_wkt,
        "resolution_m": resolution_m,
        "from_month": from_month,
        "to_month": to_month,
    }


def annual_month_pairs(all_months: list[str]) -> list[tuple[str, str]]:
    """Every ``(from_month, to_month)`` pair comparing the same calendar
    month across two consecutive years present in ``all_months`` -- e.g.
    ``("2023-08", "2024-08")`` if both are stored, regardless of whether
    every year in between has that month too."""
    by_month: dict[str, list[int]] = {}
    for period in all_months:
        year, mon = (int(v) for v in period.split("-"))
        by_month.setdefault(f"{mon:02d}", []).append(year)

    pairs: list[tuple[str, str]] = []
    for mon, years in by_month.items():
        years_sorted = sorted(years)
        for y1, y2 in zip(years_sorted, years_sorted[1:]):
            if y2 == y1 + 1:
                pairs.append((f"{y1:04d}-{mon}", f"{y2:04d}-{mon}"))
    return pairs


def month_to_month_pairs(all_months: list[str]) -> list[tuple[str, str]]:
    """Every ``(from_month, to_month)`` pair of consecutive stored months,
    regardless of season -- e.g. ``("2023-08", "2023-09")`` if both are
    stored; a gap (no composite for the month in between) simply means no
    pair spans it, rather than pairing across the gap."""
    from datetime import date

    sorted_months = sorted(all_months)
    pairs: list[tuple[str, str]] = []
    for m1, m2 in zip(sorted_months, sorted_months[1:]):
        y1, mo1 = (int(v) for v in m1.split("-"))
        expected_next = date(y1, mo1, 1)
        expected_next = date(y1 + 1, 1, 1) if mo1 == 12 else date(y1, mo1 + 1, 1)
        y2, mo2 = (int(v) for v in m2.split("-"))
        if date(y2, mo2, 1) == expected_next:
            pairs.append((m1, m2))
    return pairs
