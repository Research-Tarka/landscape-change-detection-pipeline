"""Name the change events from the detected surface states.

Purpose
-------
The classifier only knows *surface states* (forest, grassland, bare ground,
burned, snow, ...). A cutblock, a field or a built surface is a *use*, read
from how the states behave in time -- from the monthly class series of each
pixel (Stage 07 composites), with Stage 09's breaks only where an abrupt
event is the thing to date. Cloud, shadow, snow and ice months are
unobserved: skipped, never evidence either way. Both rules below only look at
the *growing season* (months outside it are not used at all), because that is
when the surface can be told apart and it is what a farmer or a forest does
year after year.

Cropland: bare ground and grassland rolling through the season, every year since it began
------------------------------------------------------------------------------------------
Within the crop season (``crop_season``, default May-October) a field is
worked and then covered, so both bare ground and grassland show in the same
season. A year "cycles" when, among its observed months of the season, each
of the two classes shows at least ``crop_min_months_per_year`` times. A pixel
cycling every observed year from some first year *to the end of the series* --
cultivation, once started, comes back every year -- for at least
``crop_min_run_years`` observed years (``crop_allowed_gaps`` observed years
that do not cycle are tolerated, default none; a year with no observation in
the season is neither for nor against) and that is never forest again from the
start year on (``crop_max_forest_fraction``, default 0) is
``cropland``; the table keeps the start year and the number of years since.
No break is involved.

Cutblock: a break, cleared for several seasons, forest that takes years to return
---------------------------------------------------------------------------------
A cutblock is an abrupt event, so it starts at a Stage 09 break (every break of
every pixel). Each growing season (``cutblock_season``, default May-October)
gets one forest fraction from its observed months. For a break:

- before: the last ``pre_seasons`` observed seasons before the break's year
  were forest (at least ``min_fraction`` forest);
- after: the seasons that follow are *not forest* (at most ``end_fraction``
  forest) for at least ``min_cleared_years`` observed seasons in a row, and
  not burned (burned share of the first ``min_cleared_years`` of them below ``burn_min_fraction``) --
  a clearing shorter than that is not a cutblock;
- then the forest *recovers* at the first later season with at least
  ``min_fraction`` forest. The event keeps the break date, the recovery year
  (or none: still cleared at the end of the series) and the duration in
  months. Because the cleared run has to last ``min_cleared_years``, forest
  never comes back sooner than that.

A burned share at or above ``burn_min_fraction`` in the cleared seasons says
``fire`` instead. A clearing that never recovers for ``permanent_years`` and
is mostly bare is a ``permanent_clearing`` (road/pad/mine/built *candidate* --
built-up is only ever a candidate here). Other falls are ``canopy_decline``
(forest that stays forest), ``other_loss`` and ``gain``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit, prange

from landscape_change_detection_pipeline.change.landcover_persistence import MonthlyClassStack, _abs_month

EVENT_NAMES: tuple[str, ...] = (
    "none", "fire", "cutblock", "permanent_clearing", "canopy_decline", "other_loss", "gain",
)
EVENT_CODE = {name: i for i, name in enumerate(EVENT_NAMES)}


@dataclass(frozen=True)
class EventTypingParams:
    min_drop: float = 0.15
    min_fraction: float = 0.8
    end_fraction: float = 0.2
    burn_min_fraction: float = 0.4
    cutblock_season: tuple = (5, 10)
    min_season_obs: int = 2
    pre_seasons: int = 2
    min_cleared_years: int = 3
    permanent_years: float = 8.0
    permanent_bare_fraction: float = 0.6
    crop_season: tuple = (5, 10)
    crop_min_months_per_year: int = 2
    crop_min_run_years: int = 4
    crop_max_forest_fraction: float = 0.0
    crop_allowed_gaps: int = 0


# ---------------------------------------------------------------------------
# kernels (pixel-major class series (n_pix, n_months), unobserved = masked)
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _cycle_kernel(series, year_idx, moy, s0, s1, n_years, masked, grass, bare, forest, min_months, min_years, max_gaps,
                  out_cycles, out_start, out_years, out_forest):
    """Per pixel: which years cycle (bare AND grass each seen ``min_months``
    times in the season), then the earliest year from which cultivation runs
    *every observed year to the end of the series* (at most ``max_gaps``
    observed years that do not cycle) for at least ``min_years`` observed
    years. ``out_start`` = that year index, -1 if none."""
    n_pix, n_m = series.shape
    for p in prange(n_pix):
        g = np.zeros(n_years, np.int64)
        b = np.zeros(n_years, np.int64)
        o = np.zeros(n_years, np.int64)
        nfy = np.zeros(n_years, np.int64)
        for m in range(n_m):
            if moy[m] < s0 or moy[m] > s1:
                continue
            v = series[p, m]
            if masked[v]:
                continue
            y = year_idx[m]
            o[y] += 1
            if v == grass:
                g[y] += 1
            elif v == bare:
                b[y] += 1
            elif v == forest:
                nfy[y] += 1
        cycles = 0
        for y in range(n_years):
            if g[y] >= min_months and b[y] >= min_months:
                cycles += 1
        out_cycles[p] = cycles
        out_forest[p] = 1.0
        out_start[p] = -1
        out_years[p] = 0
        for s in range(n_years):
            if not (g[s] >= min_months and b[s] >= min_months):
                continue
            missed = 0
            seen = 0
            for y in range(s, n_years):
                if o[y] == 0:
                    continue
                seen += 1
                if not (g[y] >= min_months and b[y] >= min_months):
                    missed += 1
            if missed <= max_gaps and seen >= min_years:
                out_start[p] = s
                out_years[p] = seen
                nf_after = 0
                no_after = 0
                for y in range(s, n_years):
                    nf_after += nfy[y]
                    no_after += o[y]
                out_forest[p] = nf_after / max(no_after, 1)  # forest share from the start year on
                break


@njit(parallel=True, cache=True)
def _cutblock_kernel(series, year_idx, moy, s0, s1, n_years, pix, break_year, break_moy, masked, forest, burned,
                     bare, min_obs, pre_seasons, burn_seasons, min_forest, max_forest, out_pre_forest, out_cleared_years,
                     out_burned, out_bare, out_recovered_year, out_last_year):
    """Per break, from one forest/burned/bare fraction per growing season:
    forest before? how many consecutive seasons cleared after? burned/bare
    share of those? which season is forest again (-1 = none)?"""
    n_m = series.shape[1]
    for k in prange(len(pix)):
        p = pix[k]
        nobs = np.zeros(n_years, np.int64)
        nfor = np.zeros(n_years, np.int64)
        nbur = np.zeros(n_years, np.int64)
        nbar = np.zeros(n_years, np.int64)
        for m in range(n_m):
            if moy[m] < s0 or moy[m] > s1:
                continue
            v = series[p, m]
            if masked[v]:
                continue
            y = year_idx[m]
            nobs[y] += 1
            if v == forest:
                nfor[y] += 1
            elif v == burned:
                nbur[y] += 1
            elif v == bare:
                nbar[y] += 1
        yb = break_year[k]
        # before: the last `pre_seasons` observed seasons strictly before the break's year
        seen = 0
        fsum = 0.0
        y = yb - 1
        while y >= 0 and seen < pre_seasons:
            if nobs[y] >= min_obs:
                seen += 1
                fsum += nfor[y] / nobs[y]
            y -= 1
        out_pre_forest[k] = seen >= pre_seasons and fsum / max(seen, 1) >= min_forest
        # after: consecutive cleared seasons from the first season fully after the break
        y0 = yb if break_moy[k] < s0 else yb + 1
        cleared = 0
        bsum = 0.0
        rsum = 0.0
        out_recovered_year[k] = -1
        last_obs = -1
        y = y0
        while y < n_years:
            if nobs[y] < min_obs:
                y += 1
                continue
            last_obs = y
            f = nfor[y] / nobs[y]
            if f <= max_forest:
                if cleared < burn_seasons:
                    bsum += nbur[y] / nobs[y]  # burned share of the first cleared seasons: a fire, not a later cover
                cleared += 1
                rsum += nbar[y] / nobs[y]
                y += 1
            else:
                break
        out_cleared_years[k] = cleared
        out_burned[k] = bsum / max(min(cleared, burn_seasons), 1)
        out_bare[k] = rsum / max(cleared, 1)
        # recovery: first later season that is forest again
        while y < n_years:
            if nobs[y] >= min_obs:
                last_obs = y
                if nfor[y] / nobs[y] >= min_forest:
                    out_recovered_year[k] = y
                    break
            y += 1
        out_last_year[k] = last_obs


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def _pixel_major(stack: MonthlyClassStack) -> np.ndarray:
    n_months = stack.classes.shape[0]
    return np.ascontiguousarray(stack.classes.reshape(n_months, -1).T)


def _masked_table(stack: MonthlyClassStack, masked_ids) -> np.ndarray:
    masked = np.zeros(256, dtype=np.bool_)
    masked[stack.nodata] = True
    for c in masked_ids:
        masked[c] = True
    return masked


def _calendar(stack: MonthlyClassStack):
    """Year index (0-based from the first year of the stack), month of year and
    the first year for each stack month."""
    abs_months = np.array([_abs_month(m) for m in stack.months], dtype=np.int64)
    years = abs_months // 12
    year0 = int(years.min()) + 1970  # calendar year of the first stack month
    return abs_months, (years + 1970 - year0).astype(np.int64), (abs_months % 12 + 1).astype(np.int64), year0


def detect_cropland(stack: MonthlyClassStack, class_ids: dict, masked_ids, params: EventTypingParams) -> dict:
    """Pixels where bare ground and grassland roll through the crop season
    every year from the year cultivation is first seen to the end of the series
    (see module docstring)."""
    series = _pixel_major(stack)
    _, year_idx, moy, year0 = _calendar(stack)
    n_years = int(year_idx.max()) + 1
    n_pix = series.shape[0]
    cycles = np.zeros(n_pix, dtype=np.int64)
    start = np.full(n_pix, -1, dtype=np.int64)
    years = np.zeros(n_pix, dtype=np.int64)
    forest_frac = np.zeros(n_pix, dtype=np.float64)
    _cycle_kernel(
        series, year_idx, moy, int(params.crop_season[0]), int(params.crop_season[1]), n_years,
        _masked_table(stack, masked_ids), int(class_ids["grass"]), int(class_ids["bare"]), int(class_ids["forest"]),
        int(params.crop_min_months_per_year), int(params.crop_min_run_years), int(params.crop_allowed_gaps),
        cycles, start, years, forest_frac,
    )
    is_crop = (start >= 0) & (forest_frac <= params.crop_max_forest_fraction)
    pix = np.flatnonzero(is_crop)
    width = stack.shape[1]
    return {
        "row": (pix // width).astype(np.int32), "col": (pix % width).astype(np.int32),
        "start_year": (start[pix] + year0).astype(np.int16),  # first year of the uninterrupted cycling
        "years": years[pix].astype(np.int16),  # observed years cultivated since then
        "cycle_years": cycles[pix].astype(np.int16),
    }


def type_change_events(segments: dict, stack: MonthlyClassStack, class_ids: dict, masked_ids, params: EventTypingParams) -> dict:
    """Named event for every Stage 09 break, with the cutblock's duration and
    forest recovery read from the growing-season class series (see module
    docstring)."""
    from datetime import date

    height, width = segments["shape"]
    t_break = np.asarray(segments["t_break"])
    sel = np.flatnonzero(t_break > 0)
    row = np.asarray(segments["row"])[sel].astype(np.int64)
    col = np.asarray(segments["col"])[sel].astype(np.int64)
    mag = np.asarray(segments.get("resid_magnitude", segments["magnitude"]))[sel]
    names = list(segments["feature_names"])
    cols = [names.index(n) for n in ("NBR", "NDVI") if n in names]
    drop = np.max(-mag[:, cols], axis=1)
    rise = np.max(mag[:, cols], axis=1)

    abs_months, year_idx, moy, year0 = _calendar(stack)
    n_years = int(year_idx.max()) + 1
    dates = [date.fromordinal(int(o)) for o in t_break[sel]]
    break_year = np.array([d.year - year0 for d in dates], dtype=np.int64)
    break_moy = np.array([d.month for d in dates], dtype=np.int64)
    # a break before the first stacked year cannot be checked: clamp so it reads as "no forest before"
    break_year = np.clip(break_year, 0, n_years)

    series = _pixel_major(stack)
    pix = row * width + col
    n = len(sel)
    pre_forest = np.zeros(n, dtype=np.bool_)
    cleared_years = np.zeros(n, dtype=np.int64)
    burned_frac = np.zeros(n, dtype=np.float64)
    bare_frac = np.zeros(n, dtype=np.float64)
    recovered_year = np.full(n, -1, dtype=np.int64)
    last_year = np.full(n, -1, dtype=np.int64)
    _cutblock_kernel(
        series, year_idx, moy, int(params.cutblock_season[0]), int(params.cutblock_season[1]), n_years,
        pix.astype(np.int64), break_year, break_moy, _masked_table(stack, masked_ids),
        int(class_ids["forest"]), int(class_ids["burned"]), int(class_ids["bare"]),
        int(params.min_season_obs), int(params.pre_seasons), int(params.min_cleared_years), float(params.min_fraction), float(params.end_fraction),
        pre_forest, cleared_years, burned_frac, bare_frac, recovered_year, last_year,
    )

    is_recovered = recovered_year >= 0
    end_year = np.where(is_recovered, recovered_year, np.maximum(last_year, break_year))
    duration_months = ((end_year - break_year) * 12).astype(np.int64)

    cleared = cleared_years >= params.min_cleared_years
    big_drop = drop >= params.min_drop
    big_rise = rise >= params.min_drop
    burned = burned_frac >= params.burn_min_fraction
    event = np.zeros(n, dtype=np.int8)
    event[big_rise] = EVENT_CODE["gain"]
    event[big_drop] = EVENT_CODE["other_loss"]
    event[big_drop & pre_forest & (cleared_years == 0)] = EVENT_CODE["canopy_decline"]
    cut = big_drop & pre_forest & cleared & ~burned
    event[cut] = EVENT_CODE["cutblock"]
    permanent = cut & ~is_recovered & (duration_months >= params.permanent_years * 12) & (bare_frac >= params.permanent_bare_fraction)
    event[permanent] = EVENT_CODE["permanent_clearing"]
    event[big_drop & burned & (cleared_years >= 1)] = EVENT_CODE["fire"]

    recovery_month = np.array([f"{year0 + int(y):04d}" if y >= 0 else "" for y in recovered_year])
    return {
        "row": row.astype(np.int32), "col": col.astype(np.int32),
        "break_date": t_break[sel].astype(np.int32),
        "event_type": event,
        "drop": drop.astype(np.float32), "rise": rise.astype(np.float32),
        "pre_forest": pre_forest, "cleared_seasons": cleared_years.astype(np.int16),
        "burned_fraction": burned_frac.astype(np.float32), "bare_fraction": bare_frac.astype(np.float32),
        "recovered": is_recovered,
        "recovery_year": recovery_month,  # season year the forest is back ("" = not within the series)
        "duration_months": duration_months.astype(np.int32),  # break -> forest back, or -> last observed season
        "shape": np.array([height, width], dtype=np.int32),
    }


def event_grids(events: dict, cropland: dict) -> dict[str, np.ndarray]:
    """Per-pixel rasters: the code and year of each pixel's *latest* named
    event (0 = none), the cutblock duration in months of the latest cutblock,
    and the cropland flag."""
    from datetime import date

    height, width = (int(v) for v in events["shape"])
    code = np.zeros((height, width), dtype=np.int8)
    year = np.zeros((height, width), dtype=np.int16)
    named = events["event_type"] > 0
    order = np.argsort(events["break_date"][named], kind="stable")  # later dates overwrite earlier ones
    r, c = events["row"][named][order], events["col"][named][order]
    code[r, c] = events["event_type"][named][order]
    year[r, c] = np.array([date.fromordinal(int(d)).year for d in events["break_date"][named][order]], dtype=np.int16)
    cut_dur = np.full((height, width), -1, dtype=np.int32)
    is_cut = np.isin(events["event_type"], [EVENT_CODE["cutblock"], EVENT_CODE["permanent_clearing"]])
    o2 = np.argsort(events["break_date"][is_cut], kind="stable")
    cut_dur[events["row"][is_cut][o2], events["col"][is_cut][o2]] = events["duration_months"][is_cut][o2]
    crop = np.zeros((height, width), dtype=bool)
    crop[cropland["row"], cropland["col"]] = True
    return {"latest_event_type": code, "latest_event_year": year, "cutblock_duration_months": cut_dur, "cropland": crop}


def change_events_output_path(output_root: str | Path, tile_id: str) -> Path:
    return Path(output_root) / tile_id / "change_events.npz"


def write_change_events(path: str | Path, events: dict, cropland: dict, segments: dict) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        **events,
        **{f"cropland_{k}": v for k, v in cropland.items()},
        **event_grids(events, cropland),
        event_names=np.array(EVENT_NAMES),
        transform=np.array(segments["transform"][:6], dtype=np.float64),
        crs_wkt=np.array(segments["crs_wkt"]),
        resolution_m=np.array(segments["resolution_m"], dtype=np.float64),
    )
    return out


def read_change_events(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as data:
        result = {k: np.array(data[k]) for k in data.files}
    result["event_names"] = [str(n) for n in result["event_names"]]
    result["crs_wkt"] = str(result["crs_wkt"])
    result["transform"] = tuple(float(v) for v in result["transform"])
    return result
