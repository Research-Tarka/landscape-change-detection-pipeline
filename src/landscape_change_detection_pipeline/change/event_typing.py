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

Cropland: open land that has stayed open, in a patch wide enough to be a field
------------------------------------------------------------------------------
The classifier never says "crop": a field comes out as grassland, with bare
ground when it is worked. So a field is read from what it does over the years.
Within the crop season (``crop_season``, default June-September: in May and October leaf-off woodland and any grass look open) a pixel counts as
*open* in a year when at least ``crop_open_fraction`` of its observed months
are grassland / bare / crop, and as *closed* (forest, wetland, water, burned)
when less than half are. A pixel is cropland when, from the year after its last
closed year to the end of the series, it stays open for at least
``crop_min_run_years`` observed years, is closed for at most
``crop_max_forest_fraction`` of those months and shows bare ground (it is
worked) in at least ``crop_min_bare_fraction`` of them. Then the space has to
look like a field and not a road, pad or seismic line: the pixels are opened
by a ``crop_min_width_px`` square (thinner strips disappear) and only patches
of at least ``crop_min_patch_px`` pixels remain. Events that fall inside a
field (gain, other loss, loss on open land) are renamed ``cropland_change``
(:func:`apply_cropland_to_events`).

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
    "recent_clearing", "linear_feature", "open_land_disturbance", "regrowth_change", "cropland_change",
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
    crop_season: tuple = (6, 9)  # summer only: in May / October leaf-off woodland and any grass read as open or bare
    crop_min_run_years: int = 8
    crop_max_forest_fraction: float = 0.2
    crop_open_fraction: float = 0.8
    crop_min_bare_fraction: float = 0.03
    crop_min_bare_year_fraction: float = 0.15  # applies to land cleared from forest during the series (worked, not a regrowing cut)
    crop_since_start_years: int = 1  # open from the first year(s) of the series = old farmland / hay: no bare ground needed
    crop_end_gap_years: int = 2  # the open run may stop this many years before the end of the series (classifier drift), not more
    crop_min_patch_px: int = 300
    crop_min_width_px: int = 5
    cleared_gap_seasons: int = 1  # noisy seasons (a little forest) tolerated inside a clearing
    pre_min_fraction: float = 0.5  # forest share of a "before" season (mixed / young stands are thinner than a recovered one)
    fire_season: tuple = (6, 8)  # months a fire can be dated in; dry soil and autumn colour read as burned outside it
    object_fire_min_pixels: int = 2000  # a same-year patch this big can be retyped as one fire
    object_fire_burned: float = 0.10  # ... when its mean burned share is at least this
    object_fire_dnbr: float = 0.45  # ... or its mean NBR fall is at least this (with object_fire_dnbr_burned burned)
    object_fire_dnbr_burned: float = 0.05
    object_fire_local_burned: float = 0.03  # inside a fire patch, a pixel needs this burned share within ~5 px of it
    linear_max_width_px: int = 3  # strips narrower than this (pixels) are road / seismic line / pipeline candidates
    linear_min_extent_px: int = 15  # ... if they run at least this long
    linear_gap_px: int = 3  # holes up to this wide in a track still join its dashes


# ---------------------------------------------------------------------------
# kernels (pixel-major class series (n_pix, n_months), unobserved = masked)
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _open_run_kernel(series, year_idx, moy, s0, s1, n_years, masked, grass, bare, crop, min_obs, open_frac, min_years,
                     out_start, out_years, out_closed, out_bare, out_bare_years, out_end):
    """Per pixel: its last open run of ``min_years`` or more -- observed years from a year at least
    ``open_frac`` open until a strongly closed year (under half open) -- and, in
    that run, the closed and bare shares of the observed months and the number of
    years with bare ground. ``out_start`` = -1 if the longest run is under ``min_years``."""
    n_pix, n_m = series.shape
    for p in prange(n_pix):
        o = np.zeros(n_years, np.int64)
        op = np.zeros(n_years, np.int64)
        b = np.zeros(n_years, np.int64)
        for m in range(n_m):
            if moy[m] < s0 or moy[m] > s1:
                continue
            v = series[p, m]
            if masked[v]:
                continue
            y = year_idx[m]
            o[y] += 1
            if v == grass or v == bare or v == crop:
                op[y] += 1
            if v == bare:
                b[y] += 1
        out_start[p] = -1
        out_years[p] = 0
        out_closed[p] = 1.0
        out_bare[p] = 0.0
        out_bare_years[p] = 0
        out_end[p] = -1
        rs = -1
        last = -1
        seen = 0
        tot = 0
        opn = 0
        bar = 0
        by = 0
        for y in range(n_years + 1):
            ended = y == n_years
            if not ended:
                if o[y] < min_obs:
                    continue
                f = op[y] / o[y]
                if f >= 0.5 and (rs >= 0 or f >= open_frac):
                    if rs < 0:
                        rs = y
                    seen += 1
                    last = y
                    tot += o[y]
                    opn += op[y]
                    bar += b[y]
                    if b[y] > 0:
                        by += 1
                    continue
                if f < 0.5 and rs >= 0:
                    # one strongly closed year between two open ones is classifier noise: step over it
                    y2 = y + 1
                    while y2 < n_years and o[y2] < min_obs:
                        y2 += 1
                    if y2 < n_years and op[y2] / o[y2] >= open_frac:
                        continue
                ended = f < 0.5
            if ended:
                if rs >= 0 and seen >= min_years:  # the last qualifying run: the one that reaches the end, if any
                    out_start[p] = rs
                    out_years[p] = seen
                    out_closed[p] = 1.0 - opn / max(tot, 1)
                    out_bare[p] = bar / max(tot, 1)
                    out_bare_years[p] = by
                    out_end[p] = last
                rs = -1
                seen = 0
                tot = 0
                opn = 0
                bar = 0
                by = 0


@njit(parallel=True, cache=True)
def _cutblock_kernel(series, year_idx, moy, s0, s1, n_years, pix, break_year, break_moy, masked, forest, burned,
                     bare, min_obs, pre_seasons, burn_seasons, min_forest, pre_min_forest, max_forest, gap_tol, out_pre_forest, out_cleared_years,
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
        out_pre_forest[k] = seen >= pre_seasons and fsum / max(seen, 1) >= pre_min_forest
        # after: consecutive cleared seasons from the first season fully after the break
        y0 = yb if break_moy[k] < s0 else yb + 1
        cleared = 0
        gaps = 0
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
            noisy = False
            if f > max_forest and f < min_forest and gaps < gap_tol and cleared >= 1:
                # one season with a few forest months inside a clearing is classifier noise if the next one is cleared again
                y2 = y + 1
                while y2 < n_years and nobs[y2] < min_obs:
                    y2 += 1
                noisy = y2 < n_years and nfor[y2] / nobs[y2] <= max_forest
            if f <= max_forest or noisy:
                if noisy:
                    gaps += 1
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
    """Open, worked land that stays open to the end of the series, in patches
    wide enough to be fields (see module docstring)."""
    from scipy import ndimage as ndi

    series = _pixel_major(stack)
    _, year_idx, moy, year0 = _calendar(stack)
    n_years = int(year_idx.max()) + 1
    n_pix = series.shape[0]
    start = np.full(n_pix, -1, dtype=np.int64)
    years = np.zeros(n_pix, dtype=np.int64)
    closed = np.ones(n_pix, dtype=np.float64)
    bare = np.zeros(n_pix, dtype=np.float64)
    bare_years = np.zeros(n_pix, dtype=np.int64)
    run_end = np.full(n_pix, -1, dtype=np.int64)
    _open_run_kernel(
        series, year_idx, moy, int(params.crop_season[0]), int(params.crop_season[1]), n_years,
        _masked_table(stack, masked_ids), int(class_ids["grass"]), int(class_ids["bare"]), int(class_ids.get("crop", 255)),
        int(params.min_season_obs), float(params.crop_open_fraction), int(params.crop_min_run_years),
        start, years, closed, bare, bare_years, run_end,
    )
    height, width = stack.shape
    worked = (bare >= params.crop_min_bare_fraction) & (bare_years >= params.crop_min_bare_year_fraction * years)
    since_start = start <= params.crop_since_start_years  # open from the first year(s) of the series: farmland / hay, no bare ground needed
    cand = ((start >= 0) & (closed <= params.crop_max_forest_fraction) & (worked | since_start)
            & (run_end >= n_years - 1 - params.crop_end_gap_years)).reshape(height, width)
    k = max(int(params.crop_min_width_px), 1)
    if k > 1:
        cand = ndi.binary_opening(cand, structure=np.ones((k, k), dtype=bool))
    lab, n = ndi.label(cand, structure=np.ones((3, 3), dtype=bool))
    if n:
        keep = np.r_[False, np.bincount(lab.ravel(), minlength=n + 1)[1:] >= params.crop_min_patch_px]
        cand = keep[lab]
    pix = np.flatnonzero(cand.ravel())
    return {
        "row": (pix // width).astype(np.int32), "col": (pix % width).astype(np.int32),
        "start_year": (start[pix] + year0).astype(np.int16),  # first year of its longest open run
        "years": years[pix].astype(np.int16),  # observed years in that run
        "cycle_years": bare_years[pix].astype(np.int16),  # of those, years with bare ground in the season (worked)
    }


def apply_cropland_to_events(events: dict, cropland: dict, shape) -> dict:
    """Events inside a field are its own activity (tillage, harvest, fallow,
    dark ploughed soil read as burned): ``cropland_change``. Losses that cleared
    *forest* (pre_forest) stay what they are -- that is how the field began."""
    height, width = (int(v) for v in shape)
    in_field = np.zeros((height, width), dtype=bool)
    in_field[cropland["row"], cropland["col"]] = True
    inside = in_field[events["row"], events["col"]]
    always = [EVENT_CODE[n] for n in ("gain", "other_loss", "open_land_disturbance", "regrowth_change", "linear_feature")]
    on_open_ground = [EVENT_CODE[n] for n in ("fire", "cutblock", "permanent_clearing", "canopy_decline", "recent_clearing")]
    was_forest = np.asarray(events.get("pre_forest", np.zeros(len(events["row"]), dtype=bool)), dtype=bool)
    out = dict(events)
    etype = np.array(events["event_type"], dtype=np.int8)
    etype[inside & (np.isin(etype, always) | (np.isin(etype, on_open_ground) & ~was_forest))] = EVENT_CODE["cropland_change"]
    out["event_type"] = etype
    return out


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
    dnbr = -mag[:, names.index("NBR")] if "NBR" in names else drop

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
        int(params.min_season_obs), int(params.pre_seasons), int(params.min_cleared_years), float(params.min_fraction), float(params.pre_min_fraction),
        float(params.end_fraction), int(params.cleared_gap_seasons),
        pre_forest, cleared_years, burned_frac, bare_frac, recovered_year, last_year,
    )

    is_recovered = recovered_year >= 0
    end_year = np.where(is_recovered, recovered_year, np.maximum(last_year, break_year))
    duration_months = ((end_year - break_year) * 12).astype(np.int64)

    cleared = cleared_years >= params.min_cleared_years
    big_drop = drop >= params.min_drop
    big_rise = rise >= params.min_drop
    in_fire_season = (break_moy >= params.fire_season[0]) & (break_moy <= params.fire_season[1])
    burned = (burned_frac >= params.burn_min_fraction) & in_fire_season
    event = np.zeros(n, dtype=np.int8)
    event[big_rise] = EVENT_CODE["gain"]
    event[big_drop] = EVENT_CODE["other_loss"]
    event[big_drop & ~pre_forest] = EVENT_CODE["open_land_disturbance"]
    event[big_drop & pre_forest & (cleared_years == 0)] = EVENT_CODE["canopy_decline"]
    # cleared seasons run into the end of the series: too recent to tell cutblock from anything else
    recent = (big_drop & pre_forest & ~is_recovered & (cleared_years >= 1) & (cleared_years < params.min_cleared_years)
              & (n_years - break_year <= params.min_cleared_years + 1))
    event[recent] = EVENT_CODE["recent_clearing"]
    cut = big_drop & pre_forest & cleared & ~burned
    event[cut] = EVENT_CODE["cutblock"]
    permanent = cut & ~is_recovered & (duration_months >= params.permanent_years * 12) & (bare_frac >= params.permanent_bare_fraction)
    event[permanent] = EVENT_CODE["permanent_clearing"]
    event[big_drop & burned & pre_forest & (cleared_years >= 1)] = EVENT_CODE["fire"]  # a fire burns forest; farmland that reads burned is not one
    event = _regrowth_after_disturbance(event, pix, t_break[sel])

    recovery_month = np.array([f"{year0 + int(y):04d}" if y >= 0 else "" for y in recovered_year])
    return {
        "row": row.astype(np.int32), "col": col.astype(np.int32),
        "break_date": t_break[sel].astype(np.int32),
        "event_type": event,
        "drop": drop.astype(np.float32), "dnbr": dnbr.astype(np.float32), "rise": rise.astype(np.float32),
        "pre_forest": pre_forest, "cleared_seasons": cleared_years.astype(np.int16),
        "burned_fraction": burned_frac.astype(np.float32), "bare_fraction": bare_frac.astype(np.float32),
        "recovered": is_recovered,
        "recovery_year": recovery_month,  # season year the forest is back ("" = not within the series)
        "duration_months": duration_months.astype(np.int32),  # break -> forest back, or -> last observed season
        "shape": np.array([height, width], dtype=np.int32),
    }


_FOREST_LOSS = ("fire", "cutblock", "permanent_clearing", "recent_clearing")


def _regrowth_after_disturbance(event: np.ndarray, pix: np.ndarray, t_break: np.ndarray) -> np.ndarray:
    """An open-land loss on a pixel that already had a fire / cutblock / clearing
    earlier is a change *during regrowth* (a second cut, a new burn of the
    young stand, a road through it) -- not a loss on land that was always open."""
    out = event.copy()
    order = np.lexsort((t_break, pix))
    p = pix[order]
    prior_loss = np.isin(event[order], [EVENT_CODE[n] for n in _FOREST_LOSS]).astype(np.int64)
    before = np.cumsum(prior_loss) - prior_loss  # forest-loss events strictly earlier in the sorted order
    first = np.r_[True, p[1:] != p[:-1]]
    group_start = np.maximum.accumulate(np.where(first, np.arange(len(p)), 0))
    earlier_in_pixel = before - before[group_start]
    regrow = (event[order] == EVENT_CODE["open_land_disturbance"]) & (earlier_in_pixel > 0)
    out[order[regrow]] = EVENT_CODE["regrowth_change"]
    return out


def refine_events_by_object(events: dict, params: EventTypingParams) -> dict:
    """Second pass, on the patches rather than the pixels (per break year,
    8-connected). A fire is one patch whose pixels the per-pixel rules scatter
    over cutblock / canopy_decline / other_loss: a big patch (``object_fire_min_pixels``)
    with a clear burn signal (mean burned share, or a strong mean NBR fall plus a
    little burn) is retyped fire as a whole. Strips narrower than
    ``linear_max_width_px`` and at least ``linear_min_extent_px`` long (found by
    a morphological opening, so a road leaving a block is found too) become
    ``linear_feature`` (road / seismic line / pipeline candidate). Adds ``object_px``."""
    from scipy import ndimage as ndi

    out = dict(events)
    etype = np.array(events["event_type"], dtype=np.int8)
    height, width = (int(v) for v in events["shape"])
    row, col = np.asarray(events["row"]), np.asarray(events["col"])
    day = np.asarray(events["break_date"]).astype(np.int64) - 719163  # proleptic ordinal -> days since 1970
    year = day.astype("datetime64[D]").astype("datetime64[Y]").astype(np.int64)
    month = day.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64) % 12 + 1
    summer = (month >= params.fire_season[0]) & (month <= params.fire_season[1])
    burned = np.asarray(events["burned_fraction"], dtype=np.float64)
    dnbr = np.asarray(events.get("dnbr", events["drop"]), dtype=np.float64)
    object_px = np.zeros(len(etype), dtype=np.int32)
    fire, linear = EVENT_CODE["fire"], EVENT_CODE["linear_feature"]
    line_from = [EVENT_CODE[n] for n in (
        "cutblock", "recent_clearing", "other_loss", "open_land_disturbance", "regrowth_change", "permanent_clearing")]
    struct8 = np.ones((3, 3), dtype=bool)
    k = max(int(params.linear_max_width_px), 1)
    for y in np.unique(year):
        idx = np.flatnonzero((year == y) & (etype != 0) & (etype != EVENT_CODE["gain"]))
        if len(idx) == 0:
            continue
        r, c = row[idx], col[idx]
        present = np.zeros((height, width), dtype=bool)
        present[r, c] = True
        lab, n = ndi.label(present, structure=struct8)
        obj = lab[r, c]
        count = np.bincount(obj, minlength=n + 1)
        mean_burn = np.bincount(obj, weights=burned[idx], minlength=n + 1) / np.maximum(count, 1)
        mean_dnbr = np.bincount(obj, weights=dnbr[idx], minlength=n + 1) / np.maximum(count, 1)
        summer_share = np.bincount(obj, weights=summer[idx].astype(np.float64), minlength=n + 1) / np.maximum(count, 1)
        is_fire = (count >= params.object_fire_min_pixels) & (summer_share >= 0.5) & (
            (mean_burn >= params.object_fire_burned)
            | ((mean_dnbr >= params.object_fire_dnbr) & (mean_burn >= params.object_fire_dnbr_burned))
        )
        object_px[idx] = count[obj]
        in_fire = is_fire[obj] & summer[idx]
        if in_fire.any():
            # the patch qualifies as a fire; a clean cutblock that merely touches the burn does not follow it
            bsum = np.zeros((height, width))
            bsum[r, c] = burned[idx]
            local = ndi.uniform_filter(bsum, size=11, mode="constant") / np.maximum(ndi.uniform_filter(present.astype(np.float64), size=11, mode="constant"), 1e-9)
            in_fire = in_fire & (local[r, c] >= params.object_fire_local_burned)
        etype[idx[in_fire]] = fire
        # strips: an opening removes everything narrower than k pixels; what it removed and runs long is linear
        cand = (etype[idx] != fire) & np.isin(etype[idx], line_from)
        if not cand.any() or k <= 1:
            continue
        mask = np.zeros((height, width), dtype=bool)
        mask[r[cand], c[cand]] = True
        thin = mask & ~ndi.binary_opening(mask, structure=np.ones((k, k), dtype=bool))
        # dashes of a track with holes in it join when the holes are at most linear_gap_px wide
        reach = ndi.binary_dilation(thin, structure=struct8, iterations=max((int(params.linear_gap_px) + 1) // 2, 0)) if params.linear_gap_px > 0 else thin
        tl, tn = ndi.label(reach, structure=struct8)
        if tn == 0:
            continue
        pad = (int(params.linear_gap_px) + 1) // 2 if params.linear_gap_px > 0 else 0
        extent = np.array([max(b[0].stop - b[0].start, b[1].stop - b[1].start) - 2 * pad for b in ndi.find_objects(tl)])
        long_strip = np.r_[False, extent >= params.linear_min_extent_px]
        sel = idx[cand]
        on_thin = thin[row[sel], col[sel]]
        etype[sel[on_thin & long_strip[tl[row[sel], col[sel]]]]] = linear
    out["event_type"] = etype
    out["object_px"] = object_px
    return out


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
