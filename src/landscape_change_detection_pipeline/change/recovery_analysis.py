"""Recovery curves, succession trajectories and recovery factors of change events.

Purpose
-------
:mod:`change.event_typing` names the events (fire, cutblock, clearing, canopy
decline ...) and dates them; :mod:`change.vegetation_dynamics` gives every
pixel's yearly growing-season index. This module joins the two to answer
*how does the land come back, and why there and not elsewhere*. Data, not
maps: one row per event, plus binned summaries, all exportable to CSV.

Unit: the event (a Stage 09 break kept by ``recovery_analysis.event_types``),
not the pixel -- a pixel with three breaks is three events. Time is counted
in **years since the break year** (0 = the year of the break).

Recovery curves (``curves``)
-----------------------------
- *baseline*: median of the yearly index over the ``pre_years`` years before
  the break year (at least ``min_pre_years`` observed), so the event's own
  year never pollutes it. Events without a positive baseline (< 0.05 -- the
  ratio below would be meaningless) get no curve.
- *curve*: yearly index in years 0..``max_years`` after the break year,
  absolute and relative (``value / baseline``).
- *trough*: lowest value in the curve and when. ``dropped`` = trough below
  ``recovery_threshold`` * baseline (an event that never really dropped this
  index is flagged, not counted as an instant recovery).
- *recovery time*: first year, from the trough on, where the value is back
  at ``recovery_threshold`` of the baseline *and stays there* for
  ``sustain_years`` observed years (a one-year spike is not recovery). Right
  censoring is explicit: ``recovered = False`` with ``last_observed_year``
  telling how long the event could be followed -- never a fake "did not
  recover".
- *rate*: relative gain per year from the trough to the recovery (or to the
  last observed year).

Succession (``succession``)
----------------------------
The dominant class of each growing season, year by year after the event
(``min_season_obs`` observed months needed; cloud/shadow/snow/ice months are
not evidence). Kept per event: the sequence, the years spent in each class,
the number of state changes, the year forest is dominant again. Pooled per
event type: a Markov table (year-to-year transition counts and
probabilities) and the share of events in each class by years since the
event -- the typical succession path (bare -> grass -> shrub -> forest ...).

Recovery factors (``factors``)
-------------------------------
The event table joined with terrain (elevation, slope, aspect as
northness/eastness and 8-sector) from the tile DEM (optional) and with the
event's own attributes (type, severity, burned/bare share, pre-event level,
year). Plus a binned summary (per event type, elevation band, slope band,
aspect sector, and event type x each) of the recovery outcomes, with the
share recovered by 5 and 10 years computed among the events actually
observable that long (censoring-aware). That table is the ready-to-plot
answer; the event table is the input for a proper statistical model.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
from numba import njit, prange

from landscape_change_detection_pipeline.change.event_typing import EVENT_NAMES

MIN_BASELINE = 0.05
NO_CLASS = 255


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def curves_kernel(annual, break_idx, max_years, pre_years, min_pre, thr, sustain,
                  baseline, curve_abs, curve_rel, trough_rel, trough_year, recovery_year, last_year,
                  dropped, rate, final_rel):
    """One event per row of ``annual`` ((n_ev, n_years) yearly index).
    ``recovery_year``: -1 not recovered (censored at ``last_year``), -2 no
    baseline. ``curve_*`` are ``(n_ev, max_years + 1)``."""
    n_ev, n_years = annual.shape
    for e in prange(n_ev):
        by = break_idx[e]
        recovery_year[e] = -2
        if by < 0 or by >= n_years:
            continue
        buf = np.empty(pre_years)
        n = 0
        for y in range(max(0, by - pre_years), by):
            v = annual[e, y]
            if v == v:
                buf[n] = v
                n += 1
        if n < min_pre:
            continue
        srt = np.sort(buf[:n])
        base = srt[n // 2] if n % 2 == 1 else 0.5 * (srt[n // 2 - 1] + srt[n // 2])
        baseline[e] = base
        if base < MIN_BASELINE:
            continue
        best = np.inf
        best_k = -1
        last = -1
        for k in range(max_years + 1):
            y = by + k
            if y >= n_years:
                break
            v = annual[e, y]
            if v == v:
                curve_abs[e, k] = v
                curve_rel[e, k] = v / base
                last = k
                if v < best:
                    best = v
                    best_k = k
        last_year[e] = last
        if last < 0:
            continue
        trough_rel[e] = best / base
        trough_year[e] = best_k
        dropped[e] = best / base < thr
        final_rel[e] = curve_rel[e, last]
        rec = -1
        for k in range(best_k, last + 1):
            if curve_rel[e, k] != curve_rel[e, k]:
                continue
            if curve_rel[e, k] < thr:
                continue
            ok = 0
            j = k
            while j <= last and ok < sustain:
                if curve_rel[e, j] == curve_rel[e, j]:
                    if curve_rel[e, j] < thr:
                        break
                    ok += 1
                j += 1
            if ok >= sustain:
                rec = k
                break
        recovery_year[e] = rec
        end_k = rec if rec >= 0 else last
        if end_k > best_k:
            rate[e] = (curve_rel[e, end_k] - curve_rel[e, best_k]) / (end_k - best_k)


@njit(parallel=True, cache=True)
def succession_kernel(cls, year_idx, moy, gs0, gs1, min_obs, masked, break_idx, max_years, out):
    """Dominant class of each growing season, years ``break_idx .. +max_years``
    (``NO_CLASS`` where fewer than ``min_obs`` observed months)."""
    n_ev, n_months = cls.shape
    for e in prange(n_ev):
        by = break_idx[e]
        if by < 0:
            continue
        counts = np.zeros((max_years + 1, 256), dtype=np.int32)
        for t in range(n_months):
            k = year_idx[t] - by
            if k < 0 or k > max_years:
                continue
            if moy[t] < gs0 or moy[t] > gs1:
                continue
            c = cls[e, t]
            if masked[c]:
                continue
            counts[k, c] += 1
        for k in range(max_years + 1):
            best = 0
            best_c = NO_CLASS
            total = 0
            for c in range(256):
                total += counts[k, c]
                if counts[k, c] > best:
                    best = counts[k, c]
                    best_c = c
            if total >= min_obs:
                out[e, k] = best_c


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def dominant_class_series(class_stack, ev_rows, ev_cols) -> np.ndarray:
    """``(n_ev, n_months)`` uint8 class series at the event pixels."""
    return np.ascontiguousarray(class_stack.classes[:, ev_rows, ev_cols].T)


def succession_summaries(seq: np.ndarray, class_ids: list[int], forest_id: int):
    """Per event: state changes, years per class, first year forest dominates."""
    n_ev, width = seq.shape
    obs = seq != NO_CLASS
    changes = np.zeros(n_ev, dtype=np.int16)
    for k in range(1, width):
        both = obs[:, k] & obs[:, k - 1]
        changes += (both & (seq[:, k] != seq[:, k - 1])).astype(np.int16)
    time_in = np.zeros((n_ev, len(class_ids)), dtype=np.uint8)
    for j, cid in enumerate(class_ids):
        time_in[:, j] = (seq == cid).sum(axis=1)
    is_forest = seq == forest_id
    first_forest = np.where(is_forest.any(axis=1), is_forest.argmax(axis=1), -1).astype(np.int16)
    last_obs = np.where(obs.any(axis=1), width - 1 - obs[:, ::-1].argmax(axis=1), -1)
    final = np.where(last_obs >= 0, seq[np.arange(n_ev), np.clip(last_obs, 0, None)], NO_CLASS).astype(np.uint8)
    return changes, time_in, first_forest, final


def transition_tables(seq: np.ndarray, event_type: np.ndarray, type_names: list[str], class_ids: list[int],
                      class_names: list[str]) -> tuple[dict, dict]:
    """Long-format Markov transition counts/probabilities and the class share
    by years since the event, per event type."""
    id_pos = {cid: j for j, cid in enumerate(class_ids)}
    n_cls = len(class_ids)
    width = seq.shape[1]
    tr = {"event_type": [], "from_class": [], "to_class": [], "count": [], "prob": []}
    sh = {"event_type": [], "years_since": [], "class": [], "n_observed": [], "share": []}
    for code, tname in enumerate(type_names):
        if code == 0:
            continue
        sel = seq[event_type == code]
        if not len(sel):
            continue
        counts = np.zeros((n_cls, n_cls), dtype=np.int64)
        for k in range(1, width):
            a, b = sel[:, k - 1], sel[:, k]
            ok = (a != NO_CLASS) & (b != NO_CLASS)
            for x, y in zip(a[ok], b[ok]):
                if int(x) in id_pos and int(y) in id_pos:
                    counts[id_pos[int(x)], id_pos[int(y)]] += 1
        for i in range(n_cls):
            row_sum = counts[i].sum()
            for j in range(n_cls):
                if counts[i, j]:
                    tr["event_type"].append(tname)
                    tr["from_class"].append(class_names[i])
                    tr["to_class"].append(class_names[j])
                    tr["count"].append(int(counts[i, j]))
                    tr["prob"].append(float(counts[i, j] / row_sum))
        for k in range(width):
            col = sel[:, k]
            n_obs = int((col != NO_CLASS).sum())
            if not n_obs:
                continue
            for j, cid in enumerate(class_ids):
                c = int((col == cid).sum())
                if c:
                    sh["event_type"].append(tname)
                    sh["years_since"].append(k)
                    sh["class"].append(class_names[j])
                    sh["n_observed"].append(n_obs)
                    sh["share"].append(c / n_obs)
    return tr, sh


def terrain_at(events_rows, events_cols, tile_dir, tile_id: str, dst_transform, dst_crs_wkt: str, dst_shape):
    """Elevation / slope / aspect at the event pixels (NaN when the tile has no DEM)."""
    n = len(events_rows)
    out = {k: np.full(n, np.nan, dtype=np.float32) for k in ("elevation", "slope", "aspect")}
    try:
        from landscape_change_detection_pipeline.change.regrowth_severity import _reproject_index_band
        from landscape_change_detection_pipeline.dem.zarr_store import dem_group_exists, read_tile_dem

        if not dem_group_exists(tile_dir, tile_id):
            return out, False
        for product in out:
            arr, transform, crs_wkt = read_tile_dem(tile_dir, tile_id, product)
            grid = _reproject_index_band(arr, tuple(transform)[:6], crs_wkt, dst_transform, dst_crs_wkt, dst_shape)
            out[product] = grid[events_rows, events_cols].astype(np.float32)
        return out, True
    except Exception as exc:  # noqa: BLE001 -- a missing/odd DEM must not lose the whole analysis
        print(f"[recovery_analysis] {tile_id}: DEM unavailable ({exc}); terrain columns are NaN", flush=True)
        return {k: np.full(n, np.nan, dtype=np.float32) for k in out}, False


def _bin_index(values: np.ndarray, edges: list[float]) -> np.ndarray:
    idx = np.digitize(values, edges[1:-1]).astype(np.int16)
    idx[~np.isfinite(values)] = -1
    return idx


def _bin_labels(edges: list[float], unit: str) -> list[str]:
    return [f"{edges[i]:g}-{edges[i + 1]:g}{unit}" for i in range(len(edges) - 1)]


def factor_summary(table: dict, cfg, group_columns: dict[str, tuple[np.ndarray, list[str]]], type_names: list[str]) -> dict:
    """Binned summary of recovery outcomes (see module docstring). ``group_columns``:
    factor name -> (int bin index per event, labels)."""
    ev_type = table["event_type"]
    recovered = table["recovered"].astype(bool)
    rec_year = table["recovery_year"].astype(np.float64)
    last = table["last_observed_year"].astype(np.float64)
    has_curve = np.isfinite(table["baseline"]) & (last >= 0)
    rows = {k: [] for k in (
        "factor", "event_type", "bin", "n", "n_recovered", "share_recovered", "median_recovery_years",
        "median_trough_rel", "median_rel_y5", "median_rel_y10", "recovered_by_y5", "recovered_by_y10", "median_rate",
    )}
    rel = table["curve_rel"]

    def add(factor, tname, label, mask):
        m = mask & has_curve
        n = int(m.sum())
        if n < cfg.min_group_size:
            return
        rec = m & recovered

        def med(a):
            a = a[np.isfinite(a)]
            return float(np.median(a)) if len(a) else np.nan

        def by(k):  # censoring-aware: only events observable for at least k years
            risk = m & (last >= k)
            return float((risk & recovered & (rec_year <= k)).sum() / risk.sum()) if risk.sum() >= cfg.min_group_size else np.nan

        rows["factor"].append(factor)
        rows["event_type"].append(tname)
        rows["bin"].append(label)
        rows["n"].append(n)
        rows["n_recovered"].append(int(rec.sum()))
        rows["share_recovered"].append(float(rec.sum() / n))
        rows["median_recovery_years"].append(med(rec_year[rec]))
        rows["median_trough_rel"].append(med(table["trough_rel"][m]))
        rows["median_rel_y5"].append(med(rel[m, 5]) if rel.shape[1] > 5 else np.nan)
        rows["median_rel_y10"].append(med(rel[m, 10]) if rel.shape[1] > 10 else np.nan)
        rows["recovered_by_y5"].append(by(5))
        rows["recovered_by_y10"].append(by(10))
        rows["median_rate"].append(med(table["rate"][m]))

    for factor, (bins, labels) in group_columns.items():
        for b, label in enumerate(labels):
            add(factor, "all", label, bins == b)
    for code, tname in enumerate(type_names):
        if code == 0:
            continue
        of_type = ev_type == code
        if of_type.any():
            add("event_type", tname, "all", of_type)
        for factor, (bins, labels) in group_columns.items():
            for b, label in enumerate(labels):
                add(f"{factor} x event_type", tname, label, of_type & (bins == b))
    return rows


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def analyse_events(
    events: dict, annual: np.ndarray, year0: int, class_stack, class_ids: list[int], class_names: list[str],
    cfg, gs: tuple[int, int], transform, crs_wkt: str, dst_shape, tile_dir, tile_id: str,
    mask_ids: tuple[int, ...], min_season_obs_override: Optional[int] = None,
) -> dict:
    """Every enabled element of ``recovery_analysis`` for one tile; returns the
    arrays to store (see :func:`write_recovery_analysis`)."""
    from datetime import date

    from affine import Affine

    from landscape_change_detection_pipeline.change.vegetation_dynamics import make_calendar

    prefix = f"[recovery_analysis] {tile_id}: "
    wanted = [EVENT_NAMES.index(n) for n in cfg.event_types if n in EVENT_NAMES]
    ev_names = list(events["event_names"])
    keep = np.flatnonzero(np.isin(events["event_type"], wanted))
    n_ev = len(keep)
    print(f"{prefix}{n_ev} events of types {list(cfg.event_types)} (of {len(events['event_type'])} breaks)", flush=True)
    out: dict[str, np.ndarray] = {}
    height, width = dst_shape
    if n_ev == 0:
        return out

    rows = events["row"][keep].astype(np.int64)
    cols = events["col"][keep].astype(np.int64)
    ev_type = events["event_type"][keep].astype(np.int8)
    ordinals = events["break_date"][keep].astype(np.int64)
    break_year = np.array([date.fromordinal(int(o)).year for o in ordinals], dtype=np.int64)
    break_idx = (break_year - year0).astype(np.int64)

    tbl = {
        "event_id": np.arange(n_ev, dtype=np.int32), "row": rows.astype(np.int32), "col": cols.astype(np.int32),
        "x": (transform[0] * (cols + 0.5) + transform[1] * (rows + 0.5) + transform[2]),
        "y": (transform[3] * (cols + 0.5) + transform[4] * (rows + 0.5) + transform[5]),
        "event_type": ev_type, "event_name": np.array([ev_names[c] for c in ev_type]),
        "break_ordinal": ordinals.astype(np.int32), "break_year": break_year.astype(np.int16),
    }
    for key in ("drop", "rise", "burned_fraction", "bare_fraction", "duration_months"):
        if key in events:
            tbl[key] = events[key][keep]
    for key, name in (("pre_forest", "pre_forest"), ("cleared_seasons", "cleared_seasons"), ("recovered", "forest_recovered")):
        if key in events:
            tbl[name] = events[key][keep]

    # ---- curves
    cur = cfg.curves
    curves_done = False
    if cur.enabled:
        t0 = time.time()
        k_max = int(cur.max_years)
        annual_ev = np.ascontiguousarray(annual[:, rows, cols].T)
        baseline = np.full(n_ev, np.nan, dtype=np.float32)
        curve_abs = np.full((n_ev, k_max + 1), np.nan, dtype=np.float32)
        curve_rel = np.full((n_ev, k_max + 1), np.nan, dtype=np.float32)
        trough_rel = np.full(n_ev, np.nan, dtype=np.float32)
        trough_year = np.full(n_ev, -1, dtype=np.int16)
        recovery_year = np.full(n_ev, -2, dtype=np.int16)
        last_year = np.full(n_ev, -1, dtype=np.int16)
        dropped = np.zeros(n_ev, dtype=np.bool_)
        rate = np.full(n_ev, np.nan, dtype=np.float32)
        final_rel = np.full(n_ev, np.nan, dtype=np.float32)
        curves_kernel(
            annual_ev, break_idx, k_max, int(cur.pre_years), int(cur.min_pre_years), float(cur.recovery_threshold),
            int(cur.sustain_years), baseline, curve_abs, curve_rel, trough_rel, trough_year, recovery_year,
            last_year, dropped, rate, final_rel,
        )
        recovered = recovery_year >= 0
        n_base = int(np.isfinite(baseline).sum())
        print(
            f"{prefix}curves: baseline for {n_base}/{n_ev} events, {int(recovered.sum())} recovered, "
            f"{int(((recovery_year == -1)).sum())} not yet, {time.time() - t0:.1f}s", flush=True,
        )
        tbl.update({
            "baseline": baseline, "trough_rel": trough_rel, "trough_year": trough_year, "dropped": dropped,
            "recovered": recovered, "recovery_year": recovery_year, "last_observed_year": last_year,
            "final_rel": final_rel, "rate": rate, "curve_abs": curve_abs, "curve_rel": curve_rel,
        })
        out["labels__tbl_events__curve_abs"] = np.arange(k_max + 1, dtype=np.int16)
        out["labels__tbl_events__curve_rel"] = np.arange(k_max + 1, dtype=np.int16)
        curves_done = True

        # per-pixel view of the latest event (later dates overwrite earlier ones)
        order = np.argsort(ordinals, kind="stable")
        state = np.zeros((height, width), dtype=np.uint8)
        ry = np.full((height, width), np.nan, dtype=np.float32)
        fr = np.full((height, width), np.nan, dtype=np.float32)
        st = np.where(recovery_year >= 0, 1, np.where(recovery_year == -1, 2, 3)).astype(np.uint8)
        r_o, c_o = rows[order], cols[order]
        state[r_o, c_o] = st[order]
        ry[r_o, c_o] = np.where(recovered, recovery_year, np.nan)[order]
        fr[r_o, c_o] = final_rel[order]
        out["grid_recovery_state"] = state  # 0 no event, 1 recovered, 2 not recovered yet, 3 no baseline
        out["grid_recovery_years"] = ry
        out["grid_final_rel"] = fr

    # ---- succession
    suc = cfg.succession
    if suc.enabled and class_stack is not None:
        t0 = time.time()
        k_max = int(suc.max_years)
        cal = make_calendar(list(class_stack.months))
        year_idx = (cal.year0 + cal.year_idx - year0).astype(np.int64)
        masked = np.zeros(256, dtype=np.bool_)
        masked[list(mask_ids)] = True
        masked[class_stack.nodata] = True
        series = dominant_class_series(class_stack, rows, cols)
        seq = np.full((n_ev, k_max + 1), NO_CLASS, dtype=np.uint8)
        succession_kernel(series, year_idx, cal.moy, int(gs[0]), int(gs[1]), int(suc.min_season_obs), masked,
                          break_idx, k_max, seq)
        forest_id = class_ids[class_names.index(suc.forest_class)] if suc.forest_class in class_names else -1
        changes, time_in, first_forest, final_state = succession_summaries(seq, class_ids, forest_id)
        tbl.update({
            "succession": seq, "n_state_changes": changes, "first_forest_year": first_forest,
            "final_class": final_state, "time_in_class": time_in,
        })
        out["labels__tbl_events__succession"] = np.arange(k_max + 1, dtype=np.int16)
        out["labels__tbl_events__time_in_class"] = np.array(class_names)
        tr, sh = transition_tables(seq, ev_type, ev_names, class_ids, class_names)
        for k, v in tr.items():
            out[f"tbl_transitions__{k}"] = np.array(v)
        for k, v in sh.items():
            out[f"tbl_state_share__{k}"] = np.array(v)
        out["succession_class_ids"] = np.array(class_ids, dtype=np.int16)
        out["succession_class_names"] = np.array(class_names)
        print(f"{prefix}succession: {len(tr['count'])} transition rows, {time.time() - t0:.1f}s", flush=True)
    elif suc.enabled:
        print(f"{prefix}succession skipped: no monthly class stack", flush=True)

    # ---- factors
    fac = cfg.factors
    if fac.enabled:
        terrain, has_dem = terrain_at(rows, cols, tile_dir, tile_id, Affine(*transform[:6]), crs_wkt, dst_shape)
        asp = np.deg2rad(terrain["aspect"])
        tbl["elevation"] = terrain["elevation"]
        tbl["slope"] = terrain["slope"]
        tbl["aspect"] = terrain["aspect"]
        flat = ~np.isfinite(terrain["slope"]) | (terrain["slope"] < 2.0)
        tbl["northness"] = np.where(flat, np.nan, np.cos(asp)).astype(np.float32)
        tbl["eastness"] = np.where(flat, np.nan, np.sin(asp)).astype(np.float32)
        n_sec = int(fac.aspect_sectors)
        sector = np.floor(((terrain["aspect"] + 180.0 / n_sec) % 360.0) / (360.0 / n_sec)).astype(np.int16)
        sector = np.where(flat | ~np.isfinite(terrain["aspect"]), -1, sector).astype(np.int16)
        el_bin = _bin_index(terrain["elevation"], list(fac.elevation_bins_m))
        sl_bin = _bin_index(terrain["slope"], list(fac.slope_bins_deg))
        tbl["elevation_band"], tbl["slope_band"], tbl["aspect_sector"] = el_bin, sl_bin, sector
        print(f"{prefix}factors: DEM {'found' if has_dem else 'not found (terrain NaN)'}", flush=True)
        if curves_done and has_dem:
            sector_labels = [f"{int(i * 360 / n_sec)}deg" for i in range(n_sec)]
            groups = {
                "elevation": (el_bin, _bin_labels(list(fac.elevation_bins_m), "m")),
                "slope": (sl_bin, _bin_labels(list(fac.slope_bins_deg), "deg")),
                "aspect": (sector, sector_labels),
            }
            summary = factor_summary(tbl, fac, groups, ev_names)
            for k, v in summary.items():
                out[f"tbl_factor_summary__{k}"] = np.array(v)
            print(f"{prefix}factors: {len(summary['n'])} summary rows", flush=True)
        elif not curves_done:
            print(f"{prefix}factors: binned summary needs curves (recovery_analysis.curves.enabled)", flush=True)

    for k, v in tbl.items():
        out[f"tbl_events__{k}"] = np.asarray(v)
    return out


def recovery_analysis_output_path(output_root: str | Path, tile_id: str) -> Path:
    return Path(output_root) / tile_id / "recovery_analysis.npz"


def write_recovery_analysis(path: str | Path, arrays: dict, transform, crs_wkt: str, resolution_m: float,
                            event_names, year0: int) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, **arrays, product=np.array("recovery_analysis"), event_names=np.array(list(event_names)), year0=np.array(year0, dtype=np.int32),
        transform=np.array(list(transform)[:6], dtype=np.float64), crs_wkt=np.array(crs_wkt),
        resolution_m=np.array(resolution_m, dtype=np.float64),
    )
    return out


def read_recovery_analysis(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as data:
        result = {k: np.array(data[k]) for k in data.files}
    result["crs_wkt"] = str(result["crs_wkt"])
    result["transform"] = tuple(float(v) for v in result["transform"])
    return result
