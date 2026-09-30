"""Per-pixel vegetation dynamics: phenology, slow trends, anomalies, variability.

Purpose
-------
Stage 09 finds *abrupt* change (breaks). Everything here is the other half:
what the vegetation does *between* breaks, read from the monthly index series
(NDVI by default) of every pixel.

Source: a monthly index cube, on disk
-------------------------------------
For each index in ``indices`` one ``(n_months, H, W)`` float16 cube is built
on the Stage 09 grid, month by month (the same per-scene indices as
:mod:`change.regrowth_severity`, cloud/shadow already excluded), and every
month whose class composite is snow, ice or water (``mask_classes``) is set
to NaN -- an NDVI under snow is not vegetation. The cube is a temporary
memmap in the tile's work folder, read back in row blocks, and deleted at the
end (nothing persistent this big; see :mod:`change.spectral_composites` for
why index composites are not stored).

What is computed (each element can be switched off in the config)
-----------------------------------------------------------------
Yearly series first (always, other elements and
:mod:`change.recovery_analysis` need them): the growing-season mean and the
maximum of each calendar year (``min_gs_obs`` observed growing-season months
needed).

- **phenology**: per pixel per year, the peak month and value, the amplitude
  and the start / end / length of the season -- the dates (day of year) the
  monthly curve, linearly interpolated between observed months, crosses
  ``low + sos_fraction * (peak - low)``, where ``low`` is the pixel's
  ``baseline_percentile`` over all its observations. A crossing that needs
  more than ``max_gap_months`` unobserved months in a row, or that happened
  before the first observed month (a snow-covered spring), is not invented:
  that year has no start/end for that pixel. Monthly data is coarse: read
  the dates as ~15-day precision at best.
- **trend**: Theil-Sen slope (per year) and Mann-Kendall statistic / p-value
  of the yearly series -- slow greening or browning the segmentation cannot
  see; optionally again on the years after each pixel's latest break only
  (``since_last_break``), because a slope fitted across a clear-cut is just
  the clear-cut. The phenology series (start/end/length) get their own
  trends: is the season getting longer.
- **anomalies**: the monthly value minus the pixel's own climatology for
  that calendar month, over the scale (robust: median / MAD*1.4826), for the
  whole series or a ``baseline_years`` window. Kept per year: the mean
  growing-season z-score, the worst month, the number of extreme months; and
  overall: the worst anomaly ever seen and when. Droughts, late snowmelt and
  odd summers show up here without being mistaken for a season.
- **variability**: coefficient of variation of the yearly mean/max and the
  mean absolute year-to-year change.
"""

from __future__ import annotations

import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numba import njit, prange

from landscape_change_detection_pipeline.change.landcover_persistence import MonthlyClassStack, _abs_month

MONTH_DAYS = 30.4375


# ---------------------------------------------------------------------------
# calendar
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Calendar:
    """Position of every cube month in the year grid."""

    months: tuple[str, ...]
    year0: int
    n_years: int
    year_idx: np.ndarray  # (n_months,) int64, 0 = year0
    moy: np.ndarray  # (n_months,) int64, 1..12

    @property
    def years(self) -> np.ndarray:
        return np.arange(self.year0, self.year0 + self.n_years, dtype=np.int32)


def make_calendar(months: list[str]) -> Calendar:
    months = sorted(months)
    abs_m = np.array([_abs_month(m) for m in months], dtype=np.int64)
    years = abs_m // 12 + 1970
    year0 = int(years.min())
    return Calendar(
        months=tuple(months), year0=year0, n_years=int(years.max()) - year0 + 1,
        year_idx=(years - year0).astype(np.int64), moy=(abs_m % 12 + 1).astype(np.int64),
    )


# ---------------------------------------------------------------------------
# the monthly cube (temporary, on disk)
# ---------------------------------------------------------------------------


def build_index_cubes(
    tile_dir, inference_root, tile_id: str, class_config, months: list[str], indices: list[str],
    dst_transform, dst_crs_wkt: str, dst_shape: tuple[int, int], work_dir: Path,
    class_stack: Optional[MonthlyClassStack], mask_ids: tuple[int, ...], month_threads: int = 1,
) -> dict[str, np.ndarray]:
    """One float16 ``(n_months, H, W)`` memmap per index (NaN = unobserved).

    Months are built once for every index (one scene read serves all of
    them), ``month_threads`` months ahead. ``class_stack`` (optional) masks
    the months whose class composite is one of ``mask_ids``."""
    from landscape_change_detection_pipeline.change.regrowth_severity import (
        _iter_prefetched,
        _reproject_index_band,
        build_month_composite_cache,
    )

    work_dir.mkdir(parents=True, exist_ok=True)
    height, width = dst_shape
    months = sorted(months)
    cubes = {
        name: np.lib.format.open_memmap(
            work_dir / f"cube_{name}.npy", mode="w+", dtype=np.float16, shape=(len(months), height, width)
        )
        for name in indices
    }
    get_month = build_month_composite_cache(tile_dir, inference_root, tile_id, class_config, tuple(indices))
    stack_pos = {m: i for i, m in enumerate(class_stack.months)} if class_stack is not None else {}
    mask_table = np.zeros(256, dtype=bool)
    mask_table[list(mask_ids)] = True

    def fetch(month: str):
        composite = get_month(month)
        if composite is None:
            return None
        out = {}
        for name in indices:
            if name in composite["stats"]:
                out[name] = _reproject_index_band(
                    composite["stats"][name]["median"], composite["transform"], composite["crs_wkt"],
                    dst_transform, dst_crs_wkt, dst_shape,
                )
        return out

    t0 = time.time()
    n_built = 0
    for k, (month, arrays) in enumerate(_iter_prefetched(months, fetch, month_threads), start=1):
        if k == 1 or k % 25 == 0 or k == len(months):
            print(
                f"[vegetation_dynamics] {tile_id}: cube month {k}/{len(months)} ({month}), "
                f"{time.time() - t0:.0f}s elapsed", flush=True,
            )
        if arrays is None:
            for cube in cubes.values():
                cube[k - 1] = np.nan
            continue
        n_built += 1
        masked = None
        if month in stack_pos:
            masked = mask_table[class_stack.classes[stack_pos[month]]]
        for name, cube in cubes.items():
            arr = arrays.get(name)
            if arr is None:
                cube[k - 1] = np.nan
                continue
            if masked is not None:
                arr = np.where(masked, np.nan, arr)
            cube[k - 1] = arr
    for cube in cubes.values():
        cube.flush()
    print(f"[vegetation_dynamics] {tile_id}: cube built ({n_built}/{len(months)} months with data)", flush=True)
    return cubes


def cleanup_cubes(cubes: dict[str, np.ndarray], work_dir: Path) -> None:
    for cube in cubes.values():
        mm = getattr(cube, "_mmap", None)
        if mm is not None:
            mm.close()
    cubes.clear()
    shutil.rmtree(work_dir, ignore_errors=True)


def block_series(cube: np.ndarray, r0: int, r1: int) -> np.ndarray:
    """``(n_pix, n_months)`` float32 for rows ``r0:r1`` (pixel-major)."""
    n_months, _, width = cube.shape
    block = np.asarray(cube[:, r0:r1, :], dtype=np.float32)
    return np.ascontiguousarray(block.reshape(n_months, -1).T)


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def yearly_stats(series, year_idx, moy, n_years, gs0, gs1, min_gs_obs, gs_mean, gs_max, year_obs, low_pct, low):
    """Growing-season mean, annual maximum, number of observed months per
    year, and the pixel's ``low_pct`` percentile over all observations."""
    n_pix, n_months = series.shape
    for p in prange(n_pix):
        s = np.zeros(n_years)
        c = np.zeros(n_years, dtype=np.int64)
        mx = np.full(n_years, -np.inf)
        nobs = np.zeros(n_years, dtype=np.int64)
        allv = np.empty(n_months)
        na = 0
        for t in range(n_months):
            v = series[p, t]
            if v != v:
                continue
            y = year_idx[t]
            nobs[y] += 1
            allv[na] = v
            na += 1
            if v > mx[y]:
                mx[y] = v
            if moy[t] >= gs0 and moy[t] <= gs1:
                s[y] += v
                c[y] += 1
        for y in range(n_years):
            year_obs[p, y] = nobs[y]
            if c[y] >= min_gs_obs:
                gs_mean[p, y] = s[y] / c[y]
                gs_max[p, y] = mx[y]
        if na > 0:
            srt = np.sort(allv[:na])
            low[p] = srt[min(na - 1, int(low_pct / 100.0 * (na - 1) + 0.5))]


@njit(parallel=True, cache=True)
def phenology_kernel(series, year_idx, moy, n_years, low, frac, min_year_obs, min_amp, max_gap,
                     peak_moy, peak_val, amp, sos, eos):
    """Per pixel-year peak month/value, amplitude and start/end of season as
    day of year (0 = not determinable)."""
    n_pix, n_months = series.shape
    for p in prange(n_pix):
        vals = np.full((n_years, 12), np.nan)
        for t in range(n_months):
            v = series[p, t]
            if v == v:
                vals[year_idx[t], moy[t] - 1] = v
        for y in range(n_years):
            n_valid = 0
            best = -np.inf
            best_m = -1
            for m in range(12):
                v = vals[y, m]
                if v == v:
                    n_valid += 1
                    if v > best:
                        best = v
                        best_m = m
            if n_valid < min_year_obs:
                continue
            a = best - low[p]
            peak_moy[p, y] = best_m + 1
            peak_val[p, y] = best
            amp[p, y] = a
            if a < min_amp:
                continue
            thr = low[p] + frac * a
            # start: walk back from the peak to the last month below the threshold
            j = best_m
            gap_ok = True
            found = False
            while j > 0:
                k = j - 1
                g = 0
                while k >= 0 and vals[y, k] != vals[y, k]:
                    k -= 1
                    g += 1
                if k < 0:
                    break
                if g > max_gap:
                    gap_ok = False
                    break
                if vals[y, k] < thr:
                    d0 = k * MONTH_DAYS + MONTH_DAYS / 2.0
                    d1 = j * MONTH_DAYS + MONTH_DAYS / 2.0
                    sos[p, y] = d0 + (thr - vals[y, k]) / (vals[y, j] - vals[y, k]) * (d1 - d0)
                    found = True
                    break
                j = k
            if not found or not gap_ok:
                sos[p, y] = 0
            j = best_m
            gap_ok = True
            found = False
            while j < 11:
                k = j + 1
                g = 0
                while k <= 11 and vals[y, k] != vals[y, k]:
                    k += 1
                    g += 1
                if k > 11:
                    break
                if g > max_gap:
                    gap_ok = False
                    break
                if vals[y, k] < thr:
                    d0 = j * MONTH_DAYS + MONTH_DAYS / 2.0
                    d1 = k * MONTH_DAYS + MONTH_DAYS / 2.0
                    eos[p, y] = d0 + (vals[y, j] - thr) / (vals[y, j] - vals[y, k]) * (d1 - d0)
                    found = True
                    break
                j = k
            if not found or not gap_ok:
                eos[p, y] = 0


@njit(parallel=True, cache=True)
def trend_kernel(yearly, start_idx, min_years, slope, mk_s, mk_p, n_used):
    """Theil-Sen slope and Mann-Kendall S / two-sided p on each row of
    ``yearly`` ((n_pix, n_years), NaN = missing), from ``start_idx[p]`` on."""
    n_pix, n_years = yearly.shape
    for p in prange(n_pix):
        xs = np.empty(n_years)
        vs = np.empty(n_years)
        n = 0
        for y in range(start_idx[p], n_years):
            v = yearly[p, y]
            if v == v and v > -1e8:
                xs[n] = y
                vs[n] = v
                n += 1
        n_used[p] = n
        if n < min_years:
            continue
        m = n * (n - 1) // 2
        sl = np.empty(m)
        s = 0.0
        k = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                d = vs[j] - vs[i]
                sl[k] = d / (xs[j] - xs[i])
                k += 1
                if d > 0:
                    s += 1.0
                elif d < 0:
                    s -= 1.0
        srt = np.sort(sl)
        slope[p] = srt[m // 2] if m % 2 == 1 else 0.5 * (srt[m // 2 - 1] + srt[m // 2])
        mk_s[p] = s
        var = n * (n - 1) * (2 * n + 5) / 18.0
        z = 0.0
        if s > 0:
            z = (s - 1.0) / math.sqrt(var)
        elif s < 0:
            z = (s + 1.0) / math.sqrt(var)
        mk_p[p] = math.erfc(abs(z) / math.sqrt(2.0))


@njit(parallel=True, cache=True)
def anomaly_kernel(series, year_idx, moy, n_years, base0, base1, robust, min_clim_obs, min_scale, extreme_z,
                   gs0, gs1, min_gs_obs, clim_center, clim_scale, gs_z, min_z, n_neg, n_pos,
                   worst_z, worst_month, z_out, want_z):
    """Monthly z-scores against the pixel's climatology per calendar month
    (see module docstring), reduced per year; ``z_out`` receives the monthly
    scores only when ``want_z``."""
    n_pix, n_months = series.shape
    for p in prange(n_pix):
        buf = np.empty(n_months)
        center = np.full(12, np.nan)
        scale = np.full(12, np.nan)
        for c in range(12):
            n = 0
            for t in range(n_months):
                if moy[t] == c + 1 and year_idx[t] >= base0 and year_idx[t] <= base1:
                    v = series[p, t]
                    if v == v:
                        buf[n] = v
                        n += 1
            if n < min_clim_obs:
                continue
            vals = np.sort(buf[:n])
            if robust:
                med = vals[n // 2] if n % 2 == 1 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
                dev = np.abs(vals - med)
                dev = np.sort(dev)
                mad = dev[n // 2] if n % 2 == 1 else 0.5 * (dev[n // 2 - 1] + dev[n // 2])
                center[c] = med
                scale[c] = max(1.4826 * mad, min_scale)
            else:
                mu = 0.0
                for i in range(n):
                    mu += vals[i]
                mu /= n
                var = 0.0
                for i in range(n):
                    var += (vals[i] - mu) ** 2
                center[c] = mu
                scale[c] = max(math.sqrt(var / max(n - 1, 1)), min_scale)
            clim_center[p, c] = center[c]
            clim_scale[p, c] = scale[c]
        gs_sum = np.zeros(n_years)
        gs_n = np.zeros(n_years, dtype=np.int64)
        wz = 0.0
        wt = -1
        for t in range(n_months):
            v = series[p, t]
            c = moy[t] - 1
            if v != v or center[c] != center[c]:
                continue
            z = (v - center[c]) / scale[c]
            if want_z:
                z_out[t, p] = z
            y = year_idx[t]
            if z < min_z[p, y]:
                min_z[p, y] = z
            if z <= -extreme_z:
                n_neg[p, y] += 1
            if z >= extreme_z:
                n_pos[p, y] += 1
            if moy[t] >= gs0 and moy[t] <= gs1:
                gs_sum[y] += z
                gs_n[y] += 1
            if abs(z) > abs(wz):
                wz = z
                wt = t
        for y in range(n_years):
            if gs_n[y] >= min_gs_obs:
                gs_z[p, y] = gs_sum[y] / gs_n[y]
        if wt >= 0:
            worst_z[p] = wz
            worst_month[p] = wt


# ---------------------------------------------------------------------------
# output accumulation
# ---------------------------------------------------------------------------


class Accumulator:
    """Full-tile output arrays, allocated on first use, filled block by block."""

    def __init__(self, height: int, width: int):
        self.height, self.width = height, width
        self.arrays: dict[str, np.ndarray] = {}
        self._fill: dict[str, object] = {}

    def add(self, name: str, lead: tuple[int, ...], dtype, fill) -> None:
        self.arrays[name] = np.full((*lead, self.height, self.width), fill, dtype=dtype)
        self._fill[name] = fill

    def put(self, name: str, r0: int, r1: int, block: np.ndarray) -> None:
        """``block``: ``(n_pix, *lead)`` -> stored at rows ``r0:r1``; lead axes go first."""
        arr = self.arrays[name]
        rows = r1 - r0
        if arr.ndim == 2:
            arr[r0:r1, :] = block.reshape(rows, self.width).astype(arr.dtype, copy=False)
        else:
            moved = np.moveaxis(block.reshape(rows, self.width, -1), -1, 0)
            arr[:, r0:r1, :] = moved.astype(arr.dtype, copy=False)


def _to_int16_doy(a: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(a), np.rint(a), 0).astype(np.int16)


def compute_tile_dynamics(
    cubes: dict[str, np.ndarray], calendar: Calendar, cfg, latest_break_year_idx: np.ndarray,
    tile_id: str = "", annual_out: Optional[dict] = None,
) -> tuple[dict[str, np.ndarray], Optional[np.ndarray]]:
    """Run every enabled element on every index block by block. Returns
    ``(arrays, monthly_z)``: the output arrays (keys ``<index>__<name>``, plus
    ``labels__<key>`` for the axes) and, if ``anomalies.store_monthly``,
    ``{index: z cube}`` memmaps in a dict (else ``None``).

    ``annual_out`` (a dict) receives ``{index: (n_years, H, W) float32}`` yearly
    growing-season means for :mod:`change.recovery_analysis`."""
    any_cube = next(iter(cubes.values()))
    _, height, width = any_cube.shape
    n_years = calendar.n_years
    gs0, gs1 = int(cfg.growing_season[0]), int(cfg.growing_season[1])
    ph, tr, an, va = cfg.phenology, cfg.trend, cfg.anomalies, cfg.variability
    out: dict[str, np.ndarray] = {}
    monthly_z: dict[str, np.ndarray] = {}
    block_rows = max(1, int(cfg.block_rows))
    blocks = [(r, min(r + block_rows, height)) for r in range(0, height, block_rows)]
    br0 = 0
    br1 = n_years - 1
    if an.enabled and an.baseline_years is not None:
        br0 = max(0, int(an.baseline_years[0]) - calendar.year0)
        br1 = min(n_years - 1, int(an.baseline_years[1]) - calendar.year0)

    for name, cube in cubes.items():
        acc = Accumulator(height, width)
        keep_annual = annual_out is not None
        if keep_annual:
            annual_out[name] = np.full((n_years, height, width), np.nan, dtype=np.float32)
        acc.add("growing_season_mean", (n_years,), np.float16, np.nan)
        acc.add("annual_max", (n_years,), np.float16, np.nan)
        acc.add("n_obs_year", (n_years,), np.uint8, 0)
        if ph.enabled:
            acc.add("peak_month", (n_years,), np.uint8, 0)
            acc.add("peak_value", (n_years,), np.float16, np.nan)
            acc.add("amplitude", (n_years,), np.float16, np.nan)
            acc.add("season_start_doy", (n_years,), np.int16, 0)
            acc.add("season_end_doy", (n_years,), np.int16, 0)
            acc.add("season_length_days", (n_years,), np.int16, 0)
        if an.enabled:
            acc.add("clim_center", (12,), np.float16, np.nan)
            acc.add("clim_scale", (12,), np.float16, np.nan)
            acc.add("gs_anomaly_z", (n_years,), np.float16, np.nan)
            acc.add("worst_month_z", (n_years,), np.float16, np.nan)
            acc.add("n_extreme_low_months", (n_years,), np.uint8, 0)
            acc.add("n_extreme_high_months", (n_years,), np.uint8, 0)
            acc.add("worst_anomaly_z", (), np.float32, np.nan)
            acc.add("worst_anomaly_month_index", (), np.int16, -1)
            if an.store_monthly:
                monthly_z[name] = None  # allocated below
        trend_series = list(tr.series) if tr.enabled else []
        if tr.enabled and ph.enabled and ph.trends:
            trend_series += ["season_start", "season_end", "season_length"]
        for s in trend_series:
            for suffix in ("", "_since_break") if tr.since_last_break else ("",):
                acc.add(f"trend_{s}{suffix}_slope", (), np.float32, np.nan)
                acc.add(f"trend_{s}{suffix}_mk_s", (), np.float32, np.nan)
                acc.add(f"trend_{s}{suffix}_p", (), np.float32, np.nan)
                acc.add(f"trend_{s}{suffix}_n_years", (), np.uint8, 0)
        if va.enabled:
            for s in ("growing_season_mean", "annual_max"):
                acc.add(f"variability_{s}_cv", (), np.float32, np.nan)
                acc.add(f"variability_{s}_mean_abs_change", (), np.float32, np.nan)

        z_cube = None
        if an.enabled and an.store_monthly:
            z_cube = np.lib.format.open_memmap(
                Path(getattr(cube, "filename", "cube.npy")).with_name(f"z_{name}.npy"),
                mode="w+", dtype=np.float16, shape=(len(calendar.months), height, width),
            )

        t0 = time.time()
        for b, (r0, r1) in enumerate(blocks, start=1):
            if b == 1 or b % 5 == 0 or b == len(blocks):
                print(
                    f"[vegetation_dynamics] {tile_id}: {name} block {b}/{len(blocks)} "
                    f"(rows {r0}-{r1}), {time.time() - t0:.0f}s", flush=True,
                )
            series = block_series(cube, r0, r1)
            n_pix = series.shape[0]
            gs_mean = np.full((n_pix, n_years), np.nan, dtype=np.float32)
            gs_max = np.full((n_pix, n_years), np.nan, dtype=np.float32)
            year_obs = np.zeros((n_pix, n_years), dtype=np.int64)
            low = np.full(n_pix, np.nan)
            yearly_stats(
                series, calendar.year_idx, calendar.moy, n_years, gs0, gs1, int(cfg.min_gs_obs),
                gs_mean, gs_max, year_obs, float(ph.baseline_percentile), low,
            )
            acc.put("growing_season_mean", r0, r1, gs_mean)
            acc.put("annual_max", r0, r1, gs_max)
            acc.put("n_obs_year", r0, r1, year_obs)
            if keep_annual:
                annual_out[name][:, r0:r1, :] = np.moveaxis(gs_mean.reshape(r1 - r0, width, n_years), -1, 0)

            yearly = {"growing_season_mean": gs_mean, "annual_max": gs_max}
            if ph.enabled:
                peak_moy = np.zeros((n_pix, n_years), dtype=np.uint8)
                peak_val = np.full((n_pix, n_years), np.nan, dtype=np.float32)
                amp = np.full((n_pix, n_years), np.nan, dtype=np.float32)
                sos = np.zeros((n_pix, n_years), dtype=np.float64)
                eos = np.zeros((n_pix, n_years), dtype=np.float64)
                phenology_kernel(
                    series, calendar.year_idx, calendar.moy, n_years, low, float(ph.sos_fraction),
                    int(cfg.min_year_obs), float(ph.min_amplitude), int(ph.max_gap_months),
                    peak_moy, peak_val, amp, sos, eos,
                )
                sos_i, eos_i = _to_int16_doy(sos), _to_int16_doy(eos)
                los = np.where((sos_i > 0) & (eos_i > sos_i), eos_i - sos_i, 0).astype(np.int16)
                acc.put("peak_month", r0, r1, peak_moy)
                acc.put("peak_value", r0, r1, peak_val)
                acc.put("amplitude", r0, r1, amp)
                acc.put("season_start_doy", r0, r1, sos_i)
                acc.put("season_end_doy", r0, r1, eos_i)
                acc.put("season_length_days", r0, r1, los)
                nan_if0 = lambda a: np.where(a > 0, a, np.nan).astype(np.float32)  # noqa: E731
                yearly["season_start"] = nan_if0(sos_i)
                yearly["season_end"] = nan_if0(eos_i)
                yearly["season_length"] = nan_if0(los)

            if an.enabled:
                clim_c = np.full((n_pix, 12), np.nan, dtype=np.float32)
                clim_s = np.full((n_pix, 12), np.nan, dtype=np.float32)
                gsz = np.full((n_pix, n_years), np.nan, dtype=np.float32)
                minz = np.full((n_pix, n_years), np.inf, dtype=np.float32)
                nneg = np.zeros((n_pix, n_years), dtype=np.uint8)
                npos = np.zeros((n_pix, n_years), dtype=np.uint8)
                worst = np.full(n_pix, np.nan, dtype=np.float32)
                worst_t = np.full(n_pix, -1, dtype=np.int64)
                z_block = np.full((series.shape[1], n_pix), np.nan, dtype=np.float32) if z_cube is not None else np.empty((1, 1), dtype=np.float32)
                anomaly_kernel(
                    series, calendar.year_idx, calendar.moy, n_years, br0, br1, bool(an.robust),
                    int(an.min_clim_obs), float(an.min_scale), float(an.extreme_z), gs0, gs1, int(cfg.min_gs_obs),
                    clim_c, clim_s, gsz, minz, nneg, npos, worst, worst_t, z_block, z_cube is not None,
                )
                minz = np.where(np.isfinite(minz), minz, np.nan)
                acc.put("clim_center", r0, r1, clim_c)
                acc.put("clim_scale", r0, r1, clim_s)
                acc.put("gs_anomaly_z", r0, r1, gsz)
                acc.put("worst_month_z", r0, r1, minz)
                acc.put("n_extreme_low_months", r0, r1, nneg)
                acc.put("n_extreme_high_months", r0, r1, npos)
                acc.put("worst_anomaly_z", r0, r1, worst)
                acc.put("worst_anomaly_month_index", r0, r1, worst_t)
                if z_cube is not None:
                    z_cube[:, r0:r1, :] = z_block.reshape(-1, r1 - r0, width).astype(np.float16)

            break_idx = latest_break_year_idx[r0:r1, :].reshape(-1).astype(np.int64)
            for s in trend_series:
                for suffix in ("", "_since_break") if tr.since_last_break else ("",):
                    start = np.zeros(n_pix, dtype=np.int64) if suffix == "" else np.maximum(break_idx + 1, 0)
                    slope = np.full(n_pix, np.nan, dtype=np.float32)
                    mk_s = np.full(n_pix, np.nan, dtype=np.float32)
                    mk_p = np.full(n_pix, np.nan, dtype=np.float32)
                    used = np.zeros(n_pix, dtype=np.int64)
                    if suffix == "_since_break":
                        no_break = break_idx < 0
                        start = np.where(no_break, 0, start)
                    trend_kernel(yearly[s], start, int(tr.min_years), slope, mk_s, mk_p, used)
                    acc.put(f"trend_{s}{suffix}_slope", r0, r1, slope)
                    acc.put(f"trend_{s}{suffix}_mk_s", r0, r1, mk_s)
                    acc.put(f"trend_{s}{suffix}_p", r0, r1, mk_p)
                    acc.put(f"trend_{s}{suffix}_n_years", r0, r1, used)

            if va.enabled:
                for s in ("growing_season_mean", "annual_max"):
                    a = yearly[s]
                    n_ok = np.isfinite(a).sum(axis=1)
                    with np.errstate(all="ignore"):
                        mean = np.nanmean(np.where(np.isfinite(a), a, np.nan), axis=1)
                        std = np.nanstd(np.where(np.isfinite(a), a, np.nan), axis=1, ddof=1)
                        cv = np.where(np.abs(mean) > 1e-3, std / np.abs(mean), np.nan)
                        diffs = np.abs(np.diff(a, axis=1))
                        # a difference only counts between two consecutive observed years
                        mac = np.nanmean(diffs, axis=1)
                    ok = n_ok >= int(va.min_years)
                    acc.put(f"variability_{s}_cv", r0, r1, np.where(ok, cv, np.nan))
                    acc.put(f"variability_{s}_mean_abs_change", r0, r1, np.where(ok, mac, np.nan))

        if z_cube is not None:
            z_cube.flush()
            monthly_z[name] = z_cube
        for key, arr in acc.arrays.items():
            out[f"{name.lower()}__{key}"] = arr
            if arr.ndim == 3 and arr.shape[0] == n_years:
                out[f"labels__{name.lower()}__{key}"] = calendar.years
            elif arr.ndim == 3 and arr.shape[0] == 12:
                out[f"labels__{name.lower()}__{key}"] = np.arange(1, 13, dtype=np.int16)
    return out, (monthly_z if monthly_z else None)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def vegetation_dynamics_output_path(output_root: str | Path, tile_id: str) -> Path:
    return Path(output_root) / tile_id / "vegetation_dynamics.npz"


def monthly_anomaly_output_path(output_root: str | Path, tile_id: str, index: str) -> Path:
    return Path(output_root) / tile_id / f"monthly_anomaly_{index.lower()}.npz"


def write_vegetation_dynamics(path: str | Path, arrays: dict, calendar: Calendar, transform, crs_wkt: str,
                              resolution_m: float, cfg) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, **arrays,
        product=np.array("vegetation_dynamics"), years=calendar.years, months=np.array(calendar.months),
        growing_season=np.array(cfg.growing_season, dtype=np.int8),
        transform=np.array(list(transform)[:6], dtype=np.float64),
        crs_wkt=np.array(crs_wkt), resolution_m=np.array(resolution_m, dtype=np.float64),
    )
    return out


def write_monthly_anomaly(path: str | Path, z_cube: np.ndarray, calendar: Calendar, transform, crs_wkt: str,
                          resolution_m: float) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, product=np.array("monthly_anomaly"), monthly_z=np.asarray(z_cube), labels__monthly_z=np.array(calendar.months),
        transform=np.array(list(transform)[:6], dtype=np.float64),
        crs_wkt=np.array(crs_wkt), resolution_m=np.array(resolution_m, dtype=np.float64),
    )
    return out


def read_vegetation_dynamics(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as data:
        result = {k: np.array(data[k]) for k in data.files}
    result["crs_wkt"] = str(result["crs_wkt"])
    result["transform"] = tuple(float(v) for v in result["transform"])
    return result


def latest_break_year_index(segments_result: dict, year0: int) -> np.ndarray:
    """``(H, W)`` int16: calendar year of each pixel's latest break minus
    ``year0`` (-1 = no break)."""
    from datetime import date

    from landscape_change_detection_pipeline.change.regrowth_severity import latest_break_arrays

    height, width = segments_result["shape"]
    grid = np.full((height, width), -1, dtype=np.int16)
    breaks, _ = latest_break_arrays(segments_result)
    if len(breaks):
        years = np.array([date.fromordinal(int(o)).year for o in breaks.ordinals], dtype=np.int64)
        grid[breaks.rows, breaks.cols] = np.clip(years - year0, -1, 32000).astype(np.int16)
    return grid
