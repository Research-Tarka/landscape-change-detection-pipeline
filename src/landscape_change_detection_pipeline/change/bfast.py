"""BFAST-Monitor: pure-Python fire/burn-severity break detection cross-check.

Purpose
-------
A per-pixel fire-focused cross-check alongside :mod:`change.ccdc` -- a 2025
multi-algorithm comparison found BFAST dramatically stronger specifically for fire
detection (~96% vs ~73% for CCDC/LandTrendr-family methods), which is why
this module exists as a narrow, fire-specific cross-check rather than a
third general-purpose change detector (that role is CCDC's).

Why hand-implemented, not a wrapped library
-----------------------------------------------
Both off-the-shelf options were rejected: the pip ``bfast`` package is dead (last release 2021, Python
<=3.6 pins, an unmaintained OpenCL GPU path); R's actively-maintained
``bfast``/``bfastmonitor`` would work but requires adding R + ``rpy2`` as a
second language runtime, which this project explicitly chose not to do to
stay a single Python/conda stack. This module reimplements
``bfastmonitor``'s actual method (Verbesselt, Zeileis & Herold 2012,
*Remote Sensing of Environment* 123:98-108) directly.

The algorithm, as implemented here
--------------------------------------
1. **History-period model**: a harmonic regression (trend + seasonal terms)
   fit by OLS on the stable "history" period:
   ``y_t = b0 + b1*trend_t + sum_k[ a_k*cos(2*pi*k*t/freq) + b_k*sin(2*pi*k*t/freq) ]``
   for harmonic order ``k = 1..K`` (default ``K=3``, matching ``bfast``'s
   own default), where ``trend_t`` is a plain integer time index and
   ``freq`` is the seasonal period in the same time units as ``t`` (for an
   irregular satellite time series, "days since history start" with
   ``freq=365.25`` -- an annual cycle -- following ``bfastpp``'s own
   treatment of irregular time series via a continuous seasonal phase
   rather than an integer season index, which only works for regularly
   sampled series).
2. **Monitoring-period test**: OLS-CUSUM, not OLS-MOSUM. ``bfastmonitor``'s
   own default is OLS-MOSUM, whose critical values come from a
   Monte-Carlo-simulated boundary-crossing table (``strucchange``'s
   ``monitorMECritvalTable``) that has no closed-form formula to
   reimplement without re-simulating it. OLS-CUSUM instead has an exact,
   closed-form boundary derived from Brownian-bridge crossing probabilities
   (see :func:`cusum_boundary_constant`), which is what
   ``strucchange``/``bfast`` themselves fall back to for the history-period
   stability pre-check (their "ROC" step) -- so this module uses that same
   closed-form branch as its one and only test, rather than approximating a
   table it cannot exactly reproduce. This is a documented, standard
   simplification when reimplementing ``bfastmonitor`` outside the
   ``strucchange`` ecosystem.
3. **Break detection**: the first monitoring-period observation where the
   cumulative-sum process's absolute value exceeds its boundary is the
   detected break. Magnitude is the median residual (observed minus
   predicted) from that point onward -- a robust summary of how large the
   departure from the history model is, not derived from the test
   statistic itself (matching ``bfastmonitor``'s own magnitude definition).

Input
-----
Runs on the same per-tile, per-pixel aligned time series
:mod:`change.ccdc` already builds (:class:`change.ccdc.TileTimeSeries`) --
one shared grid per tile at its finest-ever resolution, reflectance bands
plus a per-pixel QA mask from this project's own trained classification.
BFAST here is applied to a single burn-sensitive index (NBR, the standard
choice for fire -- normalized burn ratio) computed from that same aligned
band stack, not the raw six bands directly: a scalar index is what the
univariate ``bfastmonitor`` formulation expects, and NBR is specifically the
band combination most sensitive to the biomass/char signature fire leaves
behind.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from landscape_change_detection_pipeline.change.ccdc import CCDC_BAND_ORDER, TileTimeSeries, _QA_CLEAR

#: Seasonal period, in days -- an annual cycle, matching ``bfastpp``'s
#: continuous-time seasonal phase treatment for irregular time series.
ANNUAL_PERIOD_DAYS = 365.25


def nbr_from_bands(bands: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """NBR (normalized burn ratio) from a ``(..., n_bands)`` or
    ``(n_bands, ...)``-last-axis stack in :data:`change.ccdc.CCDC_BAND_ORDER`
    order. Accepts the ``(n_obs, n_bands)`` per-pixel stack
    :mod:`change.ccdc` builds directly."""
    nir_idx = CCDC_BAND_ORDER.index("nir")
    swir2_idx = CCDC_BAND_ORDER.index("swir2")
    nir = bands[..., nir_idx]
    swir2 = bands[..., swir2_idx]
    return ((nir - swir2) / (nir + swir2 + eps)).astype(np.float64)


def cusum_boundary_constant(alpha: float = 0.05) -> float:
    """The closed-form Brownian-bridge boundary constant ``a`` solving
    ``2*(Phi(a) - a*phi(a)) + alpha - 2 = 0`` (``strucchange``'s own
    closed-form OLS-CUSUM boundary, used here as this module's one and only
    test -- see module docstring for why, in place of OLS-MOSUM's
    Monte-Carlo table)."""
    from scipy.optimize import brentq
    from scipy.stats import norm

    def f(a: float) -> float:
        return 2 * (norm.cdf(a) - a * norm.pdf(a)) + alpha - 2

    return brentq(f, 0.01, 10.0)


def cusum_boundary(k: np.ndarray, n_hist: int, a: float) -> np.ndarray:
    """``strucchange``'s closed-form OLS-CUSUM boundary:
    ``sqrt(x*(x-1)*(a^2 + log(x/(x-1))))`` where ``x = (n_hist + k)/n_hist``
    is the *total* elapsed sample size (history plus however far into the
    monitoring period, in the boundary-crossing process's own time axis --
    the process starts running at the end of the history period, not at
    ``x=0``) relative to the history length, evaluated at each
    monitoring-period position ``k`` (1-indexed observation count into the
    monitoring period). ``x > 1`` always holds here since ``k >= 1``, so
    ``log(x/(x-1))`` stays finite and positive."""
    x = (n_hist + k) / n_hist
    return np.sqrt(x * (x - 1) * (a**2 + np.log(x / (x - 1))))


def _design_matrix(days_since_start: np.ndarray, order: int) -> np.ndarray:
    """The harmonic regression design matrix: intercept, trend, then
    ``order`` (cos, sin) pairs at the annual period -- see module
    docstring's model formula."""
    n = len(days_since_start)
    trend = days_since_start.astype(np.float64)
    columns = [np.ones(n), trend]
    for k in range(1, order + 1):
        phase = 2 * np.pi * k * days_since_start / ANNUAL_PERIOD_DAYS
        columns.append(np.cos(phase))
        columns.append(np.sin(phase))
    return np.stack(columns, axis=1)


@dataclass(frozen=True)
class BfastMonitorResult:
    has_break: bool
    break_index: Optional[int]  # index into the monitoring-period arrays
    break_date_ordinal: Optional[int]
    magnitude: Optional[float]
    n_history_obs: int
    n_monitoring_obs: int


def bfastmonitor_pixel(
    dates_ordinal: np.ndarray,
    values: np.ndarray,
    monitor_start_ordinal: int,
    order: int = 3,
    alpha: float = 0.05,
    min_history_obs: int = 12,
    min_monitoring_obs: int = 3,
) -> Optional[BfastMonitorResult]:
    """Run BFAST-Monitor on one pixel's already-computed scalar index time
    series (e.g. NBR from :func:`nbr_from_bands`), split at
    ``monitor_start_ordinal`` into history (before) and monitoring
    (on/after) periods.

    Returns ``None`` if either period has too few observations to fit/test
    meaningfully (mirrors :func:`change.ccdc.run_ccdc_pixel`'s same
    "insufficient data" contract -- not every pixel/period split has enough
    signal, and that must be distinguishable from "fit and found no break").
    """
    order_mask = np.argsort(dates_ordinal)
    dates_ordinal = dates_ordinal[order_mask]
    values = values[order_mask]

    hist_mask = dates_ordinal < monitor_start_ordinal
    mon_mask = ~hist_mask

    n_hist = int(hist_mask.sum())
    n_mon = int(mon_mask.sum())
    if n_hist < min_history_obs or n_mon < min_monitoring_obs:
        return None

    hist_dates = dates_ordinal[hist_mask]
    hist_values = values[hist_mask]
    valid_hist = np.isfinite(hist_values)
    if valid_hist.sum() < min_history_obs:
        return None
    hist_dates = hist_dates[valid_hist]
    hist_values = hist_values[valid_hist]
    n_hist = len(hist_dates)

    history_start = hist_dates[0]
    hist_days = (hist_dates - history_start).astype(np.float64)
    design = _design_matrix(hist_days, order)

    coefs, _residuals, rank, _sv = np.linalg.lstsq(design, hist_values, rcond=None)
    if rank < design.shape[1]:
        return None  # degenerate fit (e.g. too little seasonal spread to identify harmonics)

    fitted_hist = design @ coefs
    residuals_hist = hist_values - fitted_hist
    sigma = float(np.std(residuals_hist, ddof=design.shape[1]))
    if not np.isfinite(sigma) or sigma <= 0:
        return None

    mon_dates = dates_ordinal[mon_mask]
    mon_values = values[mon_mask]
    valid_mon = np.isfinite(mon_values)
    mon_dates = mon_dates[valid_mon]
    mon_values = mon_values[valid_mon]
    if len(mon_dates) < min_monitoring_obs:
        return None

    mon_days = (mon_dates - history_start).astype(np.float64)
    mon_design = _design_matrix(mon_days, order)
    fitted_mon = mon_design @ coefs
    residuals_mon = mon_values - fitted_mon

    cusum = np.cumsum(residuals_mon) / (sigma * np.sqrt(n_hist))
    k = np.arange(1, len(residuals_mon) + 1, dtype=np.float64)
    a = cusum_boundary_constant(alpha)
    boundary = cusum_boundary(k, n_hist, a)

    exceeds = np.abs(cusum) > boundary
    if not exceeds.any():
        return BfastMonitorResult(
            has_break=False,
            break_index=None,
            break_date_ordinal=None,
            magnitude=None,
            n_history_obs=n_hist,
            n_monitoring_obs=len(mon_dates),
        )

    break_index = int(np.argmax(exceeds))
    magnitude = float(np.median(residuals_mon[break_index:]))
    return BfastMonitorResult(
        has_break=True,
        break_index=break_index,
        break_date_ordinal=int(mon_dates[break_index]),
        magnitude=magnitude,
        n_history_obs=n_hist,
        n_monitoring_obs=len(mon_dates),
    )


def run_bfast_tile_nbr(
    ts: TileTimeSeries,
    monitor_start_ordinal: int,
    order: int = 3,
    alpha: float = 0.05,
    min_history_obs: int = 12,
    min_monitoring_obs: int = 3,
    n_workers: int = 1,
) -> list[tuple[int, int, BfastMonitorResult]]:
    """Run BFAST-Monitor's NBR fire cross-check independently on every pixel
    of one tile's aligned time series -- embarrassingly parallel across
    pixels, same as :func:`change.ccdc.run_ccdc_tile` (CPU multiprocessing,
    never GPU: this is plain NumPy/OLS, no CUDA path exists or would help
    here)."""
    _n_obs, _n_bands, height, width = ts.bands.shape
    nbr = nbr_from_bands(np.moveaxis(ts.bands, 1, -1))  # (n_obs, H, W)

    tasks = []
    for row in range(height):
        for col in range(width):
            values = nbr[:, row, col]
            if np.isnan(values).all():
                continue
            qa_values = ts.qas[:, row, col]
            masked_values = np.where(qa_values == _QA_CLEAR, values, np.nan)
            tasks.append((row, col, ts.dates_ordinal, masked_values))

    args = [
        (row, col, dates, values, monitor_start_ordinal, order, alpha, min_history_obs, min_monitoring_obs)
        for row, col, dates, values in tasks
    ]

    if n_workers <= 1:
        raw = [_run_bfast_worker(a) for a in args]
    else:
        from multiprocessing import Pool

        with Pool(processes=n_workers) as pool:
            raw = pool.map(_run_bfast_worker, args)

    return [item for item in raw if item is not None]


def _run_bfast_worker(args: tuple) -> Optional[tuple]:
    """Module-level (picklable) worker, mirroring
    :func:`change.ccdc._run_pixel_worker`'s same Windows-multiprocessing
    requirement."""
    row, col, dates, values, monitor_start_ordinal, order, alpha, min_history_obs, min_monitoring_obs = args
    result = bfastmonitor_pixel(dates, values, monitor_start_ordinal, order, alpha, min_history_obs, min_monitoring_obs)
    if result is None:
        return None
    return row, col, result


def bfast_output_path(output_root: str | Path, tile_id: str) -> Path:
    """``<output_root>/<tile_id>/bfast.npz``."""
    return Path(output_root) / tile_id / "bfast.npz"


def write_bfast_result(
    path: str | Path,
    results: list[tuple[int, int, BfastMonitorResult]],
    transform: tuple[float, ...],
    crs_wkt: str,
    resolution_m: float,
    shape: tuple[int, int],
) -> Path:
    """One row per pixel that could be tested (whether or not a break was
    found -- ``has_break`` distinguishes the two), not one row per detected
    break, so a consumer can tell "tested, no break" apart from "never
    reached this pixel at all" (absent from the table)."""
    rows = np.array([r for r, _c, _res in results], dtype=np.int32)
    cols = np.array([c for _r, c, _res in results], dtype=np.int32)
    has_break = np.array([res.has_break for _r, _c, res in results], dtype=bool)
    break_date = np.array([res.break_date_ordinal or 0 for _r, _c, res in results], dtype=np.int32)
    magnitude = np.array([res.magnitude if res.magnitude is not None else np.nan for _r, _c, res in results], dtype=np.float32)
    n_history_obs = np.array([res.n_history_obs for _r, _c, res in results], dtype=np.int16)
    n_monitoring_obs = np.array([res.n_monitoring_obs for _r, _c, res in results], dtype=np.int16)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        row=rows,
        col=cols,
        has_break=has_break,
        break_date=break_date,
        magnitude=magnitude,
        n_history_obs=n_history_obs,
        n_monitoring_obs=n_monitoring_obs,
        transform=np.array(list(transform)[:6], dtype=np.float64),
        crs_wkt=np.array(crs_wkt),
        resolution_m=np.array(resolution_m, dtype=np.float64),
        shape=np.array(shape, dtype=np.int32),
    )
    return out


def read_bfast_result(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as data:
        return {
            "row": np.array(data["row"]),
            "col": np.array(data["col"]),
            "has_break": np.array(data["has_break"]),
            "break_date": np.array(data["break_date"]),
            "magnitude": np.array(data["magnitude"]),
            "n_history_obs": np.array(data["n_history_obs"]),
            "n_monitoring_obs": np.array(data["n_monitoring_obs"]),
            "transform": tuple(float(v) for v in data["transform"]),
            "crs_wkt": str(data["crs_wkt"]),
            "shape": tuple(int(v) for v in data["shape"]),
            "resolution_m": float(data["resolution_m"]),
        }


def build_tile_bfast(
    tile_dir: str | Path,
    inference_root: str | Path,
    output_root: str | Path,
    tile_id: str,
    class_config,
    monitor_start_ordinal: int,
    order: int = 3,
    alpha: float = 0.05,
    min_history_obs: int = 12,
    min_monitoring_obs: int = 3,
    n_workers: int = 1,
    overwrite: bool = False,
) -> Optional[Path]:
    """Build one tile's full BFAST-Monitor NBR result: reuse
    :func:`change.ccdc.build_tile_time_series` for the same aligned-grid time
    series CCDC uses, then fit per pixel. Returns the written path, or
    ``None`` if already built and ``overwrite`` is false."""
    from landscape_change_detection_pipeline.change.ccdc import build_tile_time_series

    out_path = bfast_output_path(output_root, tile_id)
    if out_path.is_file() and not overwrite:
        return None

    ts = build_tile_time_series(tile_dir, inference_root, tile_id, class_config)
    results = run_bfast_tile_nbr(
        ts, monitor_start_ordinal, order=order, alpha=alpha,
        min_history_obs=min_history_obs, min_monitoring_obs=min_monitoring_obs, n_workers=n_workers,
    )
    write_bfast_result(out_path, results, ts.transform, ts.crs_wkt, ts.resolution_m, ts.bands.shape[2:])
    return out_path
