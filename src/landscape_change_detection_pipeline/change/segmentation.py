"""Per-pixel temporal segmentation: break dates, trends and seasonality.

Purpose
-------
Splits every pixel's multi-decade time series of spectral features into
*segments* -- stretches where one stable model (level + linear trend +
annual harmonics) explains the observations -- separated by *breaks*, the
moments the observations stop agreeing with that model. Each segment records
its own start/end, the break that closed it, the size and sign of that break
per feature, the fitted level/trend/seasonal amplitude/peak day per feature,
and the dominant land-cover class of its observations. This is the single
change-detection engine every later stage reads: disturbance timing and type,
severity and regrowth, land-use signatures and state transitions all derive
from these segments.

Method (a compact re-implementation of the COLD/CCDC idea)
-----------------------------------------------------------
For one pixel, over its *clear* observations only (see below):

1. **Model**: ``y(t) = c0 + c1*t + sum_k a_k*cos(k*w*doy) + b_k*sin(k*w*doy)
   + sum_s d_s*sensor_s(t)`` fitted per feature by (very lightly
   ridge-regularised) least squares. ``t`` is years since the segment's
   first observation; the harmonics use the calendar day of year, so
   seasonality is modelled explicitly and the annual cycle never reads as
   change. ``sensor_s(t)`` is a one-hot indicator for each non-reference
   sensor contributing observations to this pixel's whole series (one fixed
   effect per sensor, one sensor held out as the implicit reference to avoid
   collinearity with the intercept) -- this absorbs each sensor's own
   systematic radiometric offset (band-pass differences between Landsat
   generations and Sentinel-2, particularly pronounced on TOA reflectance
   over sloped terrain where no published cross-sensor correction applies,
   see ``docs/decisions/cross_sensor_harmonization.md``) directly inside the
   fitted model, instead of relying on a pre-correction of the input
   reflectance: a pixel's sensor mix simply becomes another regressor
   alongside trend and seasonality, so a change of sensor composition
   (e.g. Sentinel-2 coverage becoming denser in a given year) is no longer
   mistaken for a change in the land surface itself. The number of harmonics
   grows with the segment's observation count (1 harmonic below 18
   observations, 2 from 18, 3 from 24, capped by ``max_harmonics``) so a
   short segment is never over-fitted.
2. **Initialisation**: the first ``init_obs`` clear observations spanning at
   least ``init_min_span_days`` form the initial window. Isolated outliers
   (thin cloud, smoke -- what a per-scene classifier misses) are removed
   iteratively using a robust (MAD) residual scale. The window is accepted
   only if it is stable: its trend over the window, and its first/last
   residuals, must all stay within ``stability_threshold`` residual standard
   deviations; otherwise the window slides forward one observation.
3. **Monitoring**: each next observation is compared to the model's
   prediction. The per-feature residuals are normalised by the model's own
   RMSE (floored at ``min_rmse``) and summed in squares -- under "no change"
   this is chi-square distributed with ``n_features`` degrees of freedom. An
   observation beyond the ``p_change`` quantile is a candidate; if
   ``conse`` consecutive observations all exceed it, the first of them dates
   the break. If not, the candidate is treated as an outlier and dropped.
   Observations that pass update the model.
4. After a break the next segment is initialised from the first breaking
   observation. The final segment runs to the last observation with no break.

Only *clear* observations are used: the caller masks whatever it considers
unusable for a land-surface model (cloud, shadow, snow, ice, open water, no
data) before this module sees the data. Pixels with fewer than ``init_obs``
clear observations get no segments.

Class per segment
-----------------
Each observation carries the land-cover class its scene classification gave
it; a segment reports the most frequent class over its observations (and that
class's share), and each break reports the most frequent class over the
``conse`` observations that confirmed it. This yields clean per-segment state
(and state transitions across a break) instead of the scene-to-scene flicker
of raw per-observation classes.

Output layout
-------------
:func:`segment_block` fills a dense ``(n_pixels, max_segments, n_cols)``
float32 array (integer-valued columns -- dates are proleptic-Gregorian
ordinals, exactly representable in float32) whose column layout is given by
:func:`column_layout`. :func:`compact_block` turns it into ragged, one-row-
per-segment arrays.

Speed
-----
The per-pixel loop is compiled with numba and parallelised over pixels; the
model is maintained through running normal-equation sums so each new
observation costs a small fixed number of floating-point operations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numba
import numpy as np
from numba import njit, prange

MAX_SENSORS = 5  # L5, L7, L8, L9, S2 -- one is held out as the implicit reference
# 2 (level+trend) + 2*3 harmonics + one dummy per non-reference sensor + one
# slope-interaction column per non-reference sensor (slope_pixel * dummy_s,
# used only when SegmentationParams.use_slope_interaction is set -- see
# module docstring; the columns are simply left at zero otherwise).
P_MAX = 8 + 2 * (MAX_SENSORS - 1)
MAX_CLASSES = 32
N_FIXED_COLS = 8
_HARMONIC_THRESHOLDS = (18, 24)  # observations at which the 2nd / 3rd harmonic is added
_DAYS_PER_YEAR = 365.25
_RIDGE = 1e-8


@dataclass(frozen=True)
class SegmentationParams:
    """Detector parameters (see the module docstring for their meaning)."""

    chi_threshold: float
    outlier_threshold: float
    min_rmse: float
    stability_threshold: float
    init_obs: int
    init_min_span_days: float
    conse: int
    max_harmonics: int
    max_segments: int
    use_slope_interaction: bool = False
    min_magnitude: float = 0.0
    magnitude_features: tuple = ()
    min_break_span_days: float = 0.0
    refit_every: int = 1
    min_magnitude_same_class: float = 0.0
    persist_days: float = 0.0
    persist_obs: int = 4
    min_obs_break_year: int = 0


def column_layout(n_features: int) -> dict[str, int | slice]:
    """Column positions inside a segment record: fixed scalar columns, then
    one block of ``n_features`` columns per per-feature statistic."""
    f = n_features
    return {
        "t_start": 0,
        "t_end": 1,
        "t_break": 2,
        "n_obs": 3,
        "seg_class": 4,
        "seg_class_frac": 5,
        "break_class": 6,
        "break_class_frac": 7,
        "magnitude": slice(8, 8 + f),
        "level": slice(8 + f, 8 + 2 * f),
        "slope": slice(8 + 2 * f, 8 + 3 * f),
        "amplitude": slice(8 + 3 * f, 8 + 4 * f),
        "peak_doy": slice(8 + 4 * f, 8 + 5 * f),
        "rmse": slice(8 + 5 * f, 8 + 6 * f),
        "resid_magnitude": slice(8 + 6 * f, 8 + 7 * f),
        "persist_flag": 8 + 7 * f,
    }


def n_columns(n_features: int) -> int:
    return N_FIXED_COLS + 7 * n_features + 1


def harmonic_table(dates_ordinal: np.ndarray) -> np.ndarray:
    """``(n_obs, 6)`` float64 of ``cos/sin`` of the first three annual
    harmonics for each observation's calendar day of year."""
    from datetime import date

    doy = np.array([date.fromordinal(int(o)).timetuple().tm_yday for o in dates_ordinal], dtype=np.float64)
    phase = 2.0 * np.pi * doy / _DAYS_PER_YEAR
    table = np.empty((len(doy), 6), dtype=np.float64)
    for k in range(3):
        table[:, 2 * k] = np.cos((k + 1) * phase)
        table[:, 2 * k + 1] = np.sin((k + 1) * phase)
    return table


# ---------------------------------------------------------------------------
# numba kernels
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def _chol_solve(A, Bm, P, F, coef, L):
    """Solve ``(A[:P,:P] + ridge) x = Bm[:P]`` for every feature by Cholesky.
    ``A`` holds the lower triangle only. Returns False if not positive
    definite."""
    for i in range(P):
        for j in range(i + 1):
            s = A[i, j]
            if i == j and i >= 1:
                s += _RIDGE
            for k in range(j):
                s -= L[i, k] * L[j, k]
            if i == j:
                if s <= 1e-12:
                    return False
                L[i, i] = math.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    for f in range(F):
        for i in range(P):
            s = Bm[i, f]
            for k in range(i):
                s -= L[i, k] * coef[k, f]
            coef[i, f] = s / L[i, i]
        for i in range(P - 1, -1, -1):
            s = coef[i, f]
            for k in range(i + 1, P):
                s -= L[k, i] * coef[k, f]
            coef[i, f] = s / L[i, i]
    return True


@njit(cache=True, fastmath=True)
def _fit(A, Bm, yy, n, F, kmax, thr1, thr2, min_rmse, coef, rmse, L, n_sensor_dummies, use_slope):
    """Fit the current sums; returns the harmonic count used (0 = failed).
    ``n_sensor_dummies`` (one less than the number of distinct sensors seen
    so far by this pixel) extends the model with that many sensor fixed-
    effect columns (doubled to include the slope-interaction column per
    sensor when ``use_slope`` is set), always placed right after the
    harmonic columns -- see module docstring."""
    k = 1
    if kmax >= 2 and n >= thr1:
        k = 2
    if kmax >= 3 and n >= thr2:
        k = 3
    n_sensor_cols = n_sensor_dummies * (2 if use_slope else 1)
    P = 2 + 2 * k + n_sensor_cols
    if n < P + 1:
        return 0
    if not _chol_solve(A, Bm, P, F, coef, L):
        return 0
    for a in range(P, P_MAX):
        for f in range(F):
            coef[a, f] = 0.0
    dof = n - P
    for f in range(F):
        cb = 0.0
        cac = 0.0
        for a in range(P):
            cb += coef[a, f] * Bm[a, f]
            for b in range(P):
                if a >= b:
                    aa = A[a, b]
                else:
                    aa = A[b, a]
                cac += coef[a, f] * aa * coef[b, f]
        sse = yy[f] - 2.0 * cb + cac
        if sse < 0.0:
            sse = 0.0
        r = math.sqrt(sse / dof)
        if r < min_rmse:
            r = min_rmse
        rmse[f] = r
    return k


@njit(cache=True)
def _basis(x, t_years, H, i, sensor_dummy, n_sensor_dummies, slope_pixel, use_slope):
    """Fills the model's design-matrix row for observation ``i``: level,
    trend, 6 harmonic columns, then, per non-reference sensor (in
    first-seen order), one one-hot column and -- only when ``use_slope`` is
    set -- immediately followed by one ``slope_pixel * indicator`` column
    (the sensor offset scaled by this pixel's own terrain slope, see module
    docstring). Interleaving dummy/interaction pairs per sensor, rather than
    two separate blocks, keeps every column this pixel actually uses
    contiguous from index 8 on -- required for ``_fit``'s ``P`` contiguous
    solve."""
    x[0] = 1.0
    x[1] = t_years
    for a in range(6):
        x[2 + a] = H[i, a]
    stride = 2 if use_slope else 1
    for a in range(MAX_SENSORS - 1):
        dummy_val = 1.0 if a < n_sensor_dummies and a == sensor_dummy else 0.0
        base = 8 + a * stride
        x[base] = dummy_val
        if use_slope:
            x[base + 1] = dummy_val * slope_pixel


@njit(cache=True)
def _add_obs(A, Bm, yy, x, Y, i, F, pa):
    """Accumulate observation ``i`` into the normal-equation sums over the
    ``pa`` columns the pixel actually uses (the rest of ``x`` is zero)."""
    for a in range(pa):
        xa = x[a]
        for b in range(a + 1):
            A[a, b] += xa * x[b]
        for f in range(F):
            Bm[a, f] += xa * Y[i, f]
    for f in range(F):
        v = Y[i, f]
        yy[f] += v * v


@njit(cache=True, fastmath=True)
def _residual_stat(coef, x, Y, i, F, rmse, res, L, P, v, pa):
    """Chi-square-like statistic of observation ``i`` against the model;
    fills ``res`` with the raw per-feature residuals.

    The residuals are scaled by the *prediction* standard error,
    ``rmse * sqrt(1 + x' (X'X)^-1 x)``, not the bare RMSE: a model fitted on
    few observations (or extrapolated in time through its trend term) is
    less certain than its in-sample RMSE suggests, and ignoring that
    produces false breaks early in every segment. ``L`` is the Cholesky
    factor of the current normal matrix (``P`` columns); ``v`` is a
    preallocated scratch vector and ``pa`` the number of live columns."""
    for f in range(F):
        pred = 0.0
        for a in range(pa):
            pred += coef[a, f] * x[a]
        res[f] = Y[i, f] - pred

    # leverage h = |L^-1 x|^2 by forward substitution
    h = 0.0
    for a in range(P):
        s = x[a]
        for k in range(a):
            s -= L[a, k] * v[k]
        v[a] = s / L[a, a]
        h += v[a] * v[a]

    stat = 0.0
    for f in range(F):
        z = res[f] / rmse[f]
        stat += z * z
    return stat / (1.0 + h)


@njit(cache=True)
def _median_small(values, n):
    tmp = np.sort(values[:n])
    if n % 2 == 1:
        return tmp[n // 2]
    return 0.5 * (tmp[n // 2 - 1] + tmp[n // 2])


@njit(cache=True)
def _mode(hist):
    best = 0
    total = 0
    for c in range(MAX_CLASSES):
        total += hist[c]
        if hist[c] > hist[best]:
            best = c
    if total == 0:
        return -1.0, 0.0
    return float(best), hist[best] / total


@njit(cache=True)
def _break_magnitude(Y, vi, dead, conf, nconf, seg_first, last_pos, conse, F, magn, preidx, out):
    """``out[f]`` = median of the confirming observations minus median of the
    last ``conse`` live observations of the segment (both raw observed
    values, see the comment at the call site)."""
    n_pre = 0
    p = last_pos
    while n_pre < conse and p >= seg_first:
        if not dead[p]:
            preidx[n_pre] = vi[p]
            n_pre += 1
        p -= 1
    for f in range(F):
        for m in range(nconf):
            magn[m] = Y[vi[conf[m]], f]
        post_med = _median_small(magn, nconf)
        for m in range(n_pre):
            magn[m] = Y[preidx[m], f]
        pre_med = _median_small(magn, n_pre)
        out[f] = post_med - pre_med


@njit(cache=True, fastmath=True)
def _segment_pixel(t, H, Y, valid, cls, sensor_id, slope_pixel, use_slope, chi_thr, out_thr, min_rmse, stab_thr,
                   init_obs, min_span, conse, kmax, max_seg, min_mag, mag_idx, min_break_span, refit_every, qp, F, out):
    """Segment one pixel. Returns ``(n_segments, n_valid, truncated)``.

    ``sensor_id`` is a ``(n_obs,)`` int array, one integer per observation
    identifying which sensor recorded it (0..MAX_SENSORS-1 across the whole
    tile, see module docstring). Each pixel's own sensor fixed effects are
    built independently: the first sensor a pixel's *valid* observations
    ever see becomes that pixel's own reference (dummy -1, no column), and
    every other sensor seen gets its own dummy column, assigned in first-
    seen order -- so a pixel touched by only two sensors uses one dummy
    column, not ``MAX_SENSORS - 1`` unused ones."""
    n_obs = t.shape[0]
    vi = np.empty(n_obs, np.int64)
    nv = 0
    for i in range(n_obs):
        if valid[i]:
            vi[nv] = i
            nv += 1
    if nv < init_obs:
        return 0, nv, 0

    # Map this pixel's sensors to dummy columns in first-seen order: the
    # first-seen sensor is this pixel's reference (dummy -1, no column),
    # every later new sensor gets the next dummy column (0, 1, 2, ...).
    sensor_dummy_of = np.full(MAX_SENSORS, -2, np.int64)  # -2 = not seen yet
    sdum = np.empty(nv, np.int64)
    next_dummy = 0
    for p in range(nv):
        sid = sensor_id[vi[p]]
        if sensor_dummy_of[sid] == -2:
            sensor_dummy_of[sid] = -1 if next_dummy == 0 else next_dummy - 1
            next_dummy += 1
        sdum[p] = sensor_dummy_of[sid]
    n_sensor_dummies = next_dummy - 1
    n_sensor_cols = n_sensor_dummies * (2 if use_slope else 1)
    pa = 8 + n_sensor_cols  # live model columns (fixed 8 + sensor terms)

    dead = np.zeros(nv, np.bool_)
    A = np.zeros((P_MAX, P_MAX))
    Bm = np.zeros((P_MAX, F))
    yy = np.zeros(F)
    coef = np.zeros((P_MAX, F))
    rmse = np.zeros(F)
    L = np.zeros((P_MAX, P_MAX))
    x = np.zeros(P_MAX)
    res = np.zeros(F)
    resid = np.empty((init_obs + nv, F))  # window residuals (window can exceed init_obs)
    absr = np.empty(nv)
    wpos = np.empty(nv, np.int64)
    conf = np.empty(nv, np.int64)
    preidx = np.empty(conse, np.int64)
    hist = np.zeros(MAX_CLASSES)
    magn = np.empty(nv)
    mag_buf = np.zeros(F)
    cres = np.empty((nv, F))  # model residuals of the confirming run
    pres = np.empty((max(int(qp[2]), 2), F))  # residuals of the persistence-check observations
    rmag_buf = np.zeros(F)
    pflag = 0.0
    vbuf = np.empty(P_MAX)
    sigma_f = np.empty(F)
    nconf = 0

    nseg = 0
    pos = 0
    truncated = 0
    while pos < nv:
        if nseg >= max_seg:
            truncated = 1
            break

        # ---------------- initialisation ----------------
        accepted = False
        start = pos
        t_ref = 0.0
        cnt = 0
        while start < nv:
            cnt = 0
            p = start
            first_t = 0.0
            enough = False
            while p < nv:
                if not dead[p]:
                    if cnt == 0:
                        first_t = t[vi[p]]
                    wpos[cnt] = p
                    cnt += 1
                    if cnt >= init_obs and t[vi[p]] - first_t >= min_span:
                        enough = True
                        break
                p += 1
            if not enough:
                break  # no window possible from here on

            t_ref = t[vi[wpos[0]]]
            for a in range(P_MAX):
                for b in range(P_MAX):
                    A[a, b] = 0.0
                for f in range(F):
                    Bm[a, f] = 0.0
            for f in range(F):
                yy[f] = 0.0
            for w in range(cnt):
                i = vi[wpos[w]]
                _basis(x, (t[i] - t_ref) / _DAYS_PER_YEAR, H, i, sdum[wpos[w]], n_sensor_dummies, slope_pixel, use_slope)
                _add_obs(A, Bm, yy, x, Y, i, F, pa)
            kinit = _fit(A, Bm, yy, cnt, F, kmax, 18, 24, min_rmse, coef, rmse, L, n_sensor_dummies, use_slope)
            if kinit == 0:
                start = wpos[0] + 1
                continue
            pinit = 2 + 2 * kinit + n_sensor_cols

            # residuals of the window against its own fit
            for w in range(cnt):
                i = vi[wpos[w]]
                _basis(x, (t[i] - t_ref) / _DAYS_PER_YEAR, H, i, sdum[wpos[w]], n_sensor_dummies, slope_pixel, use_slope)
                _residual_stat(coef, x, Y, i, F, rmse, res, L, pinit, vbuf, pa)
                for f in range(F):
                    resid[w, f] = res[f]

            # robust outlier screening: drop the single worst obs and retry
            for f in range(F):
                for q in range(cnt):
                    absr[q] = abs(resid[q, f])
                sg = 1.4826 * _median_small(absr, cnt)
                if sg < min_rmse:
                    sg = min_rmse
                sigma_f[f] = sg
            worst = -1
            worst_stat = 0.0
            for w in range(cnt):
                stat = 0.0
                for f in range(F):
                    z = resid[w, f] / sigma_f[f]
                    stat += z * z
                if stat > worst_stat:
                    worst_stat = stat
                    worst = w
            if worst >= 0 and worst_stat > out_thr:
                dead[wpos[worst]] = True
                continue

            # stability of the window
            span_years = (t[vi[wpos[cnt - 1]]] - t_ref) / _DAYS_PER_YEAR
            stable = True
            for f in range(F):
                drift = abs(coef[1, f]) * span_years
                first = abs(resid[0, f])
                last = abs(resid[cnt - 1, f])
                m = max(drift, max(first, last)) / rmse[f]
                if m > stab_thr:
                    stable = False
                    break
            if not stable:
                start = wpos[0] + 1
                continue
            accepted = True
            break

        if not accepted:
            break

        seg_first = wpos[0]
        last_pos = wpos[cnt - 1]
        n_seg = cnt
        kcur = _fit(A, Bm, yy, n_seg, F, kmax, 18, 24, min_rmse, coef, rmse, L, n_sensor_dummies, use_slope)
        if kcur == 0:
            pos = seg_first + 1
            continue

        # ---------------- monitoring ----------------
        j = last_pos + 1
        broke = False
        since_fit = 0
        while j < nv:
            if dead[j]:
                j += 1
                continue
            i = vi[j]
            _basis(x, (t[i] - t_ref) / _DAYS_PER_YEAR, H, i, sdum[j], n_sensor_dummies, slope_pixel, use_slope)
            pcur = 2 + 2 * kcur + n_sensor_cols
            stat = _residual_stat(coef, x, Y, i, F, rmse, res, L, pcur, vbuf, pa)
            if stat > chi_thr:
                # gather the run of consecutive exceedances: at least `conse`
                # of them AND spanning at least `min_break_span` days
                nconf = 1
                conf[0] = j
                for f in range(F):
                    cres[0, f] = res[f]
                t_first = t[i]
                q = j + 1
                while q < nv and (nconf < conse or t[vi[conf[nconf - 1]]] - t_first < min_break_span):
                    if dead[q]:
                        q += 1
                        continue
                    iq = vi[q]
                    _basis(x, (t[iq] - t_ref) / _DAYS_PER_YEAR, H, iq, sdum[q], n_sensor_dummies, slope_pixel, use_slope)
                    if _residual_stat(coef, x, Y, iq, F, rmse, res, L, pcur, vbuf, pa) > chi_thr:
                        conf[nconf] = q
                        for f in range(F):
                            cres[nconf, f] = res[f]
                        nconf += 1
                        q += 1
                    else:
                        break
                if nconf < conse or t[vi[conf[nconf - 1]]] - t_first < min_break_span:
                    dead[j] = True  # unconfirmed exceedance: an outlier, not a break
                    j += 1
                    continue
                # Confirmed statistically; keep it only if the shift is big
                # enough on at least one of the magnitude features. A
                # too-small shift is absorbed into the model instead.
                _break_magnitude(Y, vi, dead, conf, nconf, seg_first, last_pos, conse, F, magn, preidx, mag_buf)
                # (gated on the median *model residual* of the confirming run,
                # which is seasonally safe; the reported magnitude stays the
                # raw before/after difference)
                for f in range(F):
                    for c in range(nconf):
                        magn[c] = cres[c, f]
                    rmag_buf[f] = _median_small(magn, nconf)
                big = min_mag <= 0.0
                for m in range(mag_idx.shape[0]):
                    if abs(rmag_buf[mag_idx[m]]) >= min_mag:
                        big = True

                # enough live observations around the break date to trust it
                # (sparse years make any offset look like a break)
                if big and qp[3] > 0.0:
                    n_around = 0
                    for p in range(nv):
                        if not dead[p] and abs(t[vi[p]] - t_first) <= 365.0:
                            n_around += 1
                    if n_around < qp[3]:
                        big = False

                # a break that leaves the land-cover class unchanged must be
                # larger: those are the ones most often thinning, phenology
                # or sensor drift rather than a change of state
                if big and qp[0] > min_mag:
                    for c in range(MAX_CLASSES):
                        hist[c] = 0.0
                    for p in range(seg_first, last_pos + 1):
                        if not dead[p]:
                            c = cls[vi[p]]
                            if c < MAX_CLASSES:
                                hist[c] += 1.0
                    seg_c, seg_fr = _mode(hist)
                    for c in range(MAX_CLASSES):
                        hist[c] = 0.0
                    for m in range(nconf):
                        c = cls[vi[conf[m]]]
                        if c < MAX_CLASSES:
                            hist[c] += 1.0
                    brk_c, brk_fr = _mode(hist)
                    if seg_c == brk_c:
                        strong = False
                        for m in range(mag_idx.shape[0]):
                            if abs(rmag_buf[mag_idx[m]]) >= qp[0]:
                                strong = True
                        big = strong

                # persistence: a real change is still there in the following
                # seasons; a phenology/sensor artefact is not. Checked on the
                # first observations at least `persist_days` after the break
                # (residuals against the pre-break model). Unverifiable
                # breaks (too close to the end of the series) are kept.
                pflag = 0.0
                if big and qp[1] > 0.0:
                    t_brk = t[vi[conf[0]]]
                    n_chk = int(qp[2])
                    npc = 0
                    q = conf[nconf - 1] + 1
                    while q < nv and npc < n_chk:
                        if (not dead[q]) and t[vi[q]] - t_brk >= qp[1]:
                            iq = vi[q]
                            _basis(x, (t[iq] - t_ref) / _DAYS_PER_YEAR, H, iq, sdum[q], n_sensor_dummies, slope_pixel, use_slope)
                            _residual_stat(coef, x, Y, iq, F, rmse, res, L, pcur, vbuf, pa)
                            for f in range(F):
                                pres[npc, f] = res[f]
                            npc += 1
                        q += 1
                    if npc >= 2:
                        persists = False
                        thr_p = 0.5 * min_mag
                        for m in range(mag_idx.shape[0]):
                            ff = mag_idx[m]
                            for c in range(npc):
                                magn[c] = pres[c, ff]
                            med_p = _median_small(magn, npc)
                            if med_p * rmag_buf[ff] > 0.0 and abs(med_p) >= thr_p:
                                persists = True
                        big = persists
                        pflag = 1.0
                if big:
                    broke = True
                    break
                # the basis row was overwritten by the confirmation scan
                _basis(x, (t[i] - t_ref) / _DAYS_PER_YEAR, H, i, sdum[j], n_sensor_dummies, slope_pixel, use_slope)
            _add_obs(A, Bm, yy, x, Y, i, F, pa)
            n_seg += 1
            last_pos = j
            since_fit += 1
            if since_fit >= refit_every or n_seg < 30:
                k2 = _fit(A, Bm, yy, n_seg, F, kmax, 18, 24, min_rmse, coef, rmse, L, n_sensor_dummies, use_slope)
                if k2 != 0:
                    kcur = k2
                since_fit = 0
            j += 1
        if since_fit > 0:
            k2 = _fit(A, Bm, yy, n_seg, F, kmax, 18, 24, min_rmse, coef, rmse, L, n_sensor_dummies, use_slope)
            if k2 != 0:
                kcur = k2

        # ---------------- close the segment ----------------
        rec = out[nseg]
        rec[0] = t[vi[seg_first]]
        rec[1] = t[vi[last_pos]]
        for c in range(MAX_CLASSES):
            hist[c] = 0.0
        for p in range(seg_first, last_pos + 1):
            if not dead[p]:
                c = cls[vi[p]]
                if c < MAX_CLASSES:
                    hist[c] += 1.0
        seg_cls, seg_frac = _mode(hist)
        rec[3] = n_seg
        rec[4] = seg_cls
        rec[5] = seg_frac
        if broke:
            rec[2] = t[vi[conf[0]]]
            for c in range(MAX_CLASSES):
                hist[c] = 0.0
            for m in range(nconf):
                c = cls[vi[conf[m]]]
                if c < MAX_CLASSES:
                    hist[c] += 1.0
            bcls, bfrac = _mode(hist)
            rec[6] = bcls
            rec[7] = bfrac

            # Magnitude: median of the raw *observed* confirming values minus
            # median of the raw *observed* values ending the pre-break
            # segment -- both are real data, never a model prediction. A
            # model-residual magnitude (observed minus the fitted
            # trend+harmonic model, extrapolated to the break date) was tried
            # first and rejected: on a real pixel, a short/noisy segment can
            # fit a steep trend that is accurate near its own observations
            # but diverges wildly when evaluated at a later date, producing
            # magnitudes far outside any feature's physical range (confirmed
            # on real pilot data: single-digit-to-tens values for indices
            # bounded in [-1, 1]). Comparing two medians of actually-observed
            # values has no such extrapolation and stays within the
            # feature's own observed range by construction.
            for f in range(F):
                rec[8 + f] = mag_buf[f]
                rec[8 + 6 * F + f] = rmag_buf[f]
            rec[8 + 7 * F] = pflag
        else:
            rec[2] = 0.0
            rec[6] = -1.0
            rec[7] = 0.0
            for f in range(F):
                rec[8 + f] = 0.0
        for f in range(F):
            rec[8 + F + f] = coef[0, f]
            rec[8 + 2 * F + f] = coef[1, f]
            a1 = coef[2, f]
            b1 = coef[3, f]
            rec[8 + 3 * F + f] = math.sqrt(a1 * a1 + b1 * b1)
            ph = math.atan2(b1, a1)
            if ph < 0.0:
                ph += 2.0 * math.pi
            rec[8 + 4 * F + f] = ph / (2.0 * math.pi) * _DAYS_PER_YEAR
            rec[8 + 5 * F + f] = rmse[f]
        nseg += 1

        if not broke:
            pos = nv
            break
        pos = conf[0]

    return nseg, nv, truncated


@njit(parallel=True, cache=True)
def _segment_block(t, H, Y, valid, cls, sensor_id, slope, use_slope, chi_thr, out_thr, min_rmse, stab_thr,
                   init_obs, min_span, conse, kmax, max_seg, min_mag, mag_idx, min_break_span, refit_every, qp,
                   out, nseg_out, nvalid_out, trunc_out):
    n_pix = Y.shape[0]
    F = Y.shape[2]
    for k in prange(n_pix):
        ns, nv, tr = _segment_pixel(
            t, H, Y[k], valid[k], cls[k], sensor_id, slope[k], use_slope, chi_thr, out_thr, min_rmse, stab_thr,
            init_obs, min_span, conse, kmax, max_seg, min_mag, mag_idx, min_break_span, refit_every, qp, F, out[k],
        )
        nseg_out[k] = ns
        nvalid_out[k] = nv
        trunc_out[k] = tr


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def segment_block(
    dates_ordinal: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    classes: np.ndarray,
    params: SegmentationParams,
    harmonics: np.ndarray | None = None,
    sensor_ids: np.ndarray | None = None,
    slope: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Segment a block of pixels.

    Parameters
    ----------
    dates_ordinal : ``(n_obs,)`` observation dates as ordinals, ascending.
    values : ``(n_pixels, n_obs, n_features)`` float32 feature values.
    valid : ``(n_pixels, n_obs)`` bool, True where the observation is usable.
    classes : ``(n_pixels, n_obs)`` uint8 land-cover class of each observation.
    harmonics : optional precomputed :func:`harmonic_table` for ``dates_ordinal``.
    sensor_ids : ``(n_obs,)`` int, which sensor recorded each observation
        (0..MAX_SENSORS-1); defaults to all-zero (single implicit sensor,
        i.e. no sensor fixed effect fitted) when not given, so callers that
        don't track sensor identity keep the previous behaviour exactly.
    slope : ``(n_pixels,)`` float, this pixel's own terrain slope (degrees or
        any consistent unit); only used when ``params.use_slope_interaction``
        is set (see module docstring), ignored otherwise -- defaults to
        all-zero.

    Returns a dict with the dense ``segments`` array
    ``(n_pixels, max_segments, n_columns)``, per-pixel ``n_segments`` (uint8),
    ``n_valid`` (int16) and ``truncated`` (uint8, 1 when more segments were
    possible than ``max_segments``).
    """
    n_pix, n_obs, n_feat = values.shape
    t = np.ascontiguousarray(dates_ordinal, dtype=np.float64)
    table = harmonics if harmonics is not None else harmonic_table(dates_ordinal)
    sids = np.zeros(n_obs, dtype=np.int64) if sensor_ids is None else np.ascontiguousarray(sensor_ids, dtype=np.int64)
    slope_arr = np.zeros(n_pix, dtype=np.float64) if slope is None else np.ascontiguousarray(slope, dtype=np.float64)
    try:  # pixels differ wildly in cost (masked ones return at once): small chunks balance the threads
        numba.set_parallel_chunksize(8)
    except AttributeError:  # numba < 0.57
        pass
    out = np.zeros((n_pix, params.max_segments, n_columns(n_feat)), dtype=np.float32)
    nseg = np.zeros(n_pix, dtype=np.uint8)
    nvalid = np.zeros(n_pix, dtype=np.int16)
    trunc = np.zeros(n_pix, dtype=np.uint8)
    _segment_block(
        t, np.ascontiguousarray(table), values, valid, classes, sids, slope_arr, bool(params.use_slope_interaction),
        float(params.chi_threshold), float(params.outlier_threshold), float(params.min_rmse),
        float(params.stability_threshold), int(params.init_obs), float(params.init_min_span_days),
        int(params.conse), int(params.max_harmonics), int(params.max_segments),
        float(params.min_magnitude), np.asarray(params.magnitude_features, dtype=np.int64),
        float(params.min_break_span_days), max(1, int(params.refit_every)),
        np.array([params.min_magnitude_same_class, params.persist_days, params.persist_obs,
                  params.min_obs_break_year], dtype=np.float64),
        out, nseg, nvalid, trunc,
    )
    return {"segments": out, "n_segments": nseg, "n_valid": nvalid, "truncated": trunc}


def compact_block(
    result: dict[str, np.ndarray], width: int, row_offset: int, n_features: int
) -> dict[str, np.ndarray]:
    """Turn a dense block result into one-row-per-segment arrays.

    ``width`` is the raster width (to recover row/col from the flat pixel
    index); ``row_offset`` the block's first raster row.
    """
    seg = result["segments"]
    nseg = result["n_segments"]
    n_pix, max_seg, _ = seg.shape
    mask = np.arange(max_seg)[None, :] < nseg[:, None]
    pix_idx, seg_idx = np.nonzero(mask)
    rec = seg[pix_idx, seg_idx, :]
    layout = column_layout(n_features)
    out: dict[str, np.ndarray] = {
        "row": (row_offset + pix_idx // width).astype(np.int32),
        "col": (pix_idx % width).astype(np.int32),
        "seg_index": seg_idx.astype(np.int16),
    }
    for name, loc in layout.items():
        col = rec[:, loc]
        if name in ("t_start", "t_end", "t_break", "n_obs"):
            out[name] = col.astype(np.int32)
        elif name in ("seg_class", "break_class"):
            out[name] = col.astype(np.int16)
        else:
            out[name] = col.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# temporal aggregation
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _aggregate_kernel(values, valid, cls, member_ptr, members, out_values, out_valid, out_cls):
    n_pix = values.shape[0]
    n_bins = member_ptr.shape[0] - 1
    F = values.shape[2]
    max_m = 1
    for b in range(n_bins):
        max_m = max(max_m, member_ptr[b + 1] - member_ptr[b])
    for k in prange(n_pix):
        tmp = np.empty(max_m, np.float32)
        hist = np.zeros(MAX_CLASSES)
        for b in range(n_bins):
            n = 0
            for m in range(member_ptr[b], member_ptr[b + 1]):
                if valid[k, members[m]]:
                    n += 1
            if n == 0:
                continue
            out_valid[k, b] = True
            for f in range(F):
                c = 0
                for m in range(member_ptr[b], member_ptr[b + 1]):
                    o = members[m]
                    if valid[k, o]:
                        tmp[c] = values[k, o, f]
                        c += 1
                srt = np.sort(tmp[:n])
                out_values[k, b, f] = srt[n // 2] if n % 2 == 1 else 0.5 * (srt[n // 2 - 1] + srt[n // 2])
            for c in range(MAX_CLASSES):
                hist[c] = 0.0
            for m in range(member_ptr[b], member_ptr[b + 1]):
                o = members[m]
                if valid[k, o] and cls[k, o] < MAX_CLASSES:
                    hist[cls[k, o]] += 1.0
            best = 0
            for c in range(MAX_CLASSES):
                if hist[c] > hist[best]:
                    best = c
            out_cls[k, b] = best


def plan_aggregation(
    dates_ordinal: np.ndarray, sensor_ids: np.ndarray, period_days: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group observations into bins of ``period_days`` days, one bin per
    (period, sensor) so a bin never mixes sensors (the per-sensor offset stays
    well defined). Returns ``(bin_dates, bin_sensor_ids, member_ptr, members)``
    with bins sorted by date; a bin's date is the mean of its members' dates."""
    period = (dates_ordinal - dates_ordinal[0]) // int(period_days)
    keys = period * (MAX_SENSORS + 1) + sensor_ids
    uniq, inverse = np.unique(keys, return_inverse=True)
    n_bins = len(uniq)
    bin_dates = np.array([dates_ordinal[inverse == b].mean() for b in range(n_bins)])
    order = np.argsort(bin_dates, kind="stable")
    rank = np.empty(n_bins, dtype=np.int64)
    rank[order] = np.arange(n_bins)
    bin_of_obs = rank[inverse]
    obs_order = np.argsort(bin_of_obs, kind="stable")
    counts = np.bincount(bin_of_obs, minlength=n_bins)
    member_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    bin_sensors = np.empty(n_bins, dtype=np.int64)
    bin_sensors[bin_of_obs] = sensor_ids
    return np.rint(bin_dates[order]).astype(np.int64), bin_sensors, member_ptr, obs_order.astype(np.int64)


def aggregate_block(
    values: np.ndarray,
    valid: np.ndarray,
    classes: np.ndarray,
    member_ptr: np.ndarray,
    members: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pixel median of the usable observations of each bin (see
    :func:`plan_aggregation`); a bin with no usable observation is invalid.
    Fewer, less noisy, less serially-correlated observations: this speeds the
    segmentation up and makes a run of ``conse`` exceedances mean a
    sustained change instead of one dense Sentinel-2 fortnight."""
    n_pix, _, n_feat = values.shape
    n_bins = len(member_ptr) - 1
    out_values = np.zeros((n_pix, n_bins, n_feat), dtype=np.float32)
    out_valid = np.zeros((n_pix, n_bins), dtype=np.bool_)
    out_cls = np.zeros((n_pix, n_bins), dtype=np.uint8)
    _aggregate_kernel(values, valid, classes, member_ptr, members, out_values, out_valid, out_cls)
    return out_values, out_valid, out_cls
