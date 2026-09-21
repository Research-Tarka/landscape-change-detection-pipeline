"""Pansharpening: Gram-Schmidt Adaptive (analytic stack) and Brovey (RGB visuals).

Purpose
-------
Landsat 7/8/9's blue/green/red/NIR bands are natively 30 m; each sensor also
carries a 15 m panchromatic band. Two different pansharpening methods are
used for two different purposes:

- **Gram-Schmidt Adaptive (GSA)** for the analytic TOA stack that later
  pipeline stages (indices, the U-Net) actually train on -- materially lower
  spectral distortion than Brovey for Landsat, at the cost of being slower
  (a per-band adaptive-weight least-squares fit against a synthetic low-res
  pan).
- **Brovey** for the RGB visualization composites only -- fast, and visual
  fidelity (not spectral fidelity for downstream indices) is all that
  matters there.

Neither product is a general-purpose pansharpening library: both are
implemented directly against numpy arrays already resampled onto the same
grid (see :mod:`.gee_fetch` for how the 30 m bands and 15 m pan band get
there), since a full plugin (e.g. GDAL's pansharpen driver) is more machinery
than five bands per scene need.

Both bands and the pan array must already be on the pan band's grid (15 m)
before calling either function here -- upsample the coarse bands first with
:func:`landscape_change_detection_pipeline.scenes.gee_fetch.resample_band_to_grid` (plain
bilinear, purely a grid-alignment step; the pansharpening itself is what adds
real detail on top of that).
"""

from __future__ import annotations

import warnings

import numpy as np


def _nan_safe(*arrays: np.ndarray) -> np.ndarray:
    """Combined finite mask across all inputs (pixels valid in every band)."""
    mask = np.ones(arrays[0].shape, dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def brovey_pansharpen(bands: dict[str, np.ndarray], pan: np.ndarray) -> dict[str, np.ndarray]:
    """Brovey transform: each band scaled by ``pan / mean(bands)``.

    Fast, simple, and known to shift color balance under bright/dark
    features -- acceptable for a human-facing RGB composite, not used for the
    analytic stack. ``bands`` and ``pan`` must already share one grid/shape.
    """
    stacked = np.stack(list(bands.values()), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        band_mean = np.nanmean(stacked, axis=0)

    out: dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(band_mean > 0, pan / band_mean, np.nan)
    for name, arr in bands.items():
        sharpened = arr * ratio
        mask = _nan_safe(arr, pan, band_mean)
        out[name] = np.where(mask, sharpened, np.nan).astype(np.float32)
    return out


def gram_schmidt_adaptive_pansharpen(
    bands: dict[str, np.ndarray], pan: np.ndarray
) -> dict[str, np.ndarray]:
    """Gram-Schmidt Adaptive (GSA) pansharpening.

    Standard GSA procedure (Aiazzi et al. 2007):

    1. Simulate a low-resolution pan (``I``) as an *adaptive* weighted
       combination of the input bands -- weights fit by least squares against
       the real pan band downsampled to the bands' native grid, rather than a
       plain average (the "adaptive" step; a plain average is the older,
       non-adaptive Gram-Schmidt method and gives worse spectral fidelity).
    2. Gram-Schmidt-orthogonalize the band stack against ``I`` (``I`` first).
    3. Replace ``I`` with the real, full-resolution pan (histogram-matched to
       ``I``'s mean/std so injected detail doesn't shift the radiometry).
    4. Invert the Gram-Schmidt transform back to band space.

    All bands and ``pan`` must already be on the same (pan) grid; the
    "downsampled to the bands' native grid" step in (1) is simulated here by
    blurring the real pan with a box filter matched to the resolution ratio,
    not by re-fetching a coarser array.
    """
    names = list(bands.keys())
    stacked = np.stack([bands[n] for n in names], axis=0).astype(np.float64)
    pan64 = pan.astype(np.float64)
    n_bands, h, w = stacked.shape

    mask = _nan_safe(*stacked, pan64) if n_bands else _nan_safe(pan64)
    stacked_filled = np.where(np.isnan(stacked), 0.0, stacked)
    pan_filled = np.where(np.isnan(pan64), 0.0, pan64)

    # Step 1: adaptive weights for the synthetic low-res pan I, fit by
    # ordinary least squares (I = sum_k w_k * band_k) against a blurred
    # version of the real pan standing in for "pan resampled to band res".
    blurred_pan = _box_blur(pan_filled, size=3)
    valid = mask.ravel()
    X = stacked_filled.reshape(n_bands, -1)[:, valid].T  # (n_valid, n_bands)
    y = blurred_pan.ravel()[valid]
    if X.shape[0] < n_bands + 1:
        weights = np.full(n_bands, 1.0 / n_bands)
    else:
        weights, *_ = np.linalg.lstsq(X, y, rcond=None)
        if not np.all(np.isfinite(weights)) or np.allclose(weights, 0):
            weights = np.full(n_bands, 1.0 / n_bands)

    synthetic_i = np.tensordot(weights, stacked_filled, axes=(0, 0))

    # Step 2: Gram-Schmidt orthogonalization, I first.
    gs_basis = [synthetic_i]
    coeffs = np.zeros((n_bands, n_bands + 1))
    i_mean = synthetic_i[mask].mean() if mask.any() else 0.0
    i_var = synthetic_i[mask].var() if mask.any() else 1.0
    i_var = i_var if i_var > 1e-12 else 1e-12

    for k in range(n_bands):
        band_k = stacked_filled[k]
        proj = band_k.copy()
        for j, basis_j in enumerate(gs_basis):
            basis_var = basis_j[mask].var() if mask.any() else 1.0
            basis_var = basis_var if basis_var > 1e-12 else 1e-12
            cov = np.mean((band_k[mask] - band_k[mask].mean()) * (basis_j[mask] - basis_j[mask].mean())) if mask.any() else 0.0
            coeff = cov / basis_var
            coeffs[k, j] = coeff
            proj = proj - coeff * basis_j
        gs_basis.append(proj)

    # Step 3: replace I with the real pan, histogram-matched to I's stats.
    pan_valid = pan_filled[mask] if mask.any() else pan_filled.ravel()
    pan_mean, pan_std = pan_valid.mean(), pan_valid.std()
    pan_std = pan_std if pan_std > 1e-12 else 1e-12
    matched_pan = (pan_filled - pan_mean) / pan_std * np.sqrt(i_var) + i_mean

    # Step 4: invert -- add back each band's projection coefficient onto the
    # (now pan-substituted) I component, plus its own detail component.
    detail = matched_pan - synthetic_i
    out: dict[str, np.ndarray] = {}
    for k, name in enumerate(names):
        sharpened = stacked_filled[k] + coeffs[k, 0] * detail
        out[name] = np.where(mask, sharpened, np.nan).astype(np.float32)
    return out


def _box_blur(arr: np.ndarray, size: int = 3) -> np.ndarray:
    """Simple box blur via cumulative sums (no scipy dependency)."""
    if size <= 1:
        return arr
    pad = size // 2
    padded = np.pad(arr, pad, mode="edge")
    cumsum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    cumsum = np.pad(cumsum, ((1, 0), (1, 0)), mode="constant")
    h, w = arr.shape
    total = (
        cumsum[size:, size:]
        - cumsum[:h, size:]
        - cumsum[size:, :w]
        + cumsum[:h, :w]
    )
    return total / (size * size)
