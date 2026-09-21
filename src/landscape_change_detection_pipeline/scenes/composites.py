"""RGB visualization composites: RGB Raw and RGB Shadow.

Purpose
-------
Two 8-bit visual composites per scene, for human QA/annotation only (never
fed to the model -- the analytic TOA stack in :mod:`.zarr_store` is what
training reads):

- **RGB Raw**: percentile-stretched natural color.
- **RGB Shadow**: inverse-hyperbolic-sine (asinh) compression + gamma, which
  pulls up shadowed/dark terrain detail that a linear or percentile stretch
  would crush to near-black -- useful for a mountainous, heavily forested AOI
  where valley shadow is common.

Both use fast Brovey pansharpening for Landsat 7/8/9 (see
:mod:`.pansharpen`'s module docstring for why Brovey is fine here even though
the analytic stack uses GSA).
"""

from __future__ import annotations

import numpy as np

#: asinh compression steepness and gamma, as tunable starting parameters
#: (not derived from a specific calibration).
DEFAULT_ASINH_K = 8.0
DEFAULT_GAMMA = 1.0 / 2.2


def percentile_stretch(
    rgb: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0
) -> np.ndarray:
    """Stretch a ``(3, H, W)`` float array to uint8 using per-band percentiles."""
    out = np.zeros(rgb.shape, dtype=np.uint8)
    for i in range(rgb.shape[0]):
        band = rgb[i]
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            continue
        lo, hi = np.percentile(finite, [low_pct, high_pct])
        if hi <= lo:
            hi = lo + 1e-6
        stretched = np.clip((band - lo) / (hi - lo), 0, 1)
        stretched = np.where(np.isfinite(band), stretched, 0.0)
        out[i] = (stretched * 255).astype(np.uint8)
    return out


def asinh_shadow_stretch(
    rgb: np.ndarray, k: float = DEFAULT_ASINH_K, gamma: float = DEFAULT_GAMMA
) -> np.ndarray:
    """Inverse-hyperbolic-sine compression + gamma, to recover shadow detail.

    ``asinh(k * x) / asinh(k)`` maps [0, 1] TOA reflectance to [0, 1] with a
    much flatter response near zero than a linear stretch (i.e. dark pixels
    spread out over more of the output range), then gamma further brightens
    midtones. A final per-scene min/max stretch (computed jointly across all
    three bands, to preserve color balance) spreads the compressed values
    across the full 8-bit range before quantization -- without it, a scene
    whose brightest pixels are still moderately dark (typical for this AOI's
    forest canopy) would compress into a narrow, visually flat mid-gray band
    instead of showing the shadow detail the asinh step just recovered.
    """
    normalizer = np.arcsinh(k)
    finite_mask = np.isfinite(rgb)
    clipped = np.clip(np.where(finite_mask, rgb, 0.0), 0, None)
    compressed = np.arcsinh(k * clipped) / normalizer
    gamma_applied = np.clip(compressed, 0, 1) ** gamma

    valid = gamma_applied[finite_mask]
    lo = float(valid.min()) if valid.size else 0.0
    hi = float(valid.max()) if valid.size else 1.0
    if hi <= lo:
        # Degenerate (uniform-value) scene: nothing to stretch against, so
        # leave the compressed value as-is rather than collapsing to black.
        stretched = gamma_applied
    else:
        stretched = np.clip((gamma_applied - lo) / (hi - lo), 0, 1)

    out = np.where(finite_mask, stretched * 255, 0).astype(np.uint8)
    return out


def build_rgb_composites(
    rgb_bands: dict[str, np.ndarray],
    asinh_k: float = DEFAULT_ASINH_K,
    gamma: float = DEFAULT_GAMMA,
) -> dict[str, np.ndarray]:
    """Build both RGB composites from three already-pansharpened/native bands.

    ``rgb_bands`` is ``{"red": arr, "green": arr, "blue": arr}`` at the
    sensor's working resolution (already pansharpened for Landsat 7/8/9,
    native for Landsat 5 and Sentinel-2). Returns ``{"rgb_raw": (3,H,W)
    uint8, "rgb_shadow": (3,H,W) uint8}``.
    """
    stacked = np.stack([rgb_bands["red"], rgb_bands["green"], rgb_bands["blue"]], axis=0)
    return {
        "rgb_raw": percentile_stretch(stacked),
        "rgb_shadow": asinh_shadow_stretch(stacked, k=asinh_k, gamma=gamma),
    }


def save_rgb_png(rgb_uint8: np.ndarray, path: str) -> None:
    """Save a ``(3, H, W)`` uint8 array as a PNG, for quick visual QA."""
    from PIL import Image

    array_hwc = np.transpose(rgb_uint8, (1, 2, 0))
    Image.fromarray(array_hwc, mode="RGB").save(path)
