"""RGB visualization composites: 4 selectable views per scene.

Purpose
-------
Up to four 8-bit visual composites per scene, for human QA/annotation only
(never fed to the model -- the analytic TOA stack in :mod:`.zarr_store` is
what training reads). Each is independently config-gated
(``config.py::RgbCompositesConfig``) so only the views the project lead has
actually validated get computed/stored:

- **``rgb_true_color``** (was ``rgb_raw``): percentile-stretched natural
  color (red/green/blue).
- **``rgb_true_color_shadow``** (was ``rgb_shadow``): red/green/blue through
  inverse-hyperbolic-sine (asinh) compression + gamma, which pulls up
  shadowed/dark terrain detail that a linear or percentile stretch would
  crush to near-black -- useful for a mountainous, heavily forested AOI
  where valley shadow is common. Considered the project's "real" vivid
  true-color view.
- **``rgb_natural_color``**: SWIR2/NIR/red as R/G/B -- a standard
  moisture/burn-scar-sensitive interpretation composite, percentile
  stretched (not asinh -- see below).
- **``rgb_color_infrared``**: NIR/red/green as R/G/B -- the standard
  vegetation-vigor false-color composite, percentile stretched.

The two new composites use the plain percentile stretch, not the asinh
shadow-recovery stretch: they're diagnostic/interpretation views where the
conventional remote-sensing display stretch is what analysts expect, not
the true-color-specific shadow-detail recovery this AOI's valley shadow
motivated for the true-color view.

Since the unified-10m-grid revision (``docs/decisions/unified_10m_grid.md``)
every input band is already resampled (bilinear) onto the DEM's exact grid
before this module runs -- there is no pansharpening step anywhere in this
pipeline (this pipeline runs on TOA, see ``docs/decisions/toa_rollback.md``,
and deliberately does not pansharpen even though Landsat's TOA product does
carry a panchromatic band -- every sensor stays on the uniform 10 m grid).
"""

from __future__ import annotations

import numpy as np

#: asinh compression steepness and gamma, as tunable starting parameters
#: (not derived from a specific calibration) -- exposed via
#: ``config.py::RgbCompositesConfig`` so they can be tuned without a code
#: change.
DEFAULT_ASINH_K = 8.0
DEFAULT_GAMMA = 1.0 / 2.2

#: Composite view names, and which band labels (in R, G, B order) each one
#: needs -- the single source of truth both ``build_rgb_composites`` and its
#: callers (``process_scene.py``) key off of.
TRUE_COLOR = "rgb_true_color"
TRUE_COLOR_SHADOW = "rgb_true_color_shadow"
NATURAL_COLOR = "rgb_natural_color"
COLOR_INFRARED = "rgb_color_infrared"

VIEW_BAND_LABELS: dict[str, tuple[str, str, str]] = {
    TRUE_COLOR: ("red", "green", "blue"),
    TRUE_COLOR_SHADOW: ("red", "green", "blue"),
    NATURAL_COLOR: ("swir2", "nir", "red"),
    COLOR_INFRARED: ("nir", "red", "green"),
}


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


def _gray_world_balance(rgb: np.ndarray, finite_mask: np.ndarray) -> np.ndarray:
    """Scale each channel so its own mean matches the three channels' pooled
    mean (gray-world assumption), leaving relative *within-channel*
    structure untouched.

    TOA reflectance carries a real, physical per-band brightness difference
    from atmospheric Rayleigh scattering -- blue is measurably brighter than
    green, which is brighter than red, confirmed live on a real Landsat 8
    TOA scene (blue mean ~0.094 vs. red mean ~0.051 over the same pixels,
    2026-09-25). ``percentile_stretch`` incidentally absorbs this (each
    channel stretched independently to its own percentiles), but this
    function's caller stretches all three channels *jointly* (to preserve
    color balance against a different failure mode -- see its own
    docstring) -- without first correcting this per-band brightness offset,
    that joint stretch preserves the atmospheric blue bias instead of
    neutralizing it, producing a visible blue color cast confirmed live on
    real Landsat scenes. Applied before the asinh compression below (not
    after), matching MaskForge's own ``_toa_gray_balance`` reference
    implementation for the same problem.
    """
    channel_means = np.array(
        [rgb[i][finite_mask[i]].mean() if finite_mask[i].any() else 1.0 for i in range(rgb.shape[0])],
        dtype=np.float64,
    )
    valid_means = channel_means[channel_means > 0]
    target = float(valid_means.mean()) if valid_means.size else 1.0
    scale = np.where(channel_means > 0, target / np.maximum(channel_means, 1e-12), 1.0)
    return rgb * scale.reshape(-1, 1, 1)


def asinh_shadow_stretch(
    rgb: np.ndarray, k: float = DEFAULT_ASINH_K, gamma: float = DEFAULT_GAMMA
) -> np.ndarray:
    """Gray-world balance, then inverse-hyperbolic-sine compression + gamma,
    to recover shadow detail without a color cast.

    ``asinh(k * x) / asinh(k)`` maps [0, 1] TOA reflectance to [0, 1] with a
    much flatter response near zero than a linear stretch (i.e. dark pixels
    spread out over more of the output range), then gamma further brightens
    midtones. A final per-scene min/max stretch (computed jointly across all
    three bands, to preserve color balance) spreads the compressed values
    across the full 8-bit range before quantization -- without it, a scene
    whose brightest pixels are still moderately dark (typical for this AOI's
    forest canopy) would compress into a narrow, visually flat mid-gray band
    instead of showing the shadow detail the asinh step just recovered.

    Gray-world balance runs first (see :func:`_gray_world_balance`'s own
    docstring): confirmed live that without it, TOA's real per-band
    atmospheric-scattering brightness offset (blue > green > red) survives
    the joint stretch as a visible blue color cast on real Landsat scenes --
    this pipeline's own ``rgb_true_color`` (``percentile_stretch``, which
    stretches each channel independently) does not show this artifact,
    only this jointly-stretched shadow-recovery view did, before this fix.
    """
    finite_mask = np.isfinite(rgb)
    clipped = np.clip(np.where(finite_mask, rgb, 0.0), 0, None)
    balanced = _gray_world_balance(clipped, finite_mask)

    normalizer = np.arcsinh(k)
    compressed = np.arcsinh(k * balanced) / normalizer
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
    bands: dict[str, np.ndarray],
    enabled_views: set[str],
    asinh_k: float = DEFAULT_ASINH_K,
    gamma: float = DEFAULT_GAMMA,
) -> dict[str, np.ndarray]:
    """Build every enabled composite view from bands already on the DEM's grid.

    ``bands`` is ``{label: array}`` keyed by band label (red/green/blue and,
    for the natural-color/color-infrared views, nir/swir2 -- whichever
    labels the enabled views actually need), all at the pipeline's uniform
    10 m working resolution. ``enabled_views`` is a subset of
    :data:`VIEW_BAND_LABELS`'s keys (see ``config.py::RgbCompositesConfig``)
    -- a view not in this set is simply not computed, to avoid wasted
    compute/storage for a view nobody has enabled.

    Returns ``{view_name: (3,H,W) uint8}`` for exactly the enabled views.
    """
    out: dict[str, np.ndarray] = {}
    for view, (r_label, g_label, b_label) in VIEW_BAND_LABELS.items():
        if view not in enabled_views:
            continue
        stacked = np.stack([bands[r_label], bands[g_label], bands[b_label]], axis=0)
        if view == TRUE_COLOR_SHADOW:
            out[view] = asinh_shadow_stretch(stacked, k=asinh_k, gamma=gamma)
        else:
            out[view] = percentile_stretch(stacked)
    return out


def save_rgb_png(rgb_uint8: np.ndarray, path: str) -> None:
    """Save a ``(3, H, W)`` uint8 array as a PNG, for quick visual QA."""
    from PIL import Image

    array_hwc = np.transpose(rgb_uint8, (1, 2, 0))
    Image.fromarray(array_hwc, mode="RGB").save(path)
