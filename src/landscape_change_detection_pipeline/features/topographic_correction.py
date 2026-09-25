"""Topographic correction (illumination normalization): SCS+C method.

Purpose
-------
In mountainous/hilly terrain, a sun-facing slope and a shaded slope of the
same forest stand return measurably different reflectance for the same
ground cover -- an illumination artifact, not a real difference in land
cover, that degrades classification accuracy. This module corrects for it
using each tile's own slope/aspect (already computed at the DEM stage) plus
each scene's solar illumination geometry at acquisition time (sun
elevation/azimuth, fetched alongside the scene -- see
``scenes.gee_fetch.fetch_scene_solar_angles``).

This is **not** a resolution change (see
``docs/decisions/topographic_correction.md`` for why a DEM cannot add
genuine spatial resolution to optical reflectance -- the literature runs
the opposite direction, using imagery to sharpen a DEM, not the reverse).
It belongs alongside the cross-sensor bandpass adjustment
(:mod:`scenes.harmonize`) -- both affect reflectance values before indices/
training are computed -- but is conceptually independent of the unified
10 m grid.

Method: SCS+C (Soenen, Peddle & Coburn 2005, IEEE TGRS)
--------------------------------------------------------
Chosen over plain Cosine correction (over-corrects near-flat pixels,
amplifying noise where no real illumination difference exists) and plain
SCS (tends to over-correct steep slopes) -- SCS+C's ``C`` term, an
empirical per-band correction derived from the scene's own reflectance-vs-
illumination regression, damps both failure modes and is documented in the
topographic-correction literature as the standard choice for forested,
moderate-relief terrain (the Soenen et al. 2005 evaluation was itself a
Rocky Mountain forest setting).

Given:

- ``theta_z``: solar zenith angle (90 - sun elevation), radians
- ``theta_s``: slope, radians
- ``phi_a``: terrain aspect, radians
- ``phi_s``: solar azimuth, radians

The illumination condition (cosine of the incidence angle between the sun
and the local surface normal) is::

    IL = cos(theta_z) * cos(theta_s) + sin(theta_z) * sin(theta_s) * cos(phi_a - phi_s)

The corrected reflectance is::

    L_corrected = L_observed * (cos(theta_s) + C) / (IL + C)

where ``C = b / m``, with ``m``/``b`` the slope/intercept of a linear
regression of the observed reflectance against ``IL`` over the scene's own
valid pixels (Soenen et al. 2005's own empirical fit, refit per scene rather
than a fixed literature constant, since it depends on each scene's own
diffuse-irradiance balance). Unlike a literal reading of Soenen et al.
(which fits ``C`` independently per band), this module fits **one** ``C``
per scene, from a single reference band, and reuses it for every band --
see :func:`topographic_correct_scene`'s docstring for why (an independent
per-band fit was found to introduce a visible color cast in this pipeline's
RGB composites, confirmed live -- see ``docs/decisions/topographic_correction.md``).
The per-pixel correction ratio is also clipped to a bounded range (see
:data:`DEFAULT_RATIO_CLIP_MIN`/:data:`DEFAULT_RATIO_CLIP_MAX`) so a single
noisy pixel cannot blow out to black/saturated after display stretching.
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

#: Below this sun elevation (near sunrise/sunset), IL's denominator becomes
#: numerically unstable (dividing by a near-zero or negative cosine term);
#: correction is skipped entirely for such a scene rather than risk
#: amplifying noise into a large artifact.
DEFAULT_MIN_SUN_ELEVATION_DEG = 5.0

#: Per-pixel bound on the SCS+C correction ratio (``(cos(theta_s) + C) /
#: (IL + C)``). Confirmed live (2026-09-25, real pilot tiles): even when a
#: scene's fitted ``C`` passes ``_fit_c_parameter``'s own scene-wide
#: degeneracy guard, individual pixels can still land at a locally extreme
#: ``IL(x,y) + C`` (DEM-derived slope/aspect noise at pixel level, not
#: captured by the scene-aggregate fit) and produce an outlier ratio --
#: after ``composites.py::percentile_stretch``'s hard [0, 1] clip, an
#: extreme-negative-ratio pixel goes to black, an extreme-positive one
#: saturates. A >5x brightening or <0.2x dimming is never a real
#: illumination effect at this AOI's moderate relief, only fit/DEM noise, so
#: clipping the ratio directly bounds the correction regardless of cause.
DEFAULT_RATIO_CLIP_MIN = 0.2
DEFAULT_RATIO_CLIP_MAX = 5.0

#: Band label used for the single scene-wide ``C`` fit shared across every
#: band (see :func:`topographic_correct_scene`'s docstring for why a shared
#: C is used instead of an independent per-band fit). NIR typically has the
#: strongest, cleanest illumination-reflectance relationship over vegetated/
#: forested terrain, matching Soenen et al. 2005's own vegetated-terrain
#: evaluation.
DEFAULT_REFERENCE_BAND = "nir"


def illumination_condition(
    slope_deg: np.ndarray,
    aspect_deg: np.ndarray,
    sun_elevation_deg: float,
    sun_azimuth_deg: float,
) -> np.ndarray:
    """``IL``, the cosine of the incidence angle between the sun and each
    pixel's local surface normal, from the tile's slope/aspect (degrees, DEM
    convention: aspect 0=N clockwise) and the scene's solar geometry
    (degrees)."""
    theta_z = np.radians(90.0 - sun_elevation_deg)
    theta_s = np.radians(np.asarray(slope_deg, dtype=np.float64))
    phi_a = np.radians(np.asarray(aspect_deg, dtype=np.float64))
    phi_s = np.radians(sun_azimuth_deg)

    il = np.cos(theta_z) * np.cos(theta_s) + np.sin(theta_z) * np.sin(theta_s) * np.cos(phi_a - phi_s)
    return il.astype(np.float32)


def _fit_c_parameter(reflectance: np.ndarray, il: np.ndarray) -> float:
    """Fit ``L = m*IL + b`` over valid pixels, return ``C = b/m`` (Soenen et
    al. 2005's empirical diffuse-irradiance term). Falls back to ``C = 0``
    (equivalent to plain SCS, no empirical damping) if the fit is
    degenerate: near-zero slope, too few valid pixels, or -- the failure
    mode confirmed live on a low-relief tile with a narrow IL range (little
    illumination variance to regress against) -- a fitted C whose magnitude
    is large enough that ``IL + C`` goes negative (or near-zero) somewhere
    in the scene's actual IL range. That flips the correction ratio's sign
    for those pixels instead of scaling it, collapsing reflectance toward
    zero rather than normalizing illumination -- confirmed live (S2 B8 on
    tile_0057_0035, 2026-09-25: fitted C=-1.0246 against an IL range of
    [0.51, 0.89] drove the band's mean from 0.148 to 0.015). ``C=0``
    (plain SCS) is always well-defined since ``IL`` itself never goes
    negative for a physically valid illumination geometry.
    """
    valid = np.isfinite(reflectance) & np.isfinite(il)
    if valid.sum() < 30:
        return 0.0
    x = il[valid].astype(np.float64)
    y = reflectance[valid].astype(np.float64)
    if np.ptp(x) < 1e-6:
        return 0.0
    m, b = np.polyfit(x, y, 1)
    if abs(m) < 1e-9:
        return 0.0
    c = float(b / m)

    il_min = float(np.nanmin(il))
    if il_min + c <= 0.05:
        return 0.0
    return c


def scs_c_correct_band(
    reflectance: np.ndarray,
    slope_deg: np.ndarray,
    aspect_deg: np.ndarray,
    sun_elevation_deg: float,
    sun_azimuth_deg: float,
    c: Optional[float] = None,
    ratio_clip_min: float = DEFAULT_RATIO_CLIP_MIN,
    ratio_clip_max: float = DEFAULT_RATIO_CLIP_MAX,
) -> np.ndarray:
    """Apply SCS+C correction to one band's ``(H, W)`` reflectance array,
    using that tile's slope/aspect and this scene's solar geometry.

    ``c``, if given, is used as-is instead of being fit from ``reflectance``
    -- :func:`topographic_correct_scene` fits ``C`` once per scene (on a
    single reference band) and passes it into every band's correction, so
    every band shares one numerically consistent ratio (see that function's
    docstring for why). If ``c`` is ``None`` (e.g. a caller correcting a
    single band in isolation, as this project's own unit tests do), it is
    fit from this band's own reflectance-vs-illumination regression exactly
    as before.

    The per-pixel ratio is clipped to ``[ratio_clip_min, ratio_clip_max]``
    before being applied (see :data:`DEFAULT_RATIO_CLIP_MIN`/
    :data:`DEFAULT_RATIO_CLIP_MAX`'s docstring for why) -- this bounds any
    single outlier pixel's correction regardless of how well-behaved the
    scene-wide ``C`` fit is.

    Returns a NaN-preserving float32 array on the same grid. Flat pixels
    (``slope_deg`` ~0) are passed through unchanged -- ``IL`` degenerates
    toward ``cos(theta_z)`` there regardless of aspect, so the correction
    ratio is close to 1.0 already and the ``theta_s`` term in the numerator
    keeps it exactly so at zero slope.
    """
    theta_s = np.radians(np.asarray(slope_deg, dtype=np.float64))
    il = illumination_condition(slope_deg, aspect_deg, sun_elevation_deg, sun_azimuth_deg)
    if c is None:
        c = _fit_c_parameter(np.asarray(reflectance, dtype=np.float32), il)

    numerator = np.cos(theta_s) + c
    denominator = il.astype(np.float64) + c
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(denominator) > 1e-6, numerator / denominator, 1.0)
    ratio = np.clip(ratio, ratio_clip_min, ratio_clip_max)

    corrected = np.asarray(reflectance, dtype=np.float64) * ratio
    return corrected.astype(np.float32)


def topographic_correct_scene(
    bands: Mapping[str, np.ndarray],
    slope_deg: np.ndarray,
    aspect_deg: np.ndarray,
    sun_elevation_deg: float,
    sun_azimuth_deg: float,
    min_sun_elevation_deg: float = DEFAULT_MIN_SUN_ELEVATION_DEG,
    reference_band: str = DEFAULT_REFERENCE_BAND,
    ratio_clip_min: float = DEFAULT_RATIO_CLIP_MIN,
    ratio_clip_max: float = DEFAULT_RATIO_CLIP_MAX,
) -> dict[str, np.ndarray]:
    """Apply SCS+C correction to every band in ``bands`` ``{label: array}``.

    Returns the bands unchanged (a copy, still float32) if
    ``sun_elevation_deg`` is below ``min_sun_elevation_deg`` -- near-sunrise/
    sunset geometry makes the correction numerically unstable, so skipping
    is safer than risking an amplified artifact.

    One shared ``C`` for every band, not an independent fit per band
    -----------------------------------------------------------------
    ``C`` is fit **once per scene**, against ``reference_band`` (default
    NIR, see :data:`DEFAULT_REFERENCE_BAND`), and the resulting ratio is
    reused for every band -- not refit independently per band as an earlier
    version of this function did. Confirmed live: an independent per-band
    fit lets each of red/green/blue drift to its own, individually
    "well-behaved" ratio (each passing ``_fit_c_parameter``'s own
    degeneracy guard), but with no mechanism keeping the three ratios
    *proportionate* to each other. Since the RGB visualization composites
    (``scenes/composites.py::percentile_stretch``) stretch each output
    channel independently to its own percentiles, even a small systematic
    per-band bias in the correction ratio becomes a visible color cast
    (yellow/blue/red tint) once independent per-channel stretching
    amplifies it. Fitting one ``C`` from a single reference band and
    reusing it removes this degree of freedom entirely: since the ratio
    ``(cos(theta_s) + C) / (IL + C)`` never depends on a band's own
    reflectance once ``C`` is fixed (only on ``theta_s``/``IL``, both
    scene-wide/per-pixel geometry), it can be computed once and applied
    identically to every band, guaranteeing cross-band color-balance
    consistency at the cost of some band-specific correction fidelity
    (a deliberate accuracy-vs-visual-usability tradeoff -- see
    ``docs/decisions/topographic_correction.md``).

    Falls back to the first available band in ``bands`` if
    ``reference_band`` is not present (e.g. a caller correcting a band
    subset that excludes NIR).
    """
    if sun_elevation_deg < min_sun_elevation_deg:
        return {label: np.asarray(arr, dtype=np.float32) for label, arr in bands.items()}
    if not bands:
        return {}

    fit_label = reference_band if reference_band in bands else next(iter(bands))
    il = illumination_condition(slope_deg, aspect_deg, sun_elevation_deg, sun_azimuth_deg)
    c = _fit_c_parameter(np.asarray(bands[fit_label], dtype=np.float32), il)

    return {
        label: scs_c_correct_band(
            arr,
            slope_deg,
            aspect_deg,
            sun_elevation_deg,
            sun_azimuth_deg,
            c=c,
            ratio_clip_min=ratio_clip_min,
            ratio_clip_max=ratio_clip_max,
        )
        for label, arr in bands.items()
    }
