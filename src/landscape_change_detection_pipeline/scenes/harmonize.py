"""Cross-sensor radiometric bandpass adjustment: bring Landsat onto Sentinel-2's convention.

Purpose
-------
NASA's Harmonized Landsat Sentinel-2 (HLS) product aligns Landsat OLI/
ETM+/TM-equivalent bands with Sentinel-2 MSI bands via a per-band linear
transform, ``y = scale * x + offset`` (Claverie et al. 2018). This module
implements only that piece -- the bandpass-adjustment coefficients -- not
HLS's BRDF normalization or spatial co-registration (see
``docs/decisions/cross_sensor_harmonization.md`` for the full scoping
rationale and the honest effectiveness caveat: bandpass adjustment alone
leaves a real, terrain-dependent residual, better on flat ground than on
steep slopes).

Coefficients
------------
Claverie et al. 2018's own published table gives the MSI->OLI direction
(``oli = slope * msi + offset``, derived from 500 hyperspectral spectra
across 160 Hyperion scenes). Sentinel-2 is this pipeline's harmonization
*target* (Landsat is adjusted onto S2's convention, not the reverse -- S2 is
the finer-resolution, currently-larger share of the archive), so the
published coefficients are algebraically inverted here:
``msi = (oli - offset) / slope`` => ``msi = (1/slope) * oli +
(-offset/slope)``, i.e. this module's own ``scale = 1/slope``,
``offset = -offset/slope``. The published S2A coefficients are used (S2A
and S2B differ by well under 1% per band; this pipeline does not currently
distinguish S2A/S2B as separate sensors, see ``scenes/sensors.py``).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


def apply_bandpass(array: np.ndarray, scale: float, offset: float) -> np.ndarray:
    """``y = scale * x + offset``, NaN-preserving, float32 output."""
    arr = np.asarray(array, dtype=np.float32)
    out = arr * np.float32(scale) + np.float32(offset)
    return out.astype(np.float32)


def harmonize_bands(
    bands: Mapping[str, np.ndarray],
    coefficients: Mapping[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    """Apply the per-band linear transform to every band with a configured
    coefficient; a band without one is passed through unchanged.

    Parameters
    ----------
    bands
        ``{band_label: array}`` -- band *labels* (blue/green/red/nir/swir1/
        swir2), not raw EE band names, so the same coefficient table works
        across Landsat sensors' differing band-name layouts.
    coefficients
        ``{band_label: (scale, offset)}``.
    """
    out: dict[str, np.ndarray] = {}
    for label, arr in bands.items():
        if label in coefficients:
            scale, offset = coefficients[label]
            out[label] = apply_bandpass(arr, scale, offset)
        else:
            out[label] = np.asarray(arr, dtype=np.float32)
    return out
