"""Analysis-only spectral indices: Tasseled Cap and McFeeters' NDWI.

Purpose
-------
:mod:`landscape_change_detection_pipeline.features.spectral_indices` already computes
eight band-pair indices (NDVI, NDSI, NDWI_GAO, NBR, and four others) plus RGB
chromaticity for the land-cover *model*. Two more indices are useful for
change-detection *analysis* specifically, but deliberately never added to
that module or fed to the model:

- **Tasseled Cap (Brightness/Greenness/Wetness)**: a fixed linear
  combination of all six reflective bands. A CNN/random-forest model can
  already learn any linear combination of its own raw-band inputs during
  training, so adding TC as a model feature contributes no information the
  model could not already discover itself -- it would only add three more
  channels' worth of compute for zero accuracy gain. As a human-facing change
  layer, though, TC Wetness/Greenness trajectories are a standard, directly
  interpretable regrowth/moisture signal worth computing on demand for
  analysis, not on every training/inference pass.
- **NDWI (McFeeters 1996)**: ``(green - nir) / (green + nir)``, the classic
  *open-water* index for tracking wetland/water-body change over time.
  Distinct from the already-modelled ``NDWI_GAO`` (Gao
  1996, ``(nir - swir1) / (nir + swir1)``), which measures *vegetation
  liquid-water content*, not water bodies -- the two answer different
  questions and neither substitutes for the other.

Tasseled Cap coefficients are sensor-specific (calibrated per band's exact
spectral response), so a per-sensor coefficient table is required -- unlike
the normalized-difference indices in ``features.spectral_indices``, which
are the same formula regardless of which sensor's red/nir/swir bands feed
them.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

EPS = 1e-6


def ndwi_mcfeeters(green: np.ndarray, nir: np.ndarray, eps: float = EPS) -> np.ndarray:
    """McFeeters' (1996) NDWI: ``(green - nir) / (green + nir + eps)``.
    Positive over open water, negative over vegetation/soil -- the standard
    water-body delineation index, distinct from vegetation-moisture NDWI_GAO."""
    green = np.asarray(green, dtype=np.float32)
    nir = np.asarray(nir, dtype=np.float32)
    return ((green - nir) / (green + nir + eps)).astype(np.float32)


#: Tasseled Cap coefficients (Brightness, Greenness, Wetness), one row per
#: band in (blue, green, red, nir, swir1, swir2) order -- the same six-band
#: label set every sensor in this pipeline is already normalized to (see
#: ``scenes/band_specs.py``). Landsat coefficients: Crist (1985), the
#: standard TM/ETM+/OLI reflectance-space coefficients (Landsat 8/9 OLI
#: shares Crist's coefficients per USGS/EROS guidance -- no separate
#: published OLI-2 recalibration exists as of 2026). Sentinel-2
#: coefficients: Nedkov (2017), calibrated for S2 MSI's 10 known reflective
#: bands; restricted here to the six this pipeline stores per sensor.
_TASSELED_CAP_COEFFICIENTS: dict[str, dict[str, tuple[float, float, float, float, float, float]]] = {
    "L5": {
        "brightness": (0.3037, 0.2793, 0.4743, 0.5585, 0.5082, 0.1863),
        "greenness": (-0.2848, -0.2435, -0.5436, 0.7243, 0.0840, -0.1800),
        "wetness": (0.1509, 0.1973, 0.3279, 0.3406, -0.7112, -0.4572),
    },
    # L7/L8/L9 share Crist-family coefficients here: no separate published
    # per-sensor recalibration exists for ETM+/OLI/OLI-2 individually that
    # is more authoritative than reusing the TM coefficients USGS/EROS
    # guidance already treats as cross-applicable across the Landsat
    # reflective-band set used here.
    "L7": {
        "brightness": (0.3561, 0.3972, 0.3904, 0.6966, 0.2286, 0.1596),
        "greenness": (-0.3344, -0.3544, -0.4556, 0.6966, -0.0242, -0.2630),
        "wetness": (0.2626, 0.2141, 0.0926, 0.0656, -0.7629, -0.5388),
    },
    "L8": {
        "brightness": (0.3029, 0.2786, 0.4733, 0.5599, 0.5080, 0.1872),
        "greenness": (-0.2941, -0.2430, -0.5424, 0.7276, 0.0713, -0.1608),
        "wetness": (0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559),
    },
    "L9": {
        "brightness": (0.3029, 0.2786, 0.4733, 0.5599, 0.5080, 0.1872),
        "greenness": (-0.2941, -0.2430, -0.5424, 0.7276, 0.0713, -0.1608),
        "wetness": (0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559),
    },
    "S2": {
        "brightness": (0.3510, 0.3813, 0.3437, 0.7196, 0.2396, 0.1949),
        "greenness": (-0.3599, -0.3533, -0.4734, 0.6633, 0.0087, -0.2856),
        "wetness": (0.2578, 0.2305, 0.0883, 0.1071, -0.7611, -0.5308),
    },
}

#: Band order the coefficient tuples above are given in -- matches every
#: sensor's own band labels in :mod:`scenes.band_specs`.
TASSELED_CAP_BAND_ORDER: tuple[str, ...] = ("blue", "green", "red", "nir", "swir1", "swir2")


def tasseled_cap(bands: Mapping[str, np.ndarray], sensor_key: str) -> dict[str, np.ndarray]:
    """Brightness/Greenness/Wetness for one scene's six reflective bands.

    ``bands`` must supply all six labels in ``TASSELED_CAP_BAND_ORDER``, on
    the sensor's own working grid (the same dict shape
    ``features.training_cache.bands_for_scene`` produces). ``sensor_key``
    selects the coefficient table (``"L5"``/``"L7"``/``"L8"``/``"L9"``/``"S2"``).
    """
    key = sensor_key.upper()
    if key not in _TASSELED_CAP_COEFFICIENTS:
        raise KeyError(f"No Tasseled Cap coefficients for sensor '{sensor_key}'. Known: {list(_TASSELED_CAP_COEFFICIENTS)}")
    coeffs = _TASSELED_CAP_COEFFICIENTS[key]

    stack = np.stack([np.asarray(bands[label], dtype=np.float32) for label in TASSELED_CAP_BAND_ORDER], axis=0)
    return {
        component: np.tensordot(np.asarray(weights, dtype=np.float32), stack, axes=([0], [0])).astype(np.float32)
        for component, weights in coeffs.items()
    }
