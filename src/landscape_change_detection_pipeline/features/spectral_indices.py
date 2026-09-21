"""Spectral index computation: a declarative table of band-pair formulas,
each implemented as a vectorized whole-array function.

Purpose
-------
One ``eps``-stabilized ``normalized_difference`` helper, a declarative name
table, and a "compute all, return a dict" entry point (``compute_indices_dict``)
plus a "stack selected names into one array" entry point (``compute_indices``).
No pixel loops anywhere -- every index is one numpy expression over whole
``(H, W)`` (or higher-dimensional) arrays.

Every index for a given scene runs at that sensor's single uniform working
resolution (30 m for L5, 15 m for L7/L8/L9, 10 m for S2) -- every band of a
scene is already aligned onto one grid before this module runs, so there is
no per-index resolution split here.

Provenance
----------
Each *band* is tagged as ``"native"``, ``"pansharpened"``, or
``"resampled_for_alignment"`` (see ``scenes.zarr_store``'s ``band_provenance``
attr -- ``"pansharpened"`` there is the ``"resampled_for_alignment"`` case
called out by the more specific "pansharpen vs plain resample"
split via ``scenes.band_specs.BandRole``). An index computed from two bands
inherits the *worse* of its inputs' provenance, using the ordering
``native < pansharpened < resampled_for_alignment`` (worse = further from
that grid's genuine native detail): NDSI/NBR/etc. computed from a
Landsat 7/8/9 SWIR band (plainly resampled, not pansharpened) are tagged
``"resampled_for_alignment"`` even though they sit on the 15 m working grid,
so the model/consumer knows not to trust genuine 15 m detail from them. This
is metadata only -- it never changes how an index is computed.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

EPS = 1e-6

#: Provenance ordering, worst-last -- an index's provenance is the max
#: (worst) of its input bands' provenance under this ordering.
_PROVENANCE_RANK = {
    "native": 0,
    "learned_super_resolved": 0,
    "pansharpened": 1,
    "resampled_for_alignment": 2,
}


def combined_provenance(*provenances: str) -> str:
    """The provenance tag for a value derived from one or more bands: the
    worst (least-native) of the inputs' tags, so an index computed from any
    ``resampled_for_alignment`` band is itself tagged as not carrying genuine
    detail at the working resolution."""
    return max(provenances, key=lambda p: _PROVENANCE_RANK.get(p, 2))


def normalized_difference(a: np.ndarray, b: np.ndarray, eps: float = EPS) -> np.ndarray:
    """``(a - b) / (a + b + eps)``, vectorized, float32 output."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return ((a - b) / (a + b + eps)).astype(np.float32)


#: Declarative table: index name -> (band_a, band_b) for a normalized
#: difference (a - b) / (a + b + eps). Band names match the "label" field
#: in ``scenes.band_specs.Band`` (blue/green/red/nir/swir1/swir2).
NORMALIZED_DIFFERENCE_INDICES: dict[str, tuple[str, str]] = {
    "NDVI": ("nir", "red"),
    "NDSI": ("green", "swir1"),
    "NDWI_GAO": ("nir", "swir1"),
    "NBR": ("nir", "swir2"),
    "ND_SWIR1_SWIR2": ("swir1", "swir2"),
    "ND_BLUE_RED": ("blue", "red"),
    "ND_BLUE_NIR": ("blue", "nir"),
    "ND_GREEN_RED": ("green", "red"),
}

#: Chromaticity indices: band / (red + green + blue + eps).
CHROMATICITY_INDICES: dict[str, str] = {
    "r": "red",
    "g": "green",
    "b": "blue",
}

INDEX_NAMES: tuple[str, ...] = (*NORMALIZED_DIFFERENCE_INDICES, *CHROMATICITY_INDICES)

#: DEM-derived layers carried alongside spectral indices in a feature stack.
#: Always "native" provenance -- the DEM stage fixes the tile's grid exactly,
#: it is never resampled onto a scene's grid after the fact (see
#: dem/grid_check.py's exact-match invariant).
DEM_LAYER_NAMES: tuple[str, ...] = ("elevation", "slope", "aspect")


def compute_indices_dict(
    bands: Mapping[str, np.ndarray],
    band_provenance: Mapping[str, str],
    eps: float = EPS,
    names: tuple[str, ...] = INDEX_NAMES,
) -> dict[str, tuple[np.ndarray, str]]:
    """Compute every requested index from ``bands`` (keyed by label: blue,
    green, red, nir, swir1, swir2), returning ``{name: (array, provenance)}``.

    ``band_provenance`` maps the same band labels to their
    provenance tag; each index's provenance is the worst of its input
    bands' tags (see :func:`combined_provenance`).
    """
    out: dict[str, tuple[np.ndarray, str]] = {}

    for name, (label_a, label_b) in NORMALIZED_DIFFERENCE_INDICES.items():
        if name not in names:
            continue
        array = normalized_difference(bands[label_a], bands[label_b], eps)
        provenance = combined_provenance(band_provenance[label_a], band_provenance[label_b])
        out[name] = (array, provenance)

    if any(name in names for name in CHROMATICITY_INDICES):
        red = np.asarray(bands["red"], dtype=np.float32)
        green = np.asarray(bands["green"], dtype=np.float32)
        blue = np.asarray(bands["blue"], dtype=np.float32)
        rgb_sum = (red + green + blue + eps).astype(np.float32)
        rgb_provenance = combined_provenance(
            band_provenance["red"], band_provenance["green"], band_provenance["blue"]
        )
        for name, label in CHROMATICITY_INDICES.items():
            if name not in names:
                continue
            channel = np.asarray(bands[label], dtype=np.float32)
            out[name] = ((channel / rgb_sum).astype(np.float32), rgb_provenance)

    return out


def compute_indices(
    bands: Mapping[str, np.ndarray],
    band_provenance: Mapping[str, str],
    eps: float = EPS,
    names: tuple[str, ...] = INDEX_NAMES,
) -> tuple[np.ndarray, list[str]]:
    """Stack the requested indices into one ``(len(names), H, W)`` float32
    array, in ``names`` order. Returns ``(stack, provenance_per_channel)``."""
    computed = compute_indices_dict(bands, band_provenance, eps, names)
    arrays = [computed[name][0] for name in names]
    provenance = [computed[name][1] for name in names]
    return np.stack(arrays, axis=0).astype(np.float32), provenance
