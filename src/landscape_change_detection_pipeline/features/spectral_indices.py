"""Spectral index computation: a declarative table of band-pair formulas,
each implemented as a vectorized whole-array function.

Purpose
-------
One ``eps``-stabilized ``normalized_difference`` helper, a declarative name
table, and a "compute all, return a dict" entry point (``compute_indices_dict``)
plus a "stack selected names into one array" entry point (``compute_indices``).
No pixel loops anywhere -- every index is one numpy expression over whole
``(H, W)`` (or higher-dimensional) arrays.

Every index runs on the pipeline's single uniform 10 m grid, pinned to the
DEM's own exact transform/shape (see ``dem/grid_check.py``'s exact-match
invariant, now actually enforced at scene-write time rather than just
documented -- ``docs/decisions/unified_10m_grid.md``) -- every band of a
scene is already aligned onto that one grid before this module runs, so
there is no per-index resolution split here.

Provenance
----------
Each *band* is tagged as ``"native"`` or ``"resampled_for_alignment"`` (see
``scenes.zarr_store``'s ``band_provenance`` attr, via
``scenes.band_specs.BandRole``). An index computed from two bands inherits
the *worse* of its inputs' provenance, using the ordering
``native < resampled_for_alignment`` (worse = further from that grid's
genuine native detail): NDSI/NBR/etc. computed from any Landsat band
(resampled from ~30 m onto the finer 10 m grid) are tagged
``"resampled_for_alignment"``, so the model/consumer knows not to trust
genuine 10 m detail from them. This is metadata only -- it never changes how
an index is computed.
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
    "resampled_for_alignment": 1,
}


def combined_provenance(*provenances: str) -> str:
    """The provenance tag for a value derived from one or more bands: the
    worst (least-native) of the inputs' tags, so an index computed from any
    ``resampled_for_alignment`` band is itself tagged as not carrying genuine
    detail at the working resolution."""
    return max(provenances, key=lambda p: _PROVENANCE_RANK.get(p, 2))


def _nonzero(denominator: np.ndarray, eps: float) -> np.ndarray:
    """``denominator`` with exact/near zeros pushed to ``+eps`` (keeps sign otherwise)."""
    return np.where(np.abs(denominator) < eps, eps, denominator).astype(np.float32)


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
    # Built-up index (Zha et al. 2003): (swir1 - nir) / (swir1 + nir).
    # Roofing/pavement reflects more strongly in SWIR1 than NIR, the
    # opposite of vegetation and closer to bare soil than either -- this is
    # the one channel in this table specifically discriminative for
    # built-up surfaces, which the existing indices (vegetation/snow/water
    # focused) don't separate from bare ground.
    "NDBI": ("swir1", "nir"),
    # McFeeters (1996) open-water NDWI: (green - nir) / (green + nir).
    "NDWI_MCFEETERS": ("green", "nir"),
}

#: Chromaticity indices: band / (red + green + blue + eps).
CHROMATICITY_INDICES: dict[str, str] = {
    "r": "red",
    "g": "green",
    "b": "blue",
}

#: Non-normalized-difference band-combination indices, computed straight from
#: the sensor-independent band labels (same formula for every sensor).
#: name -> the band labels it reads (for provenance).
BAND_COMBINATION_INDICES: dict[str, tuple[str, ...]] = {
    "BSI": ("blue", "red", "nir", "swir1"),  # Bare Soil Index (Rikimaru 2002)
    "EVI": ("blue", "red", "nir"),  # Enhanced Vegetation Index (Huete 2002)
    "SAVI": ("red", "nir"),  # Soil-Adjusted Vegetation Index, L=0.5 (Huete 1988)
}

#: Tasseled Cap components: a per-sensor linear combination of all six bands
#: (coefficients differ per sensor, see ``change.analysis_indices``), so
#: unlike every other index here they need the scene's sensor to be computed.
TASSELED_CAP_INDICES: dict[str, str] = {
    "TC_BRIGHTNESS": "brightness",
    "TC_GREENNESS": "greenness",
    "TC_WETNESS": "wetness",
}

#: Indices computable from the band dict alone (no sensor needed).
INDEX_NAMES: tuple[str, ...] = (
    *NORMALIZED_DIFFERENCE_INDICES, *CHROMATICITY_INDICES, *BAND_COMBINATION_INDICES,
)

#: Every index the land-cover model can take as input (``features.index_names``).
MODEL_INDEX_NAMES: tuple[str, ...] = (*INDEX_NAMES, *TASSELED_CAP_INDICES)

#: DEM-derived layers carried alongside spectral indices in a feature stack.
#: Always "native" provenance -- the DEM stage fixes the tile's grid exactly,
#: it is never resampled onto a scene's grid after the fact (see
#: dem/grid_check.py's exact-match invariant).
DEM_LAYER_NAMES: tuple[str, ...] = ("elevation", "slope", "aspect")

#: Model-input channels a DEM layer expands to. ``aspect`` is circular (359 deg
#: is next to 1 deg) and NaN on flat cells, so it enters the model as
#: sin/cos with flat cells at 0 -- a raw-degrees channel would both break at the
#: wrap and make every flat pixel (lakes, plains) non-finite, i.e. nodata at inference.
DEM_LAYER_CHANNELS: dict[str, tuple[str, ...]] = {"aspect": ("aspect_sin", "aspect_cos")}


def expand_dem_feature_names(dem_layer_names) -> tuple[str, ...]:
    """Feature-channel names for ``dem_layer_names`` (``aspect`` -> sin/cos pair)."""
    return tuple(c for name in dem_layer_names for c in DEM_LAYER_CHANNELS.get(name, (name,)))


def dem_layer_to_channels(name: str, array: np.ndarray) -> list[np.ndarray]:
    """The ``(H, W)`` float32 channel(s) one DEM layer contributes to the stack."""
    array = np.asarray(array, dtype=np.float32)
    if name == "aspect":
        angle = np.deg2rad(array)
        return [np.nan_to_num(np.sin(angle), nan=0.0), np.nan_to_num(np.cos(angle), nan=0.0)]
    return [array]


#: Acquisition-date channels: cyclical day-of-year encoding, so the model
#: sees seasonality directly (e.g. bare/snow-covered ground reads very
#: differently in winter vs. summer under the same land-cover class) instead
#: of that being folded into the class taxonomy itself. sin/cos of DOY
#: (rather than a raw day-of-year integer) avoids the Dec-31 -> Jan-1
#: discontinuity a linear encoding would create.
DOY_FEATURE_NAMES: tuple[str, ...] = ("doy_sin", "doy_cos")

#: Acquisition-location channels: scene-centroid latitude/longitude,
#: normalized to roughly [-1, 1] (divided by 90/180 respectively). Same
#: rationale as DOY -- phenology/snowline/treeline shift with latitude, and
#: this lets one model generalize across a geographically broad training set
#: instead of needing a separate model per region.
LATLON_FEATURE_NAMES: tuple[str, ...] = ("lat_norm", "lon_norm")


def doy_cyclical_stack(acquisition_date, shape: tuple[int, int]) -> np.ndarray:
    """``(2, H, W)`` constant sin/cos-of-day-of-year channels for one scene,
    broadcast across the whole grid (the date is scene-wide, not per-pixel).

    ``acquisition_date`` is a ``datetime.date`` (see
    ``features.training_cache.scene_date``).
    """
    doy = acquisition_date.timetuple().tm_yday
    angle = 2.0 * np.pi * (doy / 365.25)
    sin_val = np.float32(np.sin(angle))
    cos_val = np.float32(np.cos(angle))
    return np.stack(
        [np.full(shape, sin_val, dtype=np.float32), np.full(shape, cos_val, dtype=np.float32)],
        axis=0,
    )


def centroid_latlon_norm(transform, crs, shape: tuple[int, int]) -> tuple[np.float32, np.float32]:
    """``(lat/90, lon/180)`` of the grid's centroid (``transform``/``crs`` = the scene grid)."""
    from affine import Affine
    from pyproj import Transformer
    from rasterio.crs import CRS

    h, w = shape
    aff = transform if isinstance(transform, Affine) else Affine(*transform[:6])
    center_x, center_y = aff * (w / 2.0, h / 2.0)

    transformer = Transformer.from_crs(CRS.from_user_input(crs), "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(center_x, center_y)
    return np.float32(lat / 90.0), np.float32(lon / 180.0)


def latlon_stack(
    transform,
    crs: str,
    shape: tuple[int, int],
) -> np.ndarray:
    """``(2, H, W)`` constant normalized lat/lon channels for one scene,
    computed from the scene grid's own centroid and broadcast across the
    whole grid (position is scene-wide, not per-pixel -- a scene's own
    ~8 km tile footprint is far smaller than the scale phenology/snowline
    varies over, so the centroid is an adequate stand-in for a true
    per-pixel value).

    ``transform``/``crs`` are the scene's working-resolution grid
    georeferencing (see ``features.training_cache.build_feature_stack``'s
    own ``dst_transform``/``dst_crs``).
    """
    lat_norm, lon_norm = centroid_latlon_norm(transform, crs, shape)
    return np.stack(
        [np.full(shape, lat_norm, dtype=np.float32), np.full(shape, lon_norm, dtype=np.float32)],
        axis=0,
    )


def compute_indices_dict(
    bands: Mapping[str, np.ndarray],
    band_provenance: Mapping[str, str],
    eps: float = EPS,
    names: tuple[str, ...] = INDEX_NAMES,
    sensor: str | None = None,
) -> dict[str, tuple[np.ndarray, str]]:
    """Compute every requested index from ``bands`` (keyed by label: blue,
    green, red, nir, swir1, swir2), returning ``{name: (array, provenance)}``.
    ``sensor`` (``"L8"``, ``"S2"``...) is only required when a Tasseled Cap
    index is requested.

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

    if any(name in names for name in BAND_COMBINATION_INDICES):
        f32 = {label: np.asarray(bands[label], dtype=np.float32) for label in ("blue", "red", "nir", "swir1")
               if label in bands}
        formulas = {
            "BSI": lambda: ((f32["swir1"] + f32["red"]) - (f32["nir"] + f32["blue"]))
            / ((f32["swir1"] + f32["red"]) + (f32["nir"] + f32["blue"]) + eps),
            "EVI": lambda: 2.5 * (f32["nir"] - f32["red"]) / _nonzero(
                f32["nir"] + 6.0 * f32["red"] - 7.5 * f32["blue"] + 1.0, eps
            ),
            "SAVI": lambda: 1.5 * (f32["nir"] - f32["red"]) / (f32["nir"] + f32["red"] + 0.5),
        }
        for name, labels in BAND_COMBINATION_INDICES.items():
            if name not in names:
                continue
            array = np.asarray(formulas[name](), dtype=np.float32)
            if name == "EVI":
                array = np.clip(np.nan_to_num(array, nan=0.0), -1.0, 1.0)  # denominator can approach 0
            out[name] = (array, combined_provenance(*(band_provenance[label] for label in labels)))

    tc_requested = [name for name in TASSELED_CAP_INDICES if name in names]
    if tc_requested:
        if sensor is None:
            raise ValueError(f"Tasseled Cap indices {tc_requested} need the scene's sensor (pass sensor=...).")
        from landscape_change_detection_pipeline.change.analysis_indices import (
            TASSELED_CAP_BAND_ORDER,
            tasseled_cap,
        )

        tc = tasseled_cap(bands, sensor)
        tc_provenance = combined_provenance(*(band_provenance[label] for label in TASSELED_CAP_BAND_ORDER))
        for name in tc_requested:
            out[name] = (tc[TASSELED_CAP_INDICES[name]], tc_provenance)

    return out


def compute_indices(
    bands: Mapping[str, np.ndarray],
    band_provenance: Mapping[str, str],
    eps: float = EPS,
    names: tuple[str, ...] = INDEX_NAMES,
    sensor: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Stack the requested indices into one ``(len(names), H, W)`` float32
    array, in ``names`` order. Returns ``(stack, provenance_per_channel)``."""
    computed = compute_indices_dict(bands, band_provenance, eps, names, sensor)
    arrays = [computed[name][0] for name in names]
    provenance = [computed[name][1] for name in names]
    return np.stack(arrays, axis=0).astype(np.float32), provenance
