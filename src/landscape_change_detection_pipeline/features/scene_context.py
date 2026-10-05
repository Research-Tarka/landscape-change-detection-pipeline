"""Scene-level conditioning vector for the model's FiLM layers.

One short vector per scene, built from that scene alone (its sensor, acquisition
date and grid centroid) -- nothing from other scenes. Layout (fixed, so a
checkpoint and the inference code always agree)::

    [sensor_index, doy_sin, doy_cos, lat_norm, lon_norm]

The model decides which columns it reads (``film_sensor`` / ``film_doy`` /
``film_latlon``); the vector is always built in full.
"""

from __future__ import annotations

import numpy as np

#: Sensor -> embedding index. Order is part of the checkpoint contract.
SENSOR_ORDER: tuple[str, ...] = ("l5", "l7", "l8", "l9", "s2")
N_SENSORS = len(SENSOR_ORDER)

CONTEXT_DIM = 5
SENSOR_COL = 0
DOY_COLS = slice(1, 3)
LATLON_COLS = slice(3, 5)


def sensor_index(sensor: str) -> int:
    try:
        return SENSOR_ORDER.index(sensor.lower())
    except ValueError:
        raise ValueError(f"unknown sensor {sensor!r}; expected one of {SENSOR_ORDER}") from None


def scene_context_vector(sensor: str, scene_id: str, transform, crs_wkt: str, shape: tuple[int, int]) -> np.ndarray:
    """``(CONTEXT_DIM,)`` float32 context for one scene (``shape`` = its ``(H, W)`` grid)."""
    from landscape_change_detection_pipeline.features.spectral_indices import centroid_latlon_norm
    from landscape_change_detection_pipeline.features.training_cache import scene_date

    doy = scene_date(sensor, scene_id).timetuple().tm_yday
    angle = 2.0 * np.pi * (doy / 365.25)
    lat_norm, lon_norm = centroid_latlon_norm(transform, crs_wkt, shape)
    return np.array(
        [sensor_index(sensor), np.sin(angle), np.cos(angle), lat_norm, lon_norm], dtype=np.float32
    )


def record_context(record) -> np.ndarray:
    """Context of a cached training scene (reads only the cache's metadata and array shape)."""
    from landscape_change_detection_pipeline.features.training_cache import FEATURES_NPY, load_scene_cache

    cache_dir = record.cache_dir
    if (cache_dir / FEATURES_NPY).exists():
        shape = np.load(cache_dir / FEATURES_NPY, mmap_mode="r").shape[1:]
    else:
        shape = load_scene_cache(cache_dir)["features"].shape[1:]
    meta = load_scene_cache(cache_dir, mmap=True)
    if not meta["crs_wkt"]:
        raise ValueError(f"cache {cache_dir} has no georeferencing; re-run scripts/04_export_training_cache.py")
    return scene_context_vector(record.sensor, record.scene_id, meta["transform"], meta["crs_wkt"], tuple(shape))
