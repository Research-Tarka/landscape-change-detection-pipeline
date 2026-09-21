"""End-to-end per-scene processing: fetch, enhance to working resolution, store.

Purpose
-------
Ties together :mod:`.gee_fetch` (download), :mod:`.pansharpen` (GS-Adaptive
for the analytic stack, Brovey for RGB visuals), :mod:`.composites` (RGB Raw/
Shadow), and :mod:`.zarr_store` (append) into the one function the
orchestration module calls per scene.

Per-sensor path
----------------
- **L5**: one 30 m band-group fetch, no pan band, no enhancement -- stored
  as-is.
- **L7/L8/L9**: two band-group fetches (30 m reflective bands, 15 m pan
  band); blue/green/red/NIR pansharpened with GS-Adaptive; SWIR1/SWIR2
  resampled (bilinear) to the 15 m grid; RGB additionally pansharpened with
  Brovey for the visualization composites.
- **S2**: two band-group fetches (10 m native bands, 20 m bands); the 20 m
  bands resampled (bilinear) to the 10 m grid, a resample rather than
  DSen2 super-resolution.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .band_specs import BandRole, SensorBandSpec, get_band_spec
from .gee_fetch import fetch_band_group, reflectance_to_uint16, resample_band_to_grid, window_transform
from .pansharpen import brovey_pansharpen, gram_schmidt_adaptive_pansharpen
from .composites import build_rgb_composites
from .sensors import SensorSpec, get_sensor
from .zarr_store import scene_already_stored, write_scene

#: "native" | "pansharpened" | "resampled_for_alignment" -- no sensor currently
#: writes "learned_super_resolved".
PROVENANCE_NATIVE = "native"
PROVENANCE_PANSHARPENED = "pansharpened"
PROVENANCE_RESAMPLED = "resampled_for_alignment"


def _fetch_bands_at_resolution(
    collection: str,
    scene_id: str,
    band_names: list[str],
    window_bbox: tuple[float, float, float, float],
    crs: str,
    resolution_m: float,
    reflectance_scale_divisor: float = 1.0,
) -> tuple[dict[str, np.ndarray], object]:
    """Fetch one group of same-resolution bands, returning ``{name: array}`` + transform.

    ``reflectance_scale_divisor`` normalizes a collection's raw pixel values
    to true [0, 1] TOA reflectance (see ``SensorSpec.reflectance_scale_divisor``
    -- 1.0 for the Landsat T1_TOA collections, 10000.0 for
    ``COPERNICUS/S2_HARMONIZED``).
    """
    arr, transform = fetch_band_group(collection, scene_id, band_names, window_bbox, crs, resolution_m)
    if reflectance_scale_divisor != 1.0:
        arr = arr / reflectance_scale_divisor
    bands = {name: arr[i] for i, name in enumerate(band_names)}
    return bands, transform


def process_scene_no_enhancement(
    sensor: SensorSpec,
    band_spec: SensorBandSpec,
    scene_id: str,
    window_bbox: tuple[float, float, float, float],
    crs: str,
) -> tuple[np.ndarray, list[str], dict[str, str], np.ndarray, np.ndarray, object]:
    """Landsat 5's path: one native-resolution fetch, no pansharpening/resampling."""
    band_names = list(band_spec.toa_band_names())
    bands, transform = _fetch_bands_at_resolution(
        sensor.collection, scene_id, band_names, window_bbox, crs, band_spec.working_resolution_m,
        reflectance_scale_divisor=sensor.reflectance_scale_divisor,
    )

    toa_float = np.stack([bands[n] for n in band_names], axis=0)
    toa = reflectance_to_uint16(toa_float)
    provenance = {n: PROVENANCE_NATIVE for n in band_names}

    rgb_r, rgb_g, rgb_b = band_spec.rgb_bands
    rgb = build_rgb_composites({"red": bands[rgb_r], "green": bands[rgb_g], "blue": bands[rgb_b]})
    return toa, band_names, provenance, rgb["rgb_raw"], rgb["rgb_shadow"], transform


def process_scene_pansharpened(
    sensor: SensorSpec,
    band_spec: SensorBandSpec,
    scene_id: str,
    window_bbox: tuple[float, float, float, float],
    crs: str,
) -> tuple[np.ndarray, list[str], dict[str, str], np.ndarray, np.ndarray, object]:
    """Landsat 7/8/9's path: GS-Adaptive pansharpen + resample onto the 15 m grid."""
    pansharpen_names = list(band_spec.pansharpen_band_names())
    resample_names = list(band_spec.resample_band_names())
    coarse_names = pansharpen_names + resample_names

    coarse_bands, coarse_transform = _fetch_bands_at_resolution(
        sensor.collection, scene_id, coarse_names, window_bbox, crs, 30.0,
        reflectance_scale_divisor=sensor.reflectance_scale_divisor,
    )
    pan_bands, pan_transform = _fetch_bands_at_resolution(
        sensor.collection, scene_id, [band_spec.pan_band], window_bbox, crs, band_spec.working_resolution_m,
        reflectance_scale_divisor=sensor.reflectance_scale_divisor,
    )
    pan = pan_bands[band_spec.pan_band]
    dst_shape = pan.shape

    # Upsample the coarse bands onto the pan grid (plain bilinear -- the
    # pansharpen/resample step below is what adds real detail on top).
    upsampled = {
        name: resample_band_to_grid(arr, coarse_transform, pan_transform, dst_shape, crs, method="bilinear")
        for name, arr in coarse_bands.items()
    }

    pansharpen_inputs = {n: upsampled[n] for n in pansharpen_names}
    sharpened = gram_schmidt_adaptive_pansharpen(pansharpen_inputs, pan)

    provenance: dict[str, str] = {n: PROVENANCE_PANSHARPENED for n in pansharpen_names}
    provenance.update({n: PROVENANCE_RESAMPLED for n in resample_names})

    band_names = list(band_spec.toa_band_names())
    toa_bands = {**sharpened, **{n: upsampled[n] for n in resample_names}}
    toa_float = np.stack([toa_bands[n] for n in band_names], axis=0)
    toa = reflectance_to_uint16(toa_float)

    rgb_r, rgb_g, rgb_b = band_spec.rgb_bands
    rgb_pansharpened = brovey_pansharpen(
        {"red": upsampled[rgb_r], "green": upsampled[rgb_g], "blue": upsampled[rgb_b]}, pan
    )
    rgb = build_rgb_composites(rgb_pansharpened)
    return toa, band_names, provenance, rgb["rgb_raw"], rgb["rgb_shadow"], pan_transform


def process_scene_resampled(
    sensor: SensorSpec,
    band_spec: SensorBandSpec,
    scene_id: str,
    window_bbox: tuple[float, float, float, float],
    crs: str,
) -> tuple[np.ndarray, list[str], dict[str, str], np.ndarray, np.ndarray, object]:
    """Sentinel-2's path: native 10 m bands + 20 m bands resampled to 10 m."""
    native_names = [b.name for b in band_spec.bands if b.role == BandRole.NATIVE]
    resample_names = list(band_spec.resample_band_names())

    native_bands, native_transform = _fetch_bands_at_resolution(
        sensor.collection, scene_id, native_names, window_bbox, crs, band_spec.working_resolution_m,
        reflectance_scale_divisor=sensor.reflectance_scale_divisor,
    )
    coarse_bands, coarse_transform = _fetch_bands_at_resolution(
        sensor.collection, scene_id, resample_names, window_bbox, crs, 20.0,
        reflectance_scale_divisor=sensor.reflectance_scale_divisor,
    )
    dst_shape = native_bands[native_names[0]].shape
    resampled = {
        name: resample_band_to_grid(arr, coarse_transform, native_transform, dst_shape, crs, method="bilinear")
        for name, arr in coarse_bands.items()
    }

    provenance = {n: PROVENANCE_NATIVE for n in native_names}
    provenance.update({n: PROVENANCE_RESAMPLED for n in resample_names})

    band_names = list(band_spec.toa_band_names())
    all_bands = {**native_bands, **resampled}
    toa_float = np.stack([all_bands[n] for n in band_names], axis=0)
    toa = reflectance_to_uint16(toa_float)

    rgb_r, rgb_g, rgb_b = band_spec.rgb_bands
    rgb = build_rgb_composites(
        {"red": all_bands[rgb_r], "green": all_bands[rgb_g], "blue": all_bands[rgb_b]}
    )
    return toa, band_names, provenance, rgb["rgb_raw"], rgb["rgb_shadow"], native_transform


def process_and_store_scene(
    tile_dir,
    tile_id: str,
    sensor_key: str,
    scene_id: str,
    window_bbox: tuple[float, float, float, float],
    crs: str,
    skip_existing: bool = True,
) -> str:
    """Fetch, enhance, and append one scene into the tile's zarr store.

    Returns a short status string (``"ok"``, ``"skip (already stored)"``, or
    ``"error (...)"``); never raises, so a sweep over many scenes is not
    aborted by one bad fetch (mirrors ``dem/engine.py::process_one_tile``'s
    per-item error containment).
    """
    if skip_existing and scene_already_stored(tile_dir, tile_id, sensor_key, scene_id):
        return "skip (already stored)"

    try:
        sensor = get_sensor(sensor_key)
        band_spec = get_band_spec(sensor_key)

        if band_spec.pan_band is not None:
            toa, band_names, provenance, rgb_raw, rgb_shadow, transform = process_scene_pansharpened(
                sensor, band_spec, scene_id, window_bbox, crs
            )
        elif any(n for n in band_spec.resample_band_names()):
            toa, band_names, provenance, rgb_raw, rgb_shadow, transform = process_scene_resampled(
                sensor, band_spec, scene_id, window_bbox, crs
            )
        else:
            toa, band_names, provenance, rgb_raw, rgb_shadow, transform = process_scene_no_enhancement(
                sensor, band_spec, scene_id, window_bbox, crs
            )

        write_scene(
            tile_dir,
            tile_id,
            sensor_key,
            scene_id,
            toa,
            band_names,
            provenance,
            rgb_raw,
            rgb_shadow,
            transform,
            crs_wkt=crs,
            skip_existing=skip_existing,
        )
        return "ok"
    except Exception as exc:  # noqa: BLE001 -- reported per scene, never fatal
        return f"error ({type(exc).__name__}: {exc})"
