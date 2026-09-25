"""End-to-end per-scene processing: fetch, resample onto the DEM's grid, store.

Purpose
-------
Ties together :mod:`.gee_fetch` (download), :mod:`.composites` (RGB Raw/
Shadow), and :mod:`.zarr_store` (append) into the one function the
orchestration module calls per scene. Every sensor lands on the same
uniform 10 m grid, pinned to the exact transform/shape of the tile's
already-stored DEM (see ``dem/zarr_store.py::read_tile_dem`` and
``docs/decisions/unified_10m_grid.md``) -- not just "some 10 m grid", the
DEM's own grid specifically, so alignment is exact rather than merely
same-resolution.

Per-sensor path
----------------
Two paths, not three: this pipeline deliberately never pansharpens, even
though Landsat's TOA collections do carry a 15 m panchromatic band -- every
sensor stays on the one uniform 10 m grid via plain bilinear resample (see
``docs/decisions/unified_10m_grid.md``, ``docs/decisions/toa_rollback.md``):

- **Landsat (L5/L7/L8/L9)**: one 30 m-native band-group fetch, bilinear
  resampled directly onto the DEM's exact 10 m grid. Every Landsat band is
  tagged ``resampled_for_alignment`` -- genuinely ~30 m information smoothed
  onto a finer grid, never true 10 m resolving power.
- **S2**: two band-group fetches (10 m native bands, 20 m bands); the 10 m
  bands are the only ones that can land on the DEM's grid by a pure CRS
  reproject with no resolution change, the 20 m bands are resampled
  (bilinear) onto that same grid.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .band_specs import BandRole, SensorBandSpec, get_band_spec
from .gee_fetch import fetch_band_group, fetch_scene_solar_angles, reflectance_to_uint16, resample_band_to_grid
from .composites import build_rgb_composites
from .harmonize import harmonize_bands
from .sensors import SensorSpec, get_sensor
from .zarr_store import scene_already_stored, write_scene

#: "native" | "resampled_for_alignment" -- no sensor currently writes
#: "pansharpened" or "learned_super_resolved". This is a real change for
#: Landsat 5: under the pre-revision per-sensor-resolution scheme its bands
#: were "native" (already at L5's own 30 m working resolution); on the
#: unified 10 m grid they are genuinely being upsampled to a finer grid than
#: their native pixel size, so they are retagged "resampled_for_alignment".
PROVENANCE_NATIVE = "native"
PROVENANCE_RESAMPLED = "resampled_for_alignment"


def _fetch_bands_at_resolution(
    collection: str,
    scene_id: str,
    band_names: list[str],
    window_bbox: tuple[float, float, float, float],
    crs: str,
    resolution_m: float,
    reflectance_scale_divisor: float = 1.0,
    reflectance_scale_offset: float = 0.0,
) -> tuple[dict[str, np.ndarray], object]:
    """Fetch one group of same-resolution bands, returning ``{name: array}`` + transform.

    ``reflectance = raw / reflectance_scale_divisor + reflectance_scale_offset``
    (see ``SensorSpec.reflectance_scale_divisor``/``reflectance_scale_offset``
    -- under TOA, Sentinel-2 is a pure x10000 divisor with zero offset and
    Landsat needs no conversion at all, both defaults; the scale-and-offset
    form exists for a possible future Surface Reflectance revisit, see
    ``docs/decisions/toa_rollback.md``).
    """
    arr, transform = fetch_band_group(collection, scene_id, band_names, window_bbox, crs, resolution_m)
    if reflectance_scale_divisor != 1.0 or reflectance_scale_offset != 0.0:
        arr = arr / reflectance_scale_divisor + reflectance_scale_offset
    bands = {name: arr[i] for i, name in enumerate(band_names)}
    return bands, transform


def _apply_radiometric_corrections(
    resampled: dict[str, np.ndarray],
    band_spec: SensorBandSpec,
    sensor_key: str,
    harmonization_coefficients: Optional[dict[str, tuple[float, float]]],
    topo_correction: Optional[dict],
) -> dict[str, np.ndarray]:
    """Apply cross-sensor bandpass adjustment then topographic correction,
    both label-keyed (blue/green/red/nir/swir1/swir2) so they work
    regardless of a sensor's raw EE band-name layout. Sequenced after
    reflectance scale/offset conversion and before spectral indices are
    computed downstream, per ``docs/decisions/cross_sensor_harmonization.md``
    / ``docs/decisions/topographic_correction.md``. Sentinel-2 is the
    harmonization target and is never itself bandpass-adjusted.
    Harmonization is disabled by default under TOA (its coefficients were
    derived for Surface Reflectance -- see
    ``docs/decisions/cross_sensor_harmonization.md``); this function still
    honors whatever ``harmonization_coefficients`` it is given, it does not
    itself decide the default.
    """
    label_to_name = {b.label: b.name for b in band_spec.bands}
    name_to_label = {v: k for k, v in label_to_name.items()}
    by_label = {name_to_label[name]: arr for name, arr in resampled.items() if name in name_to_label}

    if harmonization_coefficients and sensor_key.upper() != "S2":
        by_label = harmonize_bands(by_label, harmonization_coefficients)

    if topo_correction is not None:
        from ..features.topographic_correction import topographic_correct_scene

        by_label = topographic_correct_scene(
            by_label,
            slope_deg=topo_correction["slope_deg"],
            aspect_deg=topo_correction["aspect_deg"],
            sun_elevation_deg=topo_correction["sun_elevation_deg"],
            sun_azimuth_deg=topo_correction["sun_azimuth_deg"],
            min_sun_elevation_deg=topo_correction.get("min_sun_elevation_deg", 5.0),
            reference_band=topo_correction.get("reference_band", "nir"),
            ratio_clip_min=topo_correction.get("ratio_clip_min", 0.2),
            ratio_clip_max=topo_correction.get("ratio_clip_max", 5.0),
        )

    return {label_to_name[label]: arr for label, arr in by_label.items()}


def _rgb_input_bands(all_bands_by_name: dict[str, np.ndarray], band_spec: SensorBandSpec) -> dict[str, np.ndarray]:
    """Map ``{ee_band_name: array}`` to ``{label: array}`` for exactly the
    labels :data:`composites.VIEW_BAND_LABELS` can need (red/green/blue/
    nir/swir2) -- whichever of those this sensor actually defines."""
    from .composites import VIEW_BAND_LABELS

    needed_labels = {label for labels in VIEW_BAND_LABELS.values() for label in labels}
    label_to_name = {b.label: b.name for b in band_spec.bands if b.label in needed_labels}
    return {label: all_bands_by_name[name] for label, name in label_to_name.items() if name in all_bands_by_name}


def process_scene_landsat(
    sensor: SensorSpec,
    band_spec: SensorBandSpec,
    scene_id: str,
    window_bbox: tuple[float, float, float, float],
    crs: str,
    dst_transform,
    dst_shape: tuple[int, int],
    harmonization_coefficients: Optional[dict[str, tuple[float, float]]] = None,
    topo_correction: Optional[dict] = None,
    rgb_enabled_views: Optional[set[str]] = None,
    rgb_asinh_k: float = 8.0,
    rgb_gamma: float = 1.0 / 2.2,
) -> tuple[np.ndarray, list[str], dict[str, str], dict[str, np.ndarray], object]:
    """Landsat's shared path (L5/L7/L8/L9): one 30 m-native fetch, bilinear
    resampled directly onto ``dst_transform``/``dst_shape`` -- the tile's DEM
    grid, not just any matching-resolution grid -- then cross-sensor
    bandpass adjustment and topographic correction, both optional."""
    band_names = list(band_spec.toa_band_names())
    bands, src_transform = _fetch_bands_at_resolution(
        sensor.collection, scene_id, band_names, window_bbox, crs, 30.0,
        reflectance_scale_divisor=sensor.reflectance_scale_divisor,
        reflectance_scale_offset=sensor.reflectance_scale_offset,
    )

    resampled = {
        name: resample_band_to_grid(arr, src_transform, dst_transform, dst_shape, crs, method="bilinear")
        for name, arr in bands.items()
    }
    resampled = _apply_radiometric_corrections(
        resampled, band_spec, sensor.key, harmonization_coefficients, topo_correction
    )

    provenance = {n: PROVENANCE_RESAMPLED for n in band_names}
    toa_float = np.stack([resampled[n] for n in band_names], axis=0)
    toa = reflectance_to_uint16(toa_float)

    rgb_composites = build_rgb_composites(
        _rgb_input_bands(resampled, band_spec), rgb_enabled_views or set(), asinh_k=rgb_asinh_k, gamma=rgb_gamma
    )
    return toa, band_names, provenance, rgb_composites, dst_transform


def process_scene_s2(
    sensor: SensorSpec,
    band_spec: SensorBandSpec,
    scene_id: str,
    window_bbox: tuple[float, float, float, float],
    crs: str,
    dst_transform,
    dst_shape: tuple[int, int],
    topo_correction: Optional[dict] = None,
    rgb_enabled_views: Optional[set[str]] = None,
    rgb_asinh_k: float = 8.0,
    rgb_gamma: float = 1.0 / 2.2,
) -> tuple[np.ndarray, list[str], dict[str, str], dict[str, np.ndarray], object]:
    """Sentinel-2's path: native 10 m bands + 20 m bands, both reprojected
    onto ``dst_transform``/``dst_shape`` -- the tile's DEM grid -- then
    topographic correction (optional). Sentinel-2 is never cross-sensor
    bandpass-adjusted: it is this pipeline's harmonization target."""
    native_names = [b.name for b in band_spec.bands if b.role == BandRole.NATIVE]
    resample_names = list(band_spec.resample_band_names())

    native_bands, native_transform = _fetch_bands_at_resolution(
        sensor.collection, scene_id, native_names, window_bbox, crs, band_spec.working_resolution_m,
        reflectance_scale_divisor=sensor.reflectance_scale_divisor,
        reflectance_scale_offset=sensor.reflectance_scale_offset,
    )
    coarse_bands, coarse_transform = _fetch_bands_at_resolution(
        sensor.collection, scene_id, resample_names, window_bbox, crs, 20.0,
        reflectance_scale_divisor=sensor.reflectance_scale_divisor,
        reflectance_scale_offset=sensor.reflectance_scale_offset,
    )

    native_on_grid = {
        name: resample_band_to_grid(arr, native_transform, dst_transform, dst_shape, crs, method="bilinear")
        for name, arr in native_bands.items()
    }
    resampled = {
        name: resample_band_to_grid(arr, coarse_transform, dst_transform, dst_shape, crs, method="bilinear")
        for name, arr in coarse_bands.items()
    }

    provenance = {n: PROVENANCE_NATIVE for n in native_names}
    provenance.update({n: PROVENANCE_RESAMPLED for n in resample_names})

    band_names = list(band_spec.toa_band_names())
    all_bands = {**native_on_grid, **resampled}
    all_bands = _apply_radiometric_corrections(all_bands, band_spec, sensor.key, None, topo_correction)
    toa_float = np.stack([all_bands[n] for n in band_names], axis=0)
    toa = reflectance_to_uint16(toa_float)

    rgb_composites = build_rgb_composites(
        _rgb_input_bands(all_bands, band_spec), rgb_enabled_views or set(), asinh_k=rgb_asinh_k, gamma=rgb_gamma
    )
    return toa, band_names, provenance, rgb_composites, dst_transform


def process_and_store_scene(
    tile_dir,
    tile_id: str,
    sensor_key: str,
    scene_id: str,
    window_bbox: tuple[float, float, float, float],
    crs: str,
    skip_existing: bool = True,
    harmonization_coefficients: Optional[dict[str, tuple[float, float]]] = None,
    topo_correction_enabled: bool = False,
    topo_correction_min_sun_elevation_deg: float = 5.0,
    topo_correction_reference_band: str = "nir",
    topo_correction_ratio_clip_min: float = 0.2,
    topo_correction_ratio_clip_max: float = 5.0,
    rgb_enabled_views: Optional[set[str]] = None,
    rgb_asinh_k: float = 8.0,
    rgb_gamma: float = 1.0 / 2.2,
) -> str:
    """Fetch, resample onto the DEM's grid, and append one scene into the
    tile's zarr store.

    Returns a short status string (``"ok"``, ``"skip (already stored)"``, or
    ``"error (...)"``); never raises, so a sweep over many scenes is not
    aborted by one bad fetch (mirrors ``dem/engine.py::process_one_tile``'s
    per-item error containment).

    Reads the tile's DEM transform/shape (``dem/zarr_store.py::read_tile_dem``)
    as the resample target, then checks the freshly built scene against it
    via ``dem/grid_check.py::check_grid_alignment`` before writing -- a
    mismatch is a hard failure (this function's own ``"error (...)"``
    status), never a silent pass-through. The DEM must already be stored for
    this tile (run Stage 2 first); a missing DEM group is itself reported as
    an error rather than falling back to some other grid.

    ``harmonization_coefficients`` (normally
    ``config.harmonization.coefficients[sensor_key]``, ``None``/empty to
    disable) cross-sensor bandpass-adjusts Landsat bands onto Sentinel-2's
    convention; ignored for S2 itself. ``topo_correction_enabled`` applies
    SCS+C topographic correction using the tile's own slope/aspect (also
    read from the DEM store) plus this scene's fetched solar geometry.
    ``rgb_enabled_views`` (normally derived from
    ``config.rgb_composites``, see :mod:`.composites`) selects which of the
    four RGB visualization composites actually get built and stored --
    ``None``/empty builds none.
    """
    if skip_existing and scene_already_stored(tile_dir, tile_id, sensor_key, scene_id):
        return "skip (already stored)"

    try:
        from ..dem.grid_check import check_grid_alignment
        from ..dem.zarr_store import read_tile_dem

        sensor = get_sensor(sensor_key)
        band_spec = get_band_spec(sensor_key)

        dem_arr, dem_transform, dem_crs_wkt = read_tile_dem(tile_dir, tile_id, "elevation")
        dst_shape = tuple(dem_arr.shape)

        topo_correction = None
        if topo_correction_enabled:
            slope_arr, _, _ = read_tile_dem(tile_dir, tile_id, "slope")
            aspect_arr, _, _ = read_tile_dem(tile_dir, tile_id, "aspect")
            solar_angles = fetch_scene_solar_angles(
                sensor.collection, scene_id, is_sentinel=sensor.is_sentinel
            )
            topo_correction = {
                "slope_deg": slope_arr,
                "aspect_deg": aspect_arr,
                "sun_elevation_deg": solar_angles["sun_elevation_deg"],
                "sun_azimuth_deg": solar_angles["sun_azimuth_deg"],
                "min_sun_elevation_deg": topo_correction_min_sun_elevation_deg,
                "reference_band": topo_correction_reference_band,
                "ratio_clip_min": topo_correction_ratio_clip_min,
                "ratio_clip_max": topo_correction_ratio_clip_max,
            }

        if sensor_key.upper() == "S2":
            toa, band_names, provenance, rgb_composites, transform = process_scene_s2(
                sensor, band_spec, scene_id, window_bbox, crs, dem_transform, dst_shape,
                topo_correction=topo_correction,
                rgb_enabled_views=rgb_enabled_views,
                rgb_asinh_k=rgb_asinh_k,
                rgb_gamma=rgb_gamma,
            )
        else:
            toa, band_names, provenance, rgb_composites, transform = process_scene_landsat(
                sensor, band_spec, scene_id, window_bbox, crs, dem_transform, dst_shape,
                harmonization_coefficients=harmonization_coefficients,
                topo_correction=topo_correction,
                rgb_enabled_views=rgb_enabled_views,
                rgb_asinh_k=rgb_asinh_k,
                rgb_gamma=rgb_gamma,
            )

        grid_result = check_grid_alignment(
            tile_id,
            tile_dir,
            other_arrays={f"{sensor_key}/{scene_id}": (toa.shape[-2:], transform, crs)},
        )
        if not grid_result.ok:
            return f"error (grid mismatch against DEM: {grid_result.report()})"

        write_scene(
            tile_dir,
            tile_id,
            sensor_key,
            scene_id,
            toa,
            band_names,
            provenance,
            rgb_composites,
            transform,
            crs_wkt=crs,
            skip_existing=skip_existing,
        )
        return "ok"
    except Exception as exc:  # noqa: BLE001 -- reported per scene, never fatal
        return f"error ({type(exc).__name__}: {exc})"
