"""Per-tile scene storage: one zarr group per sensor, appended to over scenes.

Purpose
-------
Persist each fetched/processed scene into ``<tile_id>.zarr/<sensor>`` --
the same per-tile zarr store :mod:`landscape_change_detection_pipeline.dem.zarr_store`
writes ``dem`` into. One group per sensor (``l5``, ``l7``, ``l8``, ``l9``,
``s2``), kept fully separate -- L8/L9 are never merged into one group:
confusing which physical sensor produced a scene must not be
possible.

Layout
------
::

    <tile_id>.zarr/
        <sensor>/                      (l5, l7, l8, l9, or s2)
            toa         (n_scenes, n_bands, H, W)  uint16  nodata=65535
            rgb_raw     (n_scenes, 3, H, W)         uint8
            rgb_shadow  (n_scenes, 3, H, W)         uint8
            .attrs:
                scene_ids: [str, ...]          -- index into the scene axis
                band_names: [str, ...]         -- index into the band axis
                band_provenance: {band: "native"|"pansharpened"|"resampled_for_alignment"}
                crs_wkt, transform, scale_factor, nodata_value

One chunk per scene (``chunks=(1, n_bands, H, W)`` / ``(1, 3, H, W)``), so
appending a new scene never rewrites existing chunks. TOA uses the
zstd/bitshuffle profile (integer reflectance data benefits from bitshuffle);
RGB uses lz4 (already-quantized 8-bit display data, where zstd's extra ratio
is not worth its slower decode for a repeatedly-viewed QA image).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

_TOA_ARRAY = "toa"
_RGB_RAW_ARRAY = "rgb_raw"
_RGB_SHADOW_ARRAY = "rgb_shadow"

#: One lock per (zarr path, sensor group) pair, so concurrent appends to
#: *different* tiles/sensors never block each other, while two scenes for the
#: same tile/sensor append safely one at a time.
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(zarr_path: Path, sensor: str) -> threading.Lock:
    key = f"{zarr_path}::{sensor}"
    with _locks_guard:
        if key not in _locks:
            _locks[key] = threading.Lock()
        return _locks[key]


def zarr_path_for_tile(tile_dir: str | Path, tile_id: str) -> Path:
    """The zarr store path for one tile: ``<tile_dir>/<tile_id>.zarr``."""
    return Path(tile_dir) / f"{tile_id}.zarr"


def _toa_compressor():
    from numcodecs import Blosc

    return Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)


def _rgb_compressor():
    from numcodecs import Blosc

    return Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)


def transform_to_list(transform) -> list[float]:
    return [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f]


def list_to_transform(values: list[float]):
    from affine import Affine

    return Affine(*values[:6])


def scene_already_stored(tile_dir: str | Path, tile_id: str, sensor: str, scene_id: str) -> bool:
    """Whether ``scene_id`` is already appended to this tile/sensor's group.

    The idempotency check backing ``skip_existing`` -- reading only the
    group's ``scene_ids`` attr, never opening/decompressing array data.
    """
    import zarr

    zarr_path = zarr_path_for_tile(tile_dir, tile_id)
    if not zarr_path.exists():
        return False
    try:
        store = zarr.open_group(str(zarr_path), mode="r")
    except Exception:  # noqa: BLE001 -- a partially written store must not crash discovery
        return False
    if sensor not in store:
        return False
    return scene_id in list(store[sensor].attrs.get("scene_ids", []))


def write_scene(
    tile_dir: str | Path,
    tile_id: str,
    sensor: str,
    scene_id: str,
    toa: np.ndarray,
    band_names: list[str],
    band_provenance: dict[str, str],
    rgb_raw: np.ndarray,
    rgb_shadow: np.ndarray,
    transform,
    crs_wkt: str,
    skip_existing: bool = True,
    scene_attrs: Optional[dict] = None,
) -> Path:
    """Append one scene into ``<tile_id>.zarr/<sensor>``.

    Parameters
    ----------
    toa
        ``(n_bands, H, W)`` uint16, nodata=65535 -- one uniform-resolution
        array combining every band of this sensor's group (pansharpened,
        resampled, and native bands alike; see ``band_provenance``).
    rgb_raw, rgb_shadow
        ``(3, H, W)`` uint8 visualization composites.
    skip_existing
        If ``True`` (default) and ``scene_id`` is already present in this
        tile/sensor's group, this is a no-op -- the idempotent-append
        contract a resumed run after a partial failure relies on.
    """
    import zarr

    tile_dir = Path(tile_dir)
    zarr_path = zarr_path_for_tile(tile_dir, tile_id)
    lock = _lock_for(zarr_path, sensor)

    with lock:
        if skip_existing and scene_already_stored(tile_dir, tile_id, sensor, scene_id):
            return zarr_path

        tile_dir.mkdir(parents=True, exist_ok=True)
        store = zarr.open_group(str(zarr_path), mode="a")
        grp = store.require_group(sensor)

        n_bands, height, width = toa.shape
        _append_to_array(
            grp, _TOA_ARRAY, toa[np.newaxis, ...], chunks=(1, n_bands, height, width), compressor=_toa_compressor()
        )
        _append_to_array(
            grp, _RGB_RAW_ARRAY, rgb_raw[np.newaxis, ...], chunks=(1, 3, height, width), compressor=_rgb_compressor()
        )
        _append_to_array(
            grp,
            _RGB_SHADOW_ARRAY,
            rgb_shadow[np.newaxis, ...],
            chunks=(1, 3, height, width),
            compressor=_rgb_compressor(),
        )

        attrs = dict(grp.attrs)
        scene_ids = list(attrs.get("scene_ids", []))
        scene_ids.append(scene_id)
        attrs["scene_ids"] = scene_ids
        attrs["band_names"] = list(band_names)
        attrs["band_provenance"] = band_provenance
        attrs["crs_wkt"] = str(crs_wkt)
        attrs["transform"] = transform_to_list(transform)
        attrs["nodata_value"] = 65535
        attrs["scale_factor"] = 10000.0
        if scene_attrs:
            per_scene = dict(attrs.get("scene_attrs", {}))
            per_scene[scene_id] = scene_attrs
            attrs["scene_attrs"] = per_scene
        attrs["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        grp.attrs.update(attrs)

    return zarr_path


def _append_to_array(grp, name: str, new_slice: np.ndarray, chunks: tuple, compressor) -> None:
    """Append one scene's slice along axis 0, creating the array on first write."""
    if name not in grp:
        grp.create_dataset(
            name,
            data=new_slice,
            chunks=chunks,
            compressor=compressor,
        )
        return

    arr = grp[name]
    if arr.shape[1:] != new_slice.shape[1:]:
        raise ValueError(
            f"Grid mismatch appending to '{name}': existing {arr.shape[1:]} vs new {new_slice.shape[1:]}. "
            "Every scene of a tile/sensor must share the same grid (see dem/grid_check.py's invariant)."
        )
    arr.resize(arr.shape[0] + 1, *arr.shape[1:])
    arr[-1:] = new_slice


def read_scene(
    tile_dir: str | Path, tile_id: str, sensor: str, scene_id: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one scene's ``(toa, rgb_raw, rgb_shadow)`` back by scene id."""
    import zarr

    zarr_path = zarr_path_for_tile(tile_dir, tile_id)
    store = zarr.open_group(str(zarr_path), mode="r")
    if sensor not in store:
        raise KeyError(f"No '{sensor}' group in {zarr_path}")
    grp = store[sensor]
    scene_ids = list(grp.attrs.get("scene_ids", []))
    if scene_id not in scene_ids:
        raise KeyError(f"Scene '{scene_id}' not found in {zarr_path}/{sensor}")
    idx = scene_ids.index(scene_id)
    return (
        np.asarray(grp[_TOA_ARRAY][idx]),
        np.asarray(grp[_RGB_RAW_ARRAY][idx]),
        np.asarray(grp[_RGB_SHADOW_ARRAY][idx]),
    )


def read_sensor_group_attrs(tile_dir: str | Path, tile_id: str, sensor: str) -> dict:
    """Read a sensor group's full attrs dict."""
    import zarr

    zarr_path = zarr_path_for_tile(tile_dir, tile_id)
    store = zarr.open_group(str(zarr_path), mode="r")
    if sensor not in store:
        raise KeyError(f"No '{sensor}' group in {zarr_path}")
    return dict(store[sensor].attrs)


def sensor_group_grid_shape(tile_dir: str | Path, tile_id: str, sensor: str) -> tuple[int, int]:
    """The ``(H, W)`` shape of a sensor group's grid, for the DEM grid-check invariant."""
    import zarr

    zarr_path = zarr_path_for_tile(tile_dir, tile_id)
    store = zarr.open_group(str(zarr_path), mode="r")
    if sensor not in store or _TOA_ARRAY not in store[sensor]:
        raise KeyError(f"No '{sensor}/{_TOA_ARRAY}' array in {zarr_path}")
    shape = store[sensor][_TOA_ARRAY].shape
    return (shape[-2], shape[-1])
