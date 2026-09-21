"""Per-tile DEM storage in the tile's zarr store.

Purpose
-------
Persist elevation, slope, and aspect into ``<tile_id>.zarr/dem`` -- the same
per-tile store later stages (scene fetch, feature stacks) write into under
their own top-level groups. The array-layout convention is one array per
product, no scene axis, create-or-overwrite rather than append.

Layout
------
::

    <tile_id>.zarr/
        dem/
            elevation   (H, W)  float32   NaN nodata
            slope       (H, W)  float32   NaN nodata, degrees
            aspect      (H, W)  float32   NaN nodata, degrees (0=N, clockwise)
            .attrs: crs_wkt, transform (6-element affine list),
                    source_dem, source_base_resolution_m,
                    source_window_valid_pct, window_bounds, generated_at
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

_ARRAY_NAMES = ("elevation", "slope", "aspect")


def zarr_path_for_tile(tile_dir: str | Path, tile_id: str) -> Path:
    """The zarr store path for one tile: ``<tile_dir>/<tile_id>.zarr``."""
    return Path(tile_dir) / f"{tile_id}.zarr"


def transform_to_list(transform) -> list[float]:
    """Affine transform -> a flat 6-element list, JSON/zarr-attrs safe."""
    return [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f]


def list_to_transform(values: list[float]):
    """Inverse of :func:`transform_to_list`."""
    from affine import Affine

    return Affine(*values[:6])


def _compressor():
    from numcodecs import Blosc

    return Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)


def write_tile_dem(
    tile_dir: str | Path,
    tile_id: str,
    products: dict[str, np.ndarray],
    transform,
    crs_wkt: str,
    attrs: Optional[dict] = None,
) -> Path:
    """Write elevation/slope/aspect into ``<tile_id>.zarr/dem``.

    Parameters
    ----------
    products
        ``{"elevation": arr, "slope": arr, "aspect": arr}``, float32 with NaN
        nodata, all sharing ``transform``. Any subset of the three keys may
        be given.
    """
    import zarr

    tile_dir = Path(tile_dir)
    zarr_path = zarr_path_for_tile(tile_dir, tile_id)

    tile_dir.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(zarr_path), mode="a")
    grp = store.require_group("dem")

    compressor = _compressor()
    group_attrs: dict = dict(grp.attrs)
    group_attrs["crs_wkt"] = str(crs_wkt)
    group_attrs["transform"] = transform_to_list(transform)

    for name, array in products.items():
        if name not in _ARRAY_NAMES:
            raise ValueError(f"Unsupported DEM product '{name}'; expected one of {_ARRAY_NAMES}")
        data = np.asarray(array, dtype=np.float32)
        if name in grp:
            del grp[name]
        grp.create_dataset(name, data=data, chunks=data.shape, compressor=compressor)

    if attrs:
        group_attrs.update(attrs)
    group_attrs["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    grp.attrs.update(group_attrs)

    return zarr_path


def read_tile_dem(
    tile_dir: str | Path, tile_id: str, product: str = "elevation"
) -> tuple[np.ndarray, object, str]:
    """Read one product back as ``(array, transform, crs_wkt)``."""
    import zarr

    if product not in _ARRAY_NAMES:
        raise ValueError(f"Unsupported DEM product '{product}'; expected one of {_ARRAY_NAMES}")

    zarr_path = zarr_path_for_tile(tile_dir, tile_id)
    store = zarr.open_group(str(zarr_path), mode="r")
    if "dem" not in store:
        raise KeyError(f"No 'dem' group in {zarr_path}")
    grp = store["dem"]
    if product not in grp:
        raise KeyError(f"'{product}' is absent from {zarr_path}/dem")

    attrs = dict(grp.attrs)
    transform_values = attrs.get("transform")
    transform = list_to_transform(transform_values) if transform_values else None
    return np.asarray(grp[product][:], dtype=np.float32), transform, str(attrs.get("crs_wkt", ""))


def read_tile_dem_attrs(tile_dir: str | Path, tile_id: str) -> dict:
    """Read the ``dem`` group's full attrs dict (provenance/coverage metadata)."""
    import zarr

    zarr_path = zarr_path_for_tile(tile_dir, tile_id)
    store = zarr.open_group(str(zarr_path), mode="r")
    if "dem" not in store:
        raise KeyError(f"No 'dem' group in {zarr_path}")
    return dict(store["dem"].attrs)


def tile_dem_grid_shape(tile_dir: str | Path, tile_id: str) -> tuple[int, int]:
    """The ``(H, W)`` shape of the tile's DEM grid, for :mod:`.grid_check`."""
    import zarr

    zarr_path = zarr_path_for_tile(tile_dir, tile_id)
    store = zarr.open_group(str(zarr_path), mode="r")
    if "dem" not in store or "elevation" not in store["dem"]:
        raise KeyError(f"No 'dem/elevation' array in {zarr_path}")
    return tuple(store["dem"]["elevation"].shape)  # type: ignore[return-value]


def dem_group_exists(tile_dir: str | Path, tile_id: str) -> bool:
    """Whether ``<tile_id>.zarr/dem`` already exists (the resume/skip flag)."""
    import zarr

    zarr_path = zarr_path_for_tile(tile_dir, tile_id)
    if not zarr_path.exists():
        return False
    try:
        store = zarr.open_group(str(zarr_path), mode="r")
    except Exception:  # noqa: BLE001 -- a partially written store must not crash discovery
        return False
    return "dem" in store
