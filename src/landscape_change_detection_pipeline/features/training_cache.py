"""Annotation (MaskForge mask) -> per-scene training-cache ``.npz`` export.

Purpose
-------
For every mask MaskForge writes back, assemble a ``(C, H, W)``
feature stack (spectral indices + DEM layers) aligned to that scene's grid,
and cache it as a per-scene ``.npz`` (``features`` float16, ``labels`` uint8)
plus a JSON sidecar recording the cache's invalidation signature, using a
``scene_cache_signature()``/``cache_is_current()`` never-stale-cache contract.

Mask discovery
--------------
MaskForge saves annotated masks as RGBA GeoTIFFs at
``<mask_root>/{scene_key}/mask.tif`` (``SaveConfig.folder_structure_template``,
default ``"{scene_id}/mask.tif"``), where ``scene_key`` is
``"{tile_id}_{sensor}_{scene_id}"`` (see
``Maskforge/sidecar/maskforge_core/scene_discovery.py::_scan_zarr_store``'s
``SceneEntry.id`` construction -- this is the only identity string that
survives the round trip through MaskForge, so it is the key this module
parses back apart). The RGBA pixels are converted back to a class-index
array via the same ``rgba_to_classes`` convention MaskForge itself uses:
each land-cover class's ``configs/classes.yaml`` color maps to its integer
``id``; alpha==255 & RGB==(0,0,0) is nodata (class id is reserved as 0, see
``classes/class_config.py``).

Reproducibility discipline
---------------------------
Never rely on
``os.walk``/``glob``/``set``/``dict`` iteration order for anything that
drives a decision. Every directory scan here is followed by an explicit
sort; ``scene_cache_signature()`` hashes ``sorted(source_paths)``, never the
caller's original (unordered) sequence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from landscape_change_detection_pipeline.classes.class_config import ClassConfig
from landscape_change_detection_pipeline.dem.zarr_store import read_tile_dem
from landscape_change_detection_pipeline.features.spectral_indices import (
    DEM_LAYER_NAMES,
    INDEX_NAMES,
    compute_indices,
)
from landscape_change_detection_pipeline.scenes.band_specs import get_band_spec
from landscape_change_detection_pipeline.scenes.gee_fetch import uint16_to_reflectance
from landscape_change_detection_pipeline.scenes.zarr_store import (
    list_to_transform,
    read_scene,
    read_sensor_group_attrs,
    transform_to_list,
    zarr_path_for_tile,
)

#: Bumped whenever the .npz schema changes, so scene_cache_signature's hash
#: (which includes schema_version) forces a rebuild of every stale cache
#: instead of leaving old caches silently missing a newly-changed field.
#: Bumped here because existing caches may have unpainted pixels baked in
#: as class id 0 from before read_mask_labels wrote NODATA_LABEL (255) for
#: them -- those caches must be rebuilt, not reused as-is.
SCHEMA_VERSION = 2
CACHE_FILENAME = "features.npz"
META_FILENAME = "features.json"

#: Landsat scene ids end in an 8-digit acquisition date, e.g.
#: "LT05_048021_20100725" -> year 2010.
_LANDSAT_DATE_RE = re.compile(r"_(\d{4})(\d{2})(\d{2})$")
#: Sentinel-2 scene ids start with a 15-char acquisition timestamp, e.g.
#: "20230607T191911_20230607T192455_T10VEH" -> year 2023.
_S2_DATE_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})T\d{6}")


class MaskDiscoveryError(Exception):
    """Raised when a mask file cannot be located or parsed for a given scene key."""


@dataclass(frozen=True)
class AnnotatedScene:
    """One MaskForge-annotated scene, identified unambiguously by the
    ``(tile_id, sensor, scene_id)`` triple MaskForge's scene_key encodes."""

    tile_id: str
    sensor: str
    scene_id: str
    mask_path: Path

    @property
    def scene_key(self) -> str:
        return f"{self.tile_id}_{self.sensor}_{self.scene_id}"

    @property
    def year(self) -> int:
        return scene_year(self.sensor, self.scene_id)

    @property
    def sort_key(self) -> tuple[str, str, str]:
        return (self.tile_id, self.sensor, self.scene_id)


def scene_year(sensor: str, scene_id: str) -> int:
    """Parse the acquisition year out of a scene id.

    Landsat scene ids end in an 8-digit date; Sentinel-2 ids start with one.
    There is no single fixed position across sensors, so both patterns are
    tried explicitly rather than assuming one format.
    """
    if sensor.upper() == "S2":
        match = _S2_DATE_RE.match(scene_id)
    else:
        match = _LANDSAT_DATE_RE.search(scene_id)
    if not match:
        raise ValueError(f"Could not parse an acquisition year from scene_id '{scene_id}' (sensor '{sensor}').")
    return int(match.group(1))


def scene_date(sensor: str, scene_id: str) -> "date":
    """Parse the full acquisition date (year, month, day) out of a scene id,
    using the same two id formats :func:`scene_year` already parses.
    Needed wherever a per-pixel time series must be ordered/compared by real
    calendar date, not just by year (e.g. :mod:`change.ccdc`'s ordinal-day
    time axis)."""
    from datetime import date as _date

    if sensor.upper() == "S2":
        match = _S2_DATE_RE.match(scene_id)
    else:
        match = _LANDSAT_DATE_RE.search(scene_id)
    if not match:
        raise ValueError(f"Could not parse an acquisition date from scene_id '{scene_id}' (sensor '{sensor}').")
    year, month, day = (int(g) for g in match.groups())
    return _date(year, month, day)


def parse_scene_key(scene_key: str, known_sensors: Sequence[str] = ("l5", "l7", "l8", "l9", "s2")) -> tuple[str, str, str]:
    """Split a MaskForge scene_key ``"{tile_id}_{sensor}_{scene_id}"`` back
    into ``(tile_id, sensor, scene_id)``.

    The sensor segment is matched case-insensitively against
    ``known_sensors`` (rather than just splitting on ``_`` positionally),
    since both ``tile_id`` (``tile_0031_0019``) and ``scene_id``
    (``LT05_048021_20100725``) themselves contain underscores.
    """
    parts = scene_key.split("_")
    for i in range(1, len(parts) - 1):
        candidate = parts[i].lower()
        if candidate in known_sensors:
            tile_id = "_".join(parts[:i])
            sensor = candidate
            scene_id = "_".join(parts[i + 1 :])
            if tile_id and scene_id:
                return tile_id, sensor, scene_id
    raise MaskDiscoveryError(
        f"Could not parse a (tile_id, sensor, scene_id) triple out of scene_key '{scene_key}' "
        f"(expected one of {known_sensors} as a middle segment)."
    )


def discover_annotated_scenes(mask_root: str | Path) -> list[AnnotatedScene]:
    """Find every MaskForge-annotated scene under ``mask_root``, sorted by
    ``(tile_id, sensor, scene_id)``.

    Looks for ``<mask_root>/<scene_key>/mask.tif`` (also accepts
    ``mask.png``, MaskForge's other supported extension), matching
    ``SaveConfig.folder_structure_template``'s default
    ``"{scene_id}/mask.tif"`` (MaskForge's "scene_id" there is this module's
    scene_key). ``os.walk``'s directory order is not guaranteed, so the
    result is always sorted before returning.
    """
    mask_root = Path(mask_root)
    scenes: list[AnnotatedScene] = []
    if not mask_root.is_dir():
        return scenes

    for entry in sorted(os.listdir(mask_root)):
        scene_dir = mask_root / entry
        if not scene_dir.is_dir():
            continue
        mask_path = None
        for candidate_name in ("mask.tif", "mask.tiff", "mask.png"):
            candidate = scene_dir / candidate_name
            if candidate.is_file():
                mask_path = candidate
                break
        if mask_path is None:
            continue
        try:
            tile_id, sensor, scene_id = parse_scene_key(entry)
        except MaskDiscoveryError:
            continue
        scenes.append(AnnotatedScene(tile_id=tile_id, sensor=sensor, scene_id=scene_id, mask_path=mask_path))

    scenes.sort(key=lambda s: s.sort_key)
    return scenes


#: Label value marking a no-data / unannotated pixel -- matches both
#: MaskForge's own ``raster_io.NODATA_VALUE`` (alpha == 0 in the saved mask)
#: and the training loop's ``training.losses.IGNORE_INDEX``, which already
#: excludes this exact value from every loss/metric. This module only needs
#: to write it correctly; nothing downstream needs to change.
NODATA_LABEL = 255


def read_mask_labels(mask_path: str | Path, class_config: ClassConfig) -> np.ndarray:
    """Read a MaskForge RGBA mask GeoTIFF/PNG and convert it to a
    ``(H, W)`` uint8 class-index array, using the same color -> id mapping
    ``configs/classes.yaml`` defines (mirrors MaskForge's own
    ``raster_io.rgba_to_classes``).

    MaskForge's own nodata convention is alpha == 0 (RGB forced to
    (0, 0, 0)) for any pixel the annotator never painted -- NOT a specific
    RGB color, and NOT class id 0. Unmatched pixels are written as
    NODATA_LABEL (255), not 0, or every unpainted pixel would silently train
    as whatever class happens to have id 0 (here: forest). 255 is also
    already ``training.losses.IGNORE_INDEX``, so the training loop excludes
    these pixels from the loss/metrics with no further changes needed.
    """
    import rasterio

    with rasterio.open(mask_path) as src:
        rgba = src.read()  # (4, H, W)
    rgba = np.moveaxis(rgba, 0, -1)  # -> (H, W, 4)

    h, w = rgba.shape[:2]
    labels = np.full((h, w), NODATA_LABEL, dtype=np.uint8)
    for class_def in class_config.classes:
        color = class_def.color
        match = (
            (rgba[:, :, 0] == color[0])
            & (rgba[:, :, 1] == color[1])
            & (rgba[:, :, 2] == color[2])
            & (rgba[:, :, 3] == 255)
        )
        labels[match] = class_def.id
    return labels


def bands_for_scene(tile_dir: str | Path, tile_id: str, sensor: str, scene_id: str) -> tuple[dict, dict]:
    """Read a stored scene's TOA array back as a ``{label: reflectance_array}``
    dict plus its ``{label: provenance}`` dict, keyed by the same band labels
    :mod:`features.spectral_indices` expects (blue/green/red/nir/swir1/swir2)."""
    sensor_upper = sensor.upper()
    spec = get_band_spec(sensor_upper)
    attrs = read_sensor_group_attrs(tile_dir, tile_id, sensor)
    band_names: list[str] = list(attrs["band_names"])
    band_provenance_raw: dict[str, str] = dict(attrs.get("band_provenance", {}))

    toa, _rgb_raw, _rgb_shadow = read_scene(tile_dir, tile_id, sensor, scene_id)
    reflectance = uint16_to_reflectance(toa)

    #: BandRole.PANSHARPEN maps to the "not genuine detail at the
    #: working resolution" bucket the same way BandRole.RESAMPLE does --
    #: only truly-native bands are exempt.
    role_to_provenance = {
        "native": "native",
        "pansharpen": "pansharpened",
        "resample": "resampled_for_alignment",
    }

    bands: dict[str, np.ndarray] = {}
    provenance: dict[str, str] = {}
    for band_name in band_names:
        band_def = spec.band(band_name)
        idx = band_names.index(band_name)
        bands[band_def.label] = reflectance[idx]
        stored = band_provenance_raw.get(band_name)
        provenance[band_def.label] = stored if stored else role_to_provenance[band_def.role.value]

    return bands, provenance


def _resample_dem_to_sensor_grid(
    dem_arr: np.ndarray,
    dem_transform,
    dem_crs: str,
    dst_shape: tuple[int, int],
    dst_transform,
    dst_crs: str,
) -> np.ndarray:
    """Reproject one ``(H, W)`` DEM layer from the DEM's own 30 m grid onto a
    sensor's working-resolution grid (30 m for L5, 15 m for L7/L8/L9, 10 m
    for S2 -- see ``scenes/band_specs.py``).

    The DEM is fetched once per tile at its source's native resolution (30 m
    MRDEM-30/Copernicus, see ``dem/engine.py``) and never rescaled at fetch
    time; every later feature stack must land on *that scene's* grid, not
    the DEM's. Bilinear resampling: elevation/slope are continuous surfaces,
    and every other continuous-valued raster this pipeline resamples
    (Landsat 7/8/9 pansharpen-alignment bands) already uses bilinear (see
    ``scenes/process_scene.py``). Aspect (circular, degrees) is a known
    exception -- averaging angles that straddle the 0/360 wrap gives a wrong
    result -- but is accepted here as a documented limitation
    rather than adding a sin/cos-pair resampling path before any real use
    case has exercised it.
    """
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    destination = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=dem_arr.astype(np.float32),
        destination=destination,
        src_transform=dem_transform if isinstance(dem_transform, Affine) else Affine(*dem_transform[:6]),
        src_crs=CRS.from_user_input(dem_crs),
        dst_transform=dst_transform if isinstance(dst_transform, Affine) else Affine(*dst_transform[:6]),
        dst_crs=CRS.from_user_input(dst_crs),
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return destination


def build_feature_stack(
    tile_dir: str | Path,
    tile_id: str,
    sensor: str,
    scene_id: str,
    index_names: tuple[str, ...] = INDEX_NAMES,
    dem_layer_names: tuple[str, ...] = DEM_LAYER_NAMES,
) -> tuple[np.ndarray, list[str], list[str], list[float], str]:
    """Assemble the ``(C, H, W)`` feature stack for one scene: spectral
    indices followed by DEM layers, in that fixed order. Returns
    ``(stack, feature_names, feature_provenance, transform, crs_wkt)``, where
    ``transform``/``crs_wkt`` are the scene's own working-resolution grid
    georeferencing (see :func:`landscape_change_detection_pipeline.scenes.zarr_store.transform_to_list`),
    needed so a cached scene can later be written back out as a
    georeferenced raster without reopening the source zarr store.

    DEM layers are resampled onto the scene's own working-resolution grid
    (see :func:`_resample_dem_to_sensor_grid`) before concatenation -- the
    DEM is fetched once per tile at its source's native ~30 m resolution
    (:mod:`landscape_change_detection_pipeline.dem.engine`), which only coincides with
    Landsat 5's 30 m working grid; every other sensor (L7/L8/L9 at 15 m, S2
    at 10 m) needs the DEM regridded first or this concatenation raises a
    shape mismatch.
    """
    bands, band_provenance = bands_for_scene(tile_dir, tile_id, sensor, scene_id)
    index_stack, index_provenance = compute_indices(bands, band_provenance, names=index_names)
    dst_shape = index_stack.shape[1:]

    attrs = read_sensor_group_attrs(tile_dir, tile_id, sensor)
    dst_transform = attrs["transform"]
    dst_crs = attrs["crs_wkt"]

    dem_arrays = []
    dem_provenance = []
    for name in dem_layer_names:
        arr, dem_transform, dem_crs = read_tile_dem(tile_dir, tile_id, name)
        if tuple(arr.shape) != tuple(dst_shape):
            arr = _resample_dem_to_sensor_grid(arr, dem_transform, dem_crs, dst_shape, dst_transform, dst_crs)
            dem_provenance.append("resampled_for_alignment")
        else:
            dem_provenance.append("native")
        dem_arrays.append(arr)
    dem_stack = np.stack(dem_arrays, axis=0).astype(np.float32) if dem_arrays else np.zeros(
        (0, *index_stack.shape[1:]), dtype=np.float32
    )

    stack = np.concatenate([index_stack, dem_stack], axis=0).astype(np.float32)
    feature_names = [*index_names, *dem_layer_names]
    feature_provenance = [*index_provenance, *dem_provenance]
    transform_list = dst_transform if isinstance(dst_transform, list) else transform_to_list(dst_transform)
    return stack, feature_names, feature_provenance, transform_list, str(dst_crs)


def scene_source_paths(tile_dir: str | Path, tile_id: str, mask_path: str | Path) -> list[Path]:
    """The files whose (size, mtime) determine one scene's cache contents:
    the tile's zarr store (scenes + DEM both live inside it) and the mask
    file itself. Zarr stores are directories, so their own mtime does not
    reliably reflect an inner-chunk write on every filesystem -- callers
    that need finer-grained invalidation should extend this."""
    return [zarr_path_for_tile(tile_dir, tile_id), Path(mask_path)]


def scene_cache_signature(
    scene_key: str,
    source_paths: Sequence[Path],
    feature_names: Sequence[str],
    schema_version: int = SCHEMA_VERSION,
) -> str:
    """Hash of everything that determines a cached scene tensor's contents:
    scene identity, the exact feature list in order, the schema version, and
    each source file/store's size + mtime. ``source_paths`` is always sorted
    before hashing, never trusted in the caller's original order."""
    payload = {
        "schema_version": int(schema_version),
        "scene_key": scene_key,
        "features": list(feature_names),
        "sources": [],
    }
    for path in sorted(source_paths, key=str):
        try:
            stat = Path(path).stat()
            payload["sources"].append([str(path), int(stat.st_size), int(stat.st_mtime)])
        except OSError:
            payload["sources"].append([str(path), -1, -1])
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def cache_is_current(cache_dir: Path, expected_signature: str) -> bool:
    """True only when both the ``.npz`` and its ``.json`` sidecar exist and
    the sidecar's recorded signature matches ``expected_signature``. Any
    mismatch (including a missing/corrupt sidecar) means rebuild -- there is
    no "close enough" branch."""
    npz_path = cache_dir / CACHE_FILENAME
    meta_path = cache_dir / META_FILENAME
    if not npz_path.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return meta.get("signature") == expected_signature


def write_scene_cache(
    cache_dir: Path,
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str],
    signature: str,
    sensor: str,
    tile_id: str,
    year: int,
    scene_id: str,
    transform: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    crs_wkt: str = "",
) -> Path:
    """Write one scene's ``.npz`` cache (``features`` float16, ``labels``
    uint8, plus identity metadata and georeferencing) and its ``.json``
    signature sidecar.

    ``transform``/``crs_wkt`` are stored so the cache stays self-contained:
    anyone holding just this ``.npz`` (e.g. a shared ``train_root`` from
    another contributor, without their raw zarr scene store) can still
    reconstruct a georeferenced raster from it (see
    ``scripts/07_export_georeferenced_cache.py``). They default to
    "unavailable" sentinels only so existing call sites that predate
    georeferencing keep working -- ``export_annotated_scene`` (the only
    production caller) always passes real values.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / CACHE_FILENAME
    meta_path = cache_dir / META_FILENAME

    np.savez_compressed(
        npz_path,
        features=features.astype(np.float16),
        labels=labels.astype(np.uint8),
        sensor=np.array(sensor),
        tile_id=np.array(tile_id),
        year=np.array(year),
        scene_id=np.array(scene_id),
        feature_names=np.array(list(feature_names)),
        transform=np.array(list(transform), dtype=np.float64),
        crs_wkt=np.array(crs_wkt),
    )
    meta_path.write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "signature": signature}, indent=2),
        encoding="utf-8",
    )
    return npz_path


def load_scene_cache(cache_dir: Path) -> dict:
    """Read one scene's ``.npz`` cache back, closing the file handle
    immediately after copying arrays out (so decompression buffers do not
    accumulate across many scenes, mirroring the reference repo's
    ``load_scene_arrays``).

    ``transform``/``crs_wkt`` are read with a fallback for caches written
    before georeferencing was added to the schema (``transform`` of all
    zeros / an empty ``crs_wkt`` signal "unavailable" rather than raising).
    """
    npz_path = cache_dir / CACHE_FILENAME
    with np.load(npz_path, allow_pickle=False) as data:
        return {
            "features": np.array(data["features"]),
            "labels": np.array(data["labels"]),
            "sensor": str(data["sensor"]),
            "tile_id": str(data["tile_id"]),
            "year": int(data["year"]),
            "scene_id": str(data["scene_id"]),
            "feature_names": [str(n) for n in data["feature_names"]],
            "transform": [float(v) for v in data["transform"]] if "transform" in data else [0.0] * 6,
            "crs_wkt": str(data["crs_wkt"]) if "crs_wkt" in data else "",
        }


def export_annotated_scene(
    tile_dir: str | Path,
    train_root: str | Path,
    scene: AnnotatedScene,
    class_config: ClassConfig,
    index_names: tuple[str, ...] = INDEX_NAMES,
    dem_layer_names: tuple[str, ...] = DEM_LAYER_NAMES,
) -> Optional[Path]:
    """Export one annotated scene into its training-cache ``.npz``, skipping
    the rebuild if an up-to-date cache already exists. Returns the ``.npz``
    path, or ``None`` if nothing needed writing."""
    cache_dir = Path(train_root) / scene.tile_id / scene.sensor / scene.scene_id
    feature_names = [*index_names, *dem_layer_names]
    source_paths = scene_source_paths(tile_dir, scene.tile_id, scene.mask_path)
    signature = scene_cache_signature(scene.scene_key, source_paths, feature_names)

    if cache_is_current(cache_dir, signature):
        return None

    features, feature_names, _provenance, transform, crs_wkt = build_feature_stack(
        tile_dir, scene.tile_id, scene.sensor, scene.scene_id, index_names, dem_layer_names
    )
    labels = read_mask_labels(scene.mask_path, class_config)
    if labels.shape != features.shape[1:]:
        raise ValueError(
            f"Mask shape {labels.shape} does not match feature grid {features.shape[1:]} "
            f"for scene '{scene.scene_key}' -- mask and scene are not on the same grid."
        )

    return write_scene_cache(
        cache_dir,
        features,
        labels,
        feature_names,
        signature,
        sensor=scene.sensor,
        tile_id=scene.tile_id,
        year=scene.year,
        scene_id=scene.scene_id,
        transform=transform,
        crs_wkt=crs_wkt,
    )


def export_all_annotated_scenes(
    tile_dir: str | Path,
    mask_root: str | Path,
    train_root: str | Path,
    class_config: ClassConfig,
    index_names: tuple[str, ...] = INDEX_NAMES,
    dem_layer_names: tuple[str, ...] = DEM_LAYER_NAMES,
) -> list[Path]:
    """Export every MaskForge-annotated scene under ``mask_root`` into
    ``train_root``, in sorted ``(tile_id, sensor, scene_id)`` order. Returns
    the ``.npz`` paths actually (re)written -- an up-to-date cache is
    skipped and not included."""
    written: list[Path] = []
    for scene in discover_annotated_scenes(mask_root):
        result = export_annotated_scene(tile_dir, train_root, scene, class_config, index_names, dem_layer_names)
        if result is not None:
            written.append(result)
    return written


def export_cache_scene_as_geotiff(
    cache_dir: str | Path,
    out_dir: str | Path,
    tile_id: str,
    sensor: str,
    scene_id: str,
) -> tuple[Path, Path]:
    """Write one cached scene's ``features.npz`` back out as two standalone,
    georeferenced GeoTIFFs -- ``<out_dir>/<tile_id>/<sensor>/<scene_id>/features.tif``
    (one band per feature, band descriptions set to ``feature_names``) and
    ``.../labels.tif`` (single-band class-id raster) -- using the
    ``transform``/``crs_wkt`` the cache stores alongside its arrays.

    This is the "give someone the training data as plain rasters" escape
    hatch: unlike ``features.npz`` (a training-only format with no
    georeferencing baked into most GIS tools' readers), the two GeoTIFFs
    here open directly in QGIS/ArcGIS/rasterio, can be reprojected,
    clipped, or otherwise reused with standard raster tools, independent of
    this pipeline. Raises :class:`ValueError` if the cache predates
    georeferencing (empty ``crs_wkt``) -- re-run
    ``scripts/04_export_training_cache.py`` to rebuild such a cache first.
    """
    import rasterio
    from rasterio.crs import CRS

    cache = load_scene_cache(Path(cache_dir))
    if not cache["crs_wkt"]:
        raise ValueError(
            f"Cache at '{cache_dir}' has no stored georeferencing (built before this feature "
            f"was added) -- re-run scripts/04_export_training_cache.py to rebuild it."
        )

    scene_out_dir = Path(out_dir) / tile_id / sensor / scene_id
    scene_out_dir.mkdir(parents=True, exist_ok=True)
    #: GDAL has no float16 dtype -- the cache stores features that way to
    #: keep .npz files small, but a GeoTIFF needs a GDAL-supported dtype.
    features = cache["features"].astype(np.float32)
    labels = cache["labels"]
    transform = list_to_transform(cache["transform"])
    crs = CRS.from_user_input(cache["crs_wkt"])

    features_path = scene_out_dir / "features.tif"
    with rasterio.open(
        features_path,
        "w",
        driver="GTiff",
        height=features.shape[1],
        width=features.shape[2],
        count=features.shape[0],
        dtype=features.dtype,
        crs=crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(features)
        for band_index, name in enumerate(cache["feature_names"], start=1):
            dst.set_band_description(band_index, name)

    labels_path = scene_out_dir / "labels.tif"
    with rasterio.open(
        labels_path,
        "w",
        driver="GTiff",
        height=labels.shape[0],
        width=labels.shape[1],
        count=1,
        dtype=labels.dtype,
        crs=crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(labels, 1)
        dst.set_band_description(1, "class_id")

    return features_path, labels_path


def export_train_root_as_geotiff(train_root: str | Path, out_dir: str | Path) -> list[tuple[Path, Path]]:
    """Export every scene cached under ``train_root`` to standalone
    georeferenced GeoTIFFs under ``out_dir`` (see
    :func:`export_cache_scene_as_geotiff`), in sorted ``(tile_id, sensor,
    scene_id)`` order. Scenes whose cache predates georeferencing are
    skipped with a printed warning rather than failing the whole export."""
    from landscape_change_detection_pipeline.training.dataset import discover_scene_records

    written: list[tuple[Path, Path]] = []
    for record in discover_scene_records(train_root):
        try:
            written.append(
                export_cache_scene_as_geotiff(
                    record.cache_dir, out_dir, record.tile_id, record.sensor, record.scene_id
                )
            )
        except ValueError as exc:
            print(f"[geotiff-export] skipping {record.scene_key}: {exc}")
    return written
