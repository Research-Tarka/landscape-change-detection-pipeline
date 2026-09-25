"""Local CCDC/COLD change detection (per-tile, per-pixel break dates).

Purpose
-------
Detects per-pixel breakpoints (disturbance date + spectral magnitude) across
each tile's full multi-decade time series, independent of the land-cover
classification -- a direct, statistically-grounded disturbance-timing signal,
in place of the static inventory-age fields this kind of analysis usually
depends on. Runs entirely on
this pipeline's own already-downloaded, already-resampled-onto-the-DEM's-grid
reflectance (``data/tiles/<tile_id>.zarr``) via
`pyxccd <https://pypi.org/project/pyxccd/>`_
(the actively-maintained local Python COLD/CCDC implementation), never
against Earth Engine's server-side collections: GEE's raw
collections are not resampled onto this pipeline's unified 10 m/DEM grid the
way this pipeline's own stored bands are, so running CCDC against them would
detect breaks on pixels that do not correspond 1:1 to the pixels this
project's own classification and mosaics operate on.

One shared grid per tile, at the finest resolution ever available
---------------------------------------------------------------------
Different sensors within one tile sit on different native grids (L5 30 m,
L7/L8/L9 15 m, S2 10 m -- confirmed directly on real pilot-tile zarr attrs).
A per-pixel time series requires every scene's value at one real-world
location to land in the same array cell, so every scene is reprojected
(bilinear -- continuous reflectance, never nearest-neighbour) onto **one**
grid per tile: the finest resolution any sensor ever achieved for that tile,
across its whole history -- e.g. a tile that only gets Sentinel-2 coverage
starting in 2023 still has its 1999 Landsat-5 scenes upscaled to that same
10 m grid, so the full decades-long series lines up pixel-for-pixel. This
mirrors the mosaic/composite convention elsewhere in this pipeline
(:func:`landscape_change_detection_pipeline.inference.composites.pick_reference_grid`,
:func:`landscape_change_detection_pipeline.mosaic.mosaic.pick_mosaic_resolution`): always
resample the coarser data up to whatever finer grid exists, never the
reverse.

QA masking: this project's own trained classification, not a scene-level
cloud percentage
--------------------------------------------------------------------------
Cloud/shadow handling prefers this
project's own trained classification's cloud/shadow flags over GEE's
scene-level CLOUD_COVER/CLOUDY_PIXEL_PERCENTAGE property alone once a real
model exists, since a scene-level cloud percentage misses per-pixel cases a
full scene otherwise passes. Each scene's own class map
(``outputs/inference/<tile_id>/<sensor>/<scene_id>/class_map.npz``, Stage 6)
is reprojected onto the same shared grid and translated directly into
pyxccd's QA scheme (0 clear / 1 water / 2 shadow / 3 snow / 4 cloud) using
this project's own class names -- ``open_water``/``ice_cover`` -> water,
``shadow`` -> shadow, ``snow_cover`` -> snow, ``cloud`` -> cloud, everything
else -> clear. A scene with no classification yet is skipped entirely
(mirrors ``change.spectral_composites``'s same requirement and reasoning).

Bands fed to COLD
--------------------
``cold_detect_flex`` (not the fixed 7-band ``cold_detect``, since this
pipeline's own bands are already normalized to six labels
-- blue/green/red/nir/swir1/swir2, no thermal band at all -- rather than
pyxccd's default Landsat-7-band assumption) with the Tmask bands set to
green (index 2 in the 1-indexed stack pyxccd expects) and swir1 (index 5),
matching CCDC's own documented Tmask default.

Output
------
One ``.npz`` per tile (``<output_root>/<tile_id>/ccdc.npz``): every pixel's
detected segments flattened into parallel arrays (row, col, t_start, t_end,
t_break, num_obs, change_prob, magnitude per band), not a fixed-shape image
-- pixel segment counts vary, so a ragged/flattened table is the natural
shape here, the same reasoning ``pyxccd`` itself returns a structured array
per pixel rather than a fixed-width image.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from landscape_change_detection_pipeline.classes.class_config import ClassConfig
from landscape_change_detection_pipeline.features.training_cache import bands_for_scene, scene_date
from landscape_change_detection_pipeline.inference.engine import read_class_map, scene_output_paths
from landscape_change_detection_pipeline.scenes.zarr_store import read_sensor_group_attrs, zarr_path_for_tile

#: The six reflective bands every sensor in this pipeline is normalized to
#: (see ``scenes/band_specs.py``), in the fixed order fed to pyxccd.
CCDC_BAND_ORDER: tuple[str, ...] = ("blue", "green", "red", "nir", "swir1", "swir2")
#: 1-indexed positions of green/swir1 within CCDC_BAND_ORDER, for pyxccd's
#: Tmask parameters (CCDC's own documented default: green + swir1).
_TMASK_B1_INDEX = CCDC_BAND_ORDER.index("green") + 1
_TMASK_B2_INDEX = CCDC_BAND_ORDER.index("swir1") + 1

SENSOR_RESOLUTION_M: dict[str, float] = {"l5": 30.0, "l7": 15.0, "l8": 15.0, "l9": 15.0, "s2": 10.0}
SENSORS: tuple[str, ...] = ("l5", "l7", "l8", "l9", "s2")

#: pyxccd's QA scheme (0 clear, 1 water, 2 shadow, 3 snow, 4 cloud), mapped
#: from this project's own class names (configs/classes.yaml) rather than a
#: scene-level cloud percentage -- see module docstring.
_QA_CLEAR, _QA_WATER, _QA_SHADOW, _QA_SNOW, _QA_CLOUD = 0, 1, 2, 3, 4
_CLASS_NAME_TO_QA: dict[str, int] = {
    "open_water": _QA_WATER,
    "ice_cover": _QA_WATER,
    "shadow": _QA_SHADOW,
    "snow_cover": _QA_SNOW,
    "cloud": _QA_CLOUD,
}


@dataclass(frozen=True)
class TileTimeSeries:
    """One tile's per-pixel time series, aligned to one shared grid."""

    dates_ordinal: np.ndarray  # (n_obs,) int64
    bands: np.ndarray  # (n_obs, n_bands, H, W) float64
    qas: np.ndarray  # (n_obs, H, W) int64
    transform: tuple[float, ...]
    crs_wkt: str
    resolution_m: float


def class_map_to_qa(class_map: np.ndarray, nodata: int, class_config: ClassConfig) -> np.ndarray:
    """Translate one scene's classification into pyxccd's QA scheme, using
    this project's own class names rather than a scene-level cloud percent
    (see module docstring). Nodata and any class not in the explicit mapping
    (i.e. every real land-cover class) become ``clear`` (0)."""
    qa = np.full(class_map.shape, _QA_CLEAR, dtype=np.int64)
    for class_name, qa_code in _CLASS_NAME_TO_QA.items():
        try:
            class_id = class_config.by_name(class_name).id
        except KeyError:
            continue
        qa[class_map == class_id] = qa_code
    qa[class_map == nodata] = _QA_CLOUD  # nodata treated as unusable, not clear
    return qa


def tile_finest_resolution(tile_dir: str | Path, tile_id: str) -> float:
    """The finest (smallest) resolution any sensor ever achieved for this
    tile, across its whole stored history -- the shared grid every scene is
    reprojected onto (see module docstring)."""
    import zarr

    zarr_path = zarr_path_for_tile(tile_dir, tile_id)
    store = zarr.open_group(str(zarr_path), mode="r")
    present = [SENSOR_RESOLUTION_M[s] for s in SENSORS if s.upper() in store]
    if not present:
        raise ValueError(f"No sensor data stored for tile '{tile_id}' at '{zarr_path}'.")
    return min(present)


def _reproject_bilinear(array, src_transform, src_crs_wkt, dst_transform, dst_crs_wkt, dst_shape):
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    destination = np.full(dst_shape, np.nan, dtype=np.float64)
    reproject(
        source=array.astype(np.float64),
        destination=destination,
        src_transform=Affine(*src_transform[:6]),
        src_crs=CRS.from_user_input(src_crs_wkt),
        dst_transform=Affine(*dst_transform[:6]),
        dst_crs=CRS.from_user_input(dst_crs_wkt),
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return destination


def _reproject_nearest_int(array, src_transform, src_crs_wkt, dst_transform, dst_crs_wkt, dst_shape, fill_value):
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    destination = np.full(dst_shape, fill_value, dtype=np.int64)
    reproject(
        source=array.astype(np.int64),
        destination=destination,
        src_transform=Affine(*src_transform[:6]),
        src_crs=CRS.from_user_input(src_crs_wkt),
        dst_transform=Affine(*dst_transform[:6]),
        dst_crs=CRS.from_user_input(dst_crs_wkt),
        resampling=Resampling.nearest,
    )
    return destination


def build_tile_time_series(
    tile_dir: str | Path,
    inference_root: str | Path,
    tile_id: str,
    class_config: ClassConfig,
) -> TileTimeSeries:
    """Every stored, already-classified scene for this tile, reprojected
    onto one shared grid (the tile's finest-ever resolution), sorted by
    acquisition date. Scenes with no Stage 6 classification are skipped
    (same requirement as :mod:`change.spectral_composites`)."""
    import zarr
    from affine import Affine

    resolution_m = tile_finest_resolution(tile_dir, tile_id)

    # The reference grid: whichever sensor achieves resolution_m natively,
    # its own transform/shape/CRS -- if more than one does, the first found.
    zarr_path = zarr_path_for_tile(tile_dir, tile_id)
    store = zarr.open_group(str(zarr_path), mode="r")
    reference_sensor = next(s for s in SENSORS if s.upper() in store and SENSOR_RESOLUTION_M[s] == resolution_m)
    ref_attrs = read_sensor_group_attrs(tile_dir, tile_id, reference_sensor.upper())
    dst_transform = Affine(*[float(v) for v in ref_attrs["transform"]][:6])
    dst_crs_wkt = str(ref_attrs["crs_wkt"])
    sample_toa = store[reference_sensor.upper()]["toa"]
    dst_shape = (sample_toa.shape[2], sample_toa.shape[3])

    records: list[tuple] = []  # (date_ordinal, sensor, scene_id)
    for sensor in SENSORS:
        if sensor.upper() not in store:
            continue
        for scene_id in sorted(store[sensor.upper()].attrs.get("scene_ids", [])):
            if not scene_output_paths(inference_root, tile_id, sensor, scene_id).is_file():
                continue
            records.append((scene_date(sensor, scene_id).toordinal(), sensor, scene_id))
    records.sort()

    if not records:
        raise ValueError(f"No classified scenes found for tile '{tile_id}' -- run Stage 6 (inference) first.")

    n_obs = len(records)
    n_bands = len(CCDC_BAND_ORDER)
    bands = np.empty((n_obs, n_bands, *dst_shape), dtype=np.float64)
    qas = np.empty((n_obs, *dst_shape), dtype=np.int64)
    dates_ordinal = np.empty(n_obs, dtype=np.int64)

    for i, (ordinal, sensor, scene_id) in enumerate(records):
        dates_ordinal[i] = ordinal
        band_dict, _provenance = bands_for_scene(tile_dir, tile_id, sensor.upper(), scene_id)
        attrs = read_sensor_group_attrs(tile_dir, tile_id, sensor.upper())
        src_transform = tuple(float(v) for v in attrs["transform"])
        src_crs_wkt = str(attrs["crs_wkt"])

        class_map_data = read_class_map(scene_output_paths(inference_root, tile_id, sensor, scene_id))
        qa_native = class_map_to_qa(class_map_data["class_map"], class_map_data["nodata"], class_config)

        same_grid = SENSOR_RESOLUTION_M[sensor] == resolution_m and sensor == reference_sensor
        for b, label in enumerate(CCDC_BAND_ORDER):
            if same_grid:
                bands[i, b] = band_dict[label]
            else:
                bands[i, b] = _reproject_bilinear(
                    band_dict[label], src_transform, src_crs_wkt, dst_transform, dst_crs_wkt, dst_shape
                )
        if same_grid:
            qas[i] = qa_native
        else:
            qas[i] = _reproject_nearest_int(
                qa_native, src_transform, src_crs_wkt, dst_transform, dst_crs_wkt, dst_shape, fill_value=_QA_CLOUD
            )

    return TileTimeSeries(
        dates_ordinal=dates_ordinal,
        bands=bands,
        qas=qas,
        transform=tuple(dst_transform)[:6],
        crs_wkt=dst_crs_wkt,
        resolution_m=resolution_m,
    )


@dataclass(frozen=True)
class PixelSegments:
    row: int
    col: int
    segments: np.ndarray  # structured array, pyxccd's cold_rec_cg dtype


def run_ccdc_pixel(
    dates_ordinal: np.ndarray,
    band_values: np.ndarray,  # (n_obs, n_bands)
    qa_values: np.ndarray,  # (n_obs,)
    lam: float = 20.0,
    p_cg: float = 0.99,
    conse: int = 6,
    min_clear_obs: int = 12,
) -> Optional[np.ndarray]:
    """Run pyxccd's COLD detector on one pixel's time series. Returns the
    structured segment array, or ``None`` if this pixel cannot be fit --
    either too few clear observations up front (below ``min_clear_obs``,
    cheap to check before calling into the C extension at all), or pyxccd's
    own C core raising ``Exception("... no change records outputted ...")``
    for a pixel that passed the clear-count check but still could not
    initialize a model (confirmed on real pilot data: a pixel with 38/38
    clear observations still raised this for reasons internal to COLD's own
    fitting logic, so the upfront count alone is not a sufficient guard).
    Both cases mean "this pixel has no usable COLD result," never a genuine
    pipeline error, so both are treated as ``None`` rather than propagated.
    """
    from pyxccd import cold_detect_flex

    n_clear = int((qa_values == _QA_CLEAR).sum())
    if n_clear < min_clear_obs:
        return None

    try:
        return cold_detect_flex(
            dates=dates_ordinal,
            ts_stack=band_values,
            qas=qa_values,
            lam=lam,
            p_cg=p_cg,
            conse=conse,
            tmask_b1_index=_TMASK_B1_INDEX,
            tmask_b2_index=_TMASK_B2_INDEX,
        )
    except Exception as exc:  # noqa: BLE001 -- pyxccd's C core raises a bare Exception, not a typed one
        if "no change records outputted" in str(exc):
            return None
        raise


def _run_pixel_worker(args: tuple) -> Optional[tuple]:
    """Module-level (picklable) worker for :func:`run_ccdc_tile` 's process
    pool -- a bound method or closure cannot be pickled for
    ``multiprocessing`` on Windows (no ``fork``), so this takes a plain
    tuple and returns a plain tuple."""
    row, col, dates_ordinal, band_values, qa_values, lam, p_cg, conse, min_clear_obs = args
    result = run_ccdc_pixel(dates_ordinal, band_values, qa_values, lam, p_cg, conse, min_clear_obs)
    if result is None or len(result) == 0:
        return None
    return row, col, result


def run_ccdc_tile(
    ts: TileTimeSeries,
    lam: float = 20.0,
    p_cg: float = 0.99,
    conse: int = 6,
    min_clear_obs: int = 12,
    n_workers: int = 1,
) -> list[PixelSegments]:
    """Run COLD independently on every pixel of one tile's aligned time
    series -- embarrassingly parallel across pixels, so ``n_workers`` spreads
    it over local CPU cores via ``multiprocessing`` (never GPU: no CUDA path
    exists for this algorithm)."""
    _n_obs, _n_bands, height, width = ts.bands.shape

    tasks = []
    for row in range(height):
        for col in range(width):
            band_values = np.ascontiguousarray(ts.bands[:, :, row, col])
            qa_values = np.ascontiguousarray(ts.qas[:, row, col])
            if np.isnan(band_values).all():
                continue  # pixel outside every scene's actual footprint
            tasks.append((row, col, ts.dates_ordinal, band_values, qa_values, lam, p_cg, conse, min_clear_obs))

    if n_workers <= 1:
        raw_results = [_run_pixel_worker(t) for t in tasks]
    else:
        from multiprocessing import Pool

        with Pool(processes=n_workers) as pool:
            raw_results = pool.map(_run_pixel_worker, tasks)

    return [PixelSegments(row=item[0], col=item[1], segments=item[2]) for item in raw_results if item is not None]


def ccdc_output_path(output_root: str | Path, tile_id: str) -> Path:
    """``<output_root>/<tile_id>/ccdc.npz``."""
    return Path(output_root) / tile_id / "ccdc.npz"


def write_ccdc_result(path: str | Path, pixel_segments: list[PixelSegments], ts: TileTimeSeries) -> Path:
    """Flatten every pixel's variable-length segment list into parallel
    arrays (one row per segment, across all pixels) -- a ragged table, not a
    fixed-shape image, since segment count varies per pixel."""
    rows: list[int] = []
    cols: list[int] = []
    t_start: list[int] = []
    t_end: list[int] = []
    t_break: list[int] = []
    num_obs: list[int] = []
    change_prob: list[int] = []
    magnitude: list[np.ndarray] = []

    for ps in pixel_segments:
        for seg in ps.segments:
            rows.append(ps.row)
            cols.append(ps.col)
            t_start.append(int(seg["t_start"]))
            t_end.append(int(seg["t_end"]))
            t_break.append(int(seg["t_break"]))
            num_obs.append(int(seg["num_obs"]))
            change_prob.append(int(seg["change_prob"]))
            magnitude.append(np.array(seg["magnitude"], dtype=np.float32))

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        row=np.array(rows, dtype=np.int32),
        col=np.array(cols, dtype=np.int32),
        t_start=np.array(t_start, dtype=np.int32),
        t_end=np.array(t_end, dtype=np.int32),
        t_break=np.array(t_break, dtype=np.int32),
        num_obs=np.array(num_obs, dtype=np.int32),
        change_prob=np.array(change_prob, dtype=np.int16),
        magnitude=np.array(magnitude, dtype=np.float32) if magnitude else np.zeros((0, len(CCDC_BAND_ORDER)), dtype=np.float32),
        band_order=np.array(CCDC_BAND_ORDER),
        transform=np.array(list(ts.transform)[:6], dtype=np.float64),
        crs_wkt=np.array(ts.crs_wkt),
        resolution_m=np.array(ts.resolution_m, dtype=np.float64),
        shape=np.array(ts.bands.shape[2:], dtype=np.int32),  # (H, W) -- the tile's break-detection grid shape
    )
    return out


def read_ccdc_result(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as data:
        return {
            "row": np.array(data["row"]),
            "col": np.array(data["col"]),
            "t_start": np.array(data["t_start"]),
            "t_end": np.array(data["t_end"]),
            "t_break": np.array(data["t_break"]),
            "num_obs": np.array(data["num_obs"]),
            "change_prob": np.array(data["change_prob"]),
            "magnitude": np.array(data["magnitude"]),
            "band_order": [str(b) for b in data["band_order"]],
            "transform": tuple(float(v) for v in data["transform"]),
            "crs_wkt": str(data["crs_wkt"]),
            "resolution_m": float(data["resolution_m"]),
            "shape": tuple(int(v) for v in data["shape"]),
        }


def build_tile_ccdc(
    tile_dir: str | Path,
    inference_root: str | Path,
    output_root: str | Path,
    tile_id: str,
    class_config: ClassConfig,
    lam: float = 20.0,
    p_cg: float = 0.99,
    conse: int = 6,
    min_clear_obs: int = 12,
    n_workers: int = 1,
    overwrite: bool = False,
) -> Optional[Path]:
    """Build one tile's full CCDC result: aligned time series -> per-pixel
    COLD fit -> flattened segment table. Returns the written path, or
    ``None`` if already built and ``overwrite`` is false."""
    out_path = ccdc_output_path(output_root, tile_id)
    if out_path.is_file() and not overwrite:
        return None

    ts = build_tile_time_series(tile_dir, inference_root, tile_id, class_config)
    pixel_segments = run_ccdc_tile(ts, lam=lam, p_cg=p_cg, conse=conse, min_clear_obs=min_clear_obs, n_workers=n_workers)
    write_ccdc_result(out_path, pixel_segments, ts)
    return out_path
