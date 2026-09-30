"""Tile-level driver for the per-pixel temporal segmentation.

Purpose
-------
Builds, for one tile, the full 1984-to-present time series of spectral
features for every pixel from the pipeline's own stored scenes, masks the
observations a land-surface model cannot use, runs
:func:`change.segmentation.segment_block` on every pixel, and writes one
``segments.npz`` per tile holding every pixel's segments (see
:mod:`change.segmentation`). Everything is local: scenes come from the
tile's zarr store (already on the shared 10 m grid), the per-observation
land-cover class from each scene's own classification. No external service
is involved.

Two passes, so every scene is decoded exactly once
----------------------------------------------------
A pixel's time series needs one value from every scene, but a scene is
stored as one full frame. Reading a block of rows per scene for every block
would decode each frame once per block. Instead:

1. **Cube pass** (scene-major): each scene is read once, its features and
   its class-based validity computed on the whole frame, and the result
   written into a temporary on-disk cube split into row blocks (features as
   float16 -- the features are bounded, reflectance-derived indices, far
   coarser than float16's precision -- and classes as uint8, 255 = unusable).
   Scenes are processed by a small thread pool.
2. **Block pass** (pixel-major): each row block is read back contiguously,
   arranged pixel-major, and segmented. Each finished block is written to its
   own small file, so an interrupted run resumes from the last finished
   block; the cube is kept until the tile is complete.

Finally the blocks are merged into ``segments.npz`` and the temporary cube
and block files are deleted.

What counts as usable
----------------------
An observation is usable at a pixel only if the scene has data there, all
features are finite, and the scene's classification is *not* one of
``masked_classes`` (by default cloud, shadow, snow, ice, open water --
states that are not the land surface a segment models). Snow and water are
therefore never mistaken for change; they simply are not used.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np

from landscape_change_detection_pipeline.change.segmentation import (
    MAX_CLASSES,
    SegmentationParams,
    aggregate_block,
    compact_block,
    harmonic_table,
    plan_aggregation,
    segment_block,
)
from landscape_change_detection_pipeline.change.spectral_composites import (
    ALL_INDEX_NAMES,
    discover_tile_scenes,
    indices_from_bands,
)
from landscape_change_detection_pipeline.classes.class_config import ClassConfig
from landscape_change_detection_pipeline.features.training_cache import scene_date
from landscape_change_detection_pipeline.inference.engine import read_class_map, scene_output_paths
from landscape_change_detection_pipeline.scenes.band_specs import get_band_spec
from landscape_change_detection_pipeline.scenes.gee_fetch import uint16_to_reflectance
from landscape_change_detection_pipeline.scenes.zarr_store import read_sensor_group_attrs, zarr_path_for_tile

#: Class value marking an unusable observation in the on-disk cube.
INVALID_CLASS = 255


def season_keep_mask(ordinals: np.ndarray, window) -> np.ndarray:
    """True for observations whose calendar day of year lies inside
    ``window`` = ``[first_doy, last_doy]`` (inclusive); everything when
    ``window`` is empty/None."""
    if not window:
        return np.ones(len(ordinals), dtype=bool)
    from datetime import date

    doy = np.array([date.fromordinal(int(o)).timetuple().tm_yday for o in ordinals])
    return (doy >= window[0]) & (doy <= window[1])


def segments_output_path(output_root: str | Path, tile_id: str) -> Path:
    """``<output_root>/<tile_id>/segments.npz``."""
    return Path(output_root) / tile_id / "segments.npz"


@dataclass(frozen=True)
class TileScenes:
    """The scenes of one tile that enter the time series, sorted by date."""

    ordinals: np.ndarray  # (n,) int64
    sensors: list[str]
    scene_ids: list[str]


def list_scenes(
    tile_dir: str | Path,
    inference_root: str | Path,
    tile_id: str,
    start_year: int,
    end_year: Optional[int],
) -> TileScenes:
    """Every stored scene of the tile inside ``[start_year, end_year]`` that
    already has a classification (a scene without one has no class to
    describe its observations and is skipped), sorted by date."""
    rows: list[tuple[int, str, str]] = []
    for sensor, scene_id in discover_tile_scenes(tile_dir, tile_id):
        d = scene_date(sensor, scene_id)
        if d.year < start_year or (end_year is not None and d.year > end_year):
            continue
        if not scene_output_paths(inference_root, tile_id, sensor, scene_id).is_file():
            continue
        rows.append((d.toordinal(), sensor, scene_id))
    rows.sort()
    return TileScenes(
        ordinals=np.array([r[0] for r in rows], dtype=np.int64),
        sensors=[r[1] for r in rows],
        scene_ids=[r[2] for r in rows],
    )


class _SceneReader:
    """Reads full-frame reflectance bands (``{label: (H, W) float32}``) for
    the tile's scenes from its zarr store, and checks that every sensor sits
    on one shared grid."""

    def __init__(self, tile_dir: str | Path, tile_id: str, sensors: set[str]):
        import zarr

        self._tile_dir = tile_dir
        self._tile_id = tile_id
        self._group = zarr.open_group(str(zarr_path_for_tile(tile_dir, tile_id)), mode="r")
        self._sensor: dict[str, dict] = {}
        for sensor in sorted(sensors):
            upper = sensor.upper()
            attrs = read_sensor_group_attrs(tile_dir, tile_id, upper)
            spec = get_band_spec(upper)
            band_names = list(attrs["band_names"])
            role_to_prov = {"native": "native", "resample": "resampled_for_alignment"}
            stored = dict(attrs.get("band_provenance", {}))
            label_to_row: dict[str, int] = {}
            provenance: dict[str, str] = {}
            for row, name in enumerate(band_names):
                band = spec.band(name)
                label_to_row[band.label] = row
                provenance[band.label] = stored.get(name) or role_to_prov[band.role.value]
            self._sensor[sensor] = {
                "array": self._group[upper]["toa"],
                "index": {sid: i for i, sid in enumerate(attrs["scene_ids"])},
                "label_to_row": label_to_row,
                "provenance": provenance,
                "transform": tuple(float(v) for v in attrs["transform"]),
                "crs_wkt": str(attrs["crs_wkt"]),
                "shape": tuple(self._group[upper]["toa"].shape[2:]),
            }
        grids = {(info["shape"], info["transform"]) for info in self._sensor.values()}
        if len(grids) != 1:
            raise ValueError(
                f"Tile '{tile_id}' has scenes on more than one pixel grid across sensors "
                f"({sorted(str(g) for g in grids)}); the time series needs every scene on the tile's shared "
                "10 m grid. Re-download the scenes that are off-grid."
            )
        info = next(iter(self._sensor.values()))
        self.shape: tuple[int, int] = info["shape"]
        self.transform: tuple[float, ...] = info["transform"]
        self.crs_wkt: str = info["crs_wkt"]

    def bands(self, sensor: str, scene_id: str) -> tuple[dict[str, np.ndarray], dict[str, str]]:
        info = self._sensor[sensor]
        raw = np.asarray(info["array"][info["index"][scene_id]])
        reflectance = uint16_to_reflectance(raw)
        bands = {label: reflectance[row] for label, row in info["label_to_row"].items()}
        return bands, info["provenance"]


def _row_blocks(height: int, block_rows: int) -> list[tuple[int, int]]:
    return [(r0, min(r0 + block_rows, height)) for r0 in range(0, height, block_rows)]


def _signature(scenes: TileScenes, cfg, class_ids: list[int], grid: tuple) -> str:
    payload = {
        "scenes": list(zip(scenes.ordinals.tolist(), scenes.sensors, scenes.scene_ids)),
        "features": list(cfg.features),
        "masked": class_ids,
        "block_rows": cfg.block_rows,
        "grid": [list(grid[0]), list(grid[1])],
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _params_signature(cfg) -> str:
    keys = ("p_change", "conse", "init_obs", "init_min_span_days", "stability_threshold", "min_rmse",
            "max_harmonics", "outlier_p", "max_segments", "use_slope_interaction", "min_magnitude",
            "magnitude_features", "min_break_span_days", "refit_every", "aggregate_days", "season_window",
            "min_magnitude_same_class", "persist_days", "min_obs_break_year")
    return json.dumps({k: getattr(cfg, k, None) for k in keys}, sort_keys=True)


def _build_cube(
    reader: _SceneReader,
    scenes: TileScenes,
    inference_root: str | Path,
    tile_id: str,
    cfg,
    masked_ids: list[int],
    cube_dir: Path,
    blocks: list[tuple[int, int]],
) -> None:
    """Pass 1: one read per scene, features + validity written into the
    per-block on-disk cube."""
    n_obs = len(scenes.ordinals)
    n_feat = len(cfg.features)
    height, width = reader.shape
    feat_mm = []
    cls_mm = []
    for b, (r0, r1) in enumerate(blocks):
        feat_mm.append(np.lib.format.open_memmap(
            cube_dir / f"feat_{b:04d}.npy", mode="w+", dtype=np.float16, shape=(n_obs, n_feat, r1 - r0, width)))
        cls_mm.append(np.lib.format.open_memmap(
            cube_dir / f"cls_{b:04d}.npy", mode="w+", dtype=np.uint8, shape=(n_obs, r1 - r0, width)))

    features = tuple(cfg.features)
    masked = np.array(masked_ids, dtype=np.int64)

    def one_scene(i: int) -> None:
        sensor, scene_id = scenes.sensors[i], scenes.scene_ids[i]
        bands, provenance = reader.bands(sensor, scene_id)
        indices = indices_from_bands(bands, provenance, sensor, features)
        stack = np.stack([indices[name] for name in features], axis=0).astype(np.float32)

        cm_data = read_class_map(scene_output_paths(inference_root, tile_id, sensor, scene_id))
        class_map = cm_data["class_map"]
        if class_map.shape != (height, width):
            raise ValueError(
                f"Classification of {tile_id}/{sensor}/{scene_id} has shape {class_map.shape}, "
                f"expected the tile grid {(height, width)}."
            )
        unusable = np.isin(class_map, masked) | (class_map == cm_data["nodata"]) | ~np.isfinite(stack).all(axis=0)
        cls = np.where(unusable, INVALID_CLASS, class_map).astype(np.uint8)
        for b, (r0, r1) in enumerate(blocks):
            feat_mm[b][i] = stack[:, r0:r1, :].astype(np.float16)
            cls_mm[b][i] = cls[r0:r1]

    with ThreadPoolExecutor(max_workers=max(1, cfg.io_threads)) as pool:
        for n_done, _ in enumerate(pool.map(one_scene, range(n_obs)), start=1):
            if n_done % 100 == 0 or n_done == n_obs:
                print(f"[change_detection] {tile_id}: read {n_done}/{n_obs} scenes")

    for mm in feat_mm + cls_mm:
        mm.flush()
    del feat_mm, cls_mm


def run_tile(
    tile_dir: str | Path,
    inference_root: str | Path,
    output_root: str | Path,
    tile_id: str,
    class_config: ClassConfig,
    cfg,
    overwrite: bool = False,
) -> Optional[Path]:
    """Segment one tile. Returns the written ``segments.npz`` path, or
    ``None`` if it already exists and ``overwrite`` is false."""
    from scipy.stats import chi2

    out_path = segments_output_path(output_root, tile_id)
    if out_path.is_file() and not overwrite:
        return None

    unknown = set(cfg.features) - set(ALL_INDEX_NAMES)
    if unknown:
        raise ValueError(f"change_detection.features has unknown entries {sorted(unknown)} (allowed: {ALL_INDEX_NAMES}).")
    masked_ids = [class_config.by_name(name).id for name in cfg.masked_classes]
    if max(c.id for c in class_config.classes) >= MAX_CLASSES:
        raise ValueError(f"Class ids must be < {MAX_CLASSES} for per-segment class statistics.")

    scenes = list_scenes(tile_dir, inference_root, tile_id, cfg.start_year, cfg.end_year)
    if len(scenes.ordinals) == 0:
        raise ValueError(f"No classified scenes found for tile '{tile_id}' between {cfg.start_year} and "
                         f"{cfg.end_year or 'now'} -- run the classification stage first.")
    reader = _SceneReader(tile_dir, tile_id, set(scenes.sensors))
    height, width = reader.shape
    n_obs = len(scenes.ordinals)
    n_feat = len(cfg.features)
    print(f"[change_detection] {tile_id}: {n_obs} scenes ({scenes.sensors.count('s2')} S2), "
          f"grid {height}x{width}, features {list(cfg.features)}")

    work = Path(output_root) / tile_id / "_work"
    cube_dir = work / "cube"
    block_dir = work / "blocks"
    blocks = _row_blocks(height, cfg.block_rows)
    cube_sig = _signature(scenes, cfg, masked_ids, (reader.shape, reader.transform))
    params_sig = _params_signature(cfg)
    sig_file = work / "signature.json"

    fresh = True
    if sig_file.is_file():
        saved = json.loads(sig_file.read_text())
        if saved.get("cube") == cube_sig and (cube_dir / ".done").is_file():
            fresh = False
    if fresh:
        shutil.rmtree(work, ignore_errors=True)
        cube_dir.mkdir(parents=True)
        block_dir.mkdir(parents=True)
        _build_cube(reader, scenes, inference_root, tile_id, cfg, masked_ids, cube_dir, blocks)
        (cube_dir / ".done").write_text("ok")
        sig_file.write_text(json.dumps({"cube": cube_sig, "params": params_sig}))
    else:
        saved = json.loads(sig_file.read_text())
        if saved.get("params") != params_sig:
            shutil.rmtree(block_dir, ignore_errors=True)
            block_dir.mkdir(parents=True)
            sig_file.write_text(json.dumps({"cube": cube_sig, "params": params_sig}))
        print(f"[change_detection] {tile_id}: reusing the scene cube from an earlier run")

    if cfg.numba_threads > 0:
        import numba

        numba.set_num_threads(cfg.numba_threads)

    params = SegmentationParams(
        chi_threshold=float(chi2.ppf(cfg.p_change, n_feat)),
        outlier_threshold=float(chi2.ppf(cfg.outlier_p, n_feat)),
        min_rmse=cfg.min_rmse,
        stability_threshold=cfg.stability_threshold,
        init_obs=cfg.init_obs,
        init_min_span_days=float(cfg.init_min_span_days),
        conse=cfg.conse,
        max_harmonics=cfg.max_harmonics,
        max_segments=cfg.max_segments,
        use_slope_interaction=getattr(cfg, "use_slope_interaction", False),
        min_magnitude=float(cfg.min_magnitude),
        magnitude_features=tuple(list(cfg.features).index(n) for n in cfg.magnitude_features),
        min_break_span_days=float(cfg.min_break_span_days),
        refit_every=int(cfg.refit_every),
        min_magnitude_same_class=float(cfg.min_magnitude_same_class),
        persist_days=float(cfg.persist_days),
        min_obs_break_year=int(cfg.min_obs_break_year),
    )
    from landscape_change_detection_pipeline.scenes.sensors import SENSOR_ORDER
    sensor_index = {s: i for i, s in enumerate(SENSOR_ORDER)}
    sensor_ids = np.array([sensor_index[s.upper()] for s in scenes.sensors], dtype=np.int64)

    # growing-season window: observations outside it (snowmelt, leaf fall,
    # low sun) are dropped before anything else, so the seasonal transitions
    # that the annual harmonics model badly never reach the change test
    keep = season_keep_mask(scenes.ordinals, cfg.season_window)
    keep_idx = np.flatnonzero(keep)
    if len(keep_idx) < len(keep):
        print(f"[change_detection] {tile_id}: season window {list(cfg.season_window)} keeps {len(keep_idx)}/{len(keep)} scenes")
        params = replace(params, max_harmonics=1)  # a ~3-month window has no room for more than one harmonic
    seg_dates, seg_sensor_ids = scenes.ordinals[keep_idx], sensor_ids[keep_idx]
    member_ptr = members = None
    if cfg.aggregate_days > 0:
        seg_dates, seg_sensor_ids, member_ptr, members = plan_aggregation(seg_dates, seg_sensor_ids, cfg.aggregate_days)
        print(f"[change_detection] {tile_id}: {len(keep_idx)} scenes aggregated into {len(seg_dates)} "
              f"{cfg.aggregate_days}-day per-sensor observations")
    harmonics = harmonic_table(seg_dates)

    slope_full = None
    if getattr(cfg, "use_slope_interaction", False):
        import zarr
        zgroup = zarr.open_group(str(zarr_path_for_tile(tile_dir, tile_id)), mode="r")
        slope_full = np.asarray(zgroup["dem"]["slope"], dtype=np.float64)

    for b, (r0, r1) in enumerate(blocks):
        block_file = block_dir / f"block_{b:04d}.npz"
        if block_file.is_file():
            continue
        feat = np.load(cube_dir / f"feat_{b:04d}.npy", mmap_mode="r")
        cls = np.load(cube_dir / f"cls_{b:04d}.npy", mmap_mode="r")
        if len(keep_idx) < n_obs:
            feat, cls = feat[keep_idx], cls[keep_idx]
        rows = r1 - r0
        values = np.ascontiguousarray(np.transpose(feat, (2, 3, 0, 1)), dtype=np.float32).reshape(rows * width, len(keep_idx), n_feat)
        classes = np.ascontiguousarray(np.transpose(cls, (1, 2, 0))).reshape(rows * width, len(keep_idx))
        valid = classes != INVALID_CLASS
        if members is not None:
            values, valid, classes = aggregate_block(values, valid, classes, member_ptr, members)
        block_slope = None if slope_full is None else np.ascontiguousarray(slope_full[r0:r1, :]).reshape(rows * width)
        result = segment_block(seg_dates, values, valid, classes, params, harmonics, seg_sensor_ids, block_slope)
        compact = compact_block(result, width, r0, n_feat)
        np.savez(
            block_file,
            n_valid=result["n_valid"].reshape(rows, width),
            n_segments=result["n_segments"].reshape(rows, width),
            truncated=result["truncated"].reshape(rows, width),
            **compact,
        )
        del values, classes, valid, result, compact, feat, cls
        print(f"[change_detection] {tile_id}: block {b + 1}/{len(blocks)} done")

    _merge_blocks(block_dir, len(blocks), out_path, reader, seg_dates, seg_sensor_ids, cfg, masked_ids)
    shutil.rmtree(work, ignore_errors=True)
    return out_path


def _merge_blocks(
    block_dir: Path, n_blocks: int, out_path: Path, reader: _SceneReader, obs_dates: np.ndarray,
    obs_sensor_ids: np.ndarray, cfg, masked_ids: list[int],
) -> None:
    from landscape_change_detection_pipeline.scenes.sensors import SENSOR_ORDER

    height, width = reader.shape
    parts = [np.load(block_dir / f"block_{b:04d}.npz") for b in range(n_blocks)]
    record_keys = [k for k in parts[0].files if k not in ("n_valid", "n_segments", "truncated")]
    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in record_keys}
    maps = {k: np.concatenate([p[k] for p in parts], axis=0) for k in ("n_valid", "n_segments", "truncated")}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        **merged,
        **maps,
        transform=np.array(reader.transform[:6], dtype=np.float64),
        crs_wkt=np.array(reader.crs_wkt),
        resolution_m=np.array(abs(reader.transform[0]), dtype=np.float64),
        feature_names=np.array(list(cfg.features)),
        masked_class_ids=np.array(masked_ids, dtype=np.int16),
        obs_dates=obs_dates.astype(np.int32),
        obs_sensors=np.array([SENSOR_ORDER[i].lower() for i in obs_sensor_ids]),
    )


def read_segments(path: str | Path) -> dict:
    """Read a tile's ``segments.npz`` back into a dict of arrays plus
    georeferencing (``transform``, ``crs_wkt``, ``resolution_m``, ``shape``)
    and the ``feature_names`` the per-feature columns refer to."""
    with np.load(Path(path), allow_pickle=False) as data:
        result = {k: np.array(data[k]) for k in data.files}
    result["transform"] = tuple(float(v) for v in result["transform"])
    result["crs_wkt"] = str(result["crs_wkt"])
    result["resolution_m"] = float(result["resolution_m"])
    result["feature_names"] = [str(n) for n in result["feature_names"]]
    result["shape"] = tuple(result["n_valid"].shape)
    return result
