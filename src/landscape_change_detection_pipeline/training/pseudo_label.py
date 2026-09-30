"""Streaming semi-supervised pseudo-labeling over the unlabeled scene pool.

Purpose
-------
The raw scene store under ``config.dem.tile_dir`` (``data/tiles``) holds
every downloaded scene for every tile -- far more than the small subset
MaskForge has annotated (``config.features.mask_root`` ->
``config.features.train_root``). This module scores every *unlabeled* scene
(one with no corresponding annotated mask) with an already-trained
checkpoint, keeps only the pixels the model is confident about, and writes
the result to disk in exactly :mod:`landscape_change_detection_pipeline.features.training_cache`'s
own per-scene ``.npz`` cache schema -- so the resulting tree is a drop-in
:func:`~landscape_change_detection_pipeline.training.dataset.discover_scene_records`
target, indistinguishable in format from a real annotated-scene cache.

Memory discipline
------------------
``data/tiles`` is on the order of hundreds of GB (every raw scene for every
tile, labeled or not) -- an order of magnitude past what fits in RAM or
VRAM alongside the training run itself. This module never holds more than
one scene's feature stack in memory at a time: :func:`generate_pseudo_labels`
loops scene by scene, builds that one scene's ``(C, H, W)`` stack, runs
sliding-window inference on it (the same ``predict_scene`` inference already
uses at eval time, one batch of patches on the GPU at a time), thresholds
and discards the low-confidence pixels, writes the kept result straight to
its own ``.npz``, and frees the stack before moving to the next scene.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from landscape_change_detection_pipeline.features.training_cache import (
    CACHE_FILENAME,
    NODATA_LABEL,
    upgrade_to_memmap,
    write_scene_cache,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UnlabeledScene:
    """One scene present in the raw tile store but not annotated."""

    tile_id: str
    sensor: str
    scene_id: str

    @property
    def scene_key(self) -> str:
        return f"{self.tile_id}_{self.sensor}_{self.scene_id}"


def discover_unlabeled_scenes(
    tile_dir: str | Path,
    mask_root: str | Path,
    sensors: Optional[Sequence[str]] = None,
) -> list[UnlabeledScene]:
    """Every scene in the raw tile store with no annotated mask, sorted by
    ``(tile_id, sensor, scene_id)``.

    Walks ``<tile_dir>/*.zarr`` for tile ids, then each tile's sensor groups
    (reading only the lightweight ``scene_ids`` attr, never array data --
    see :func:`landscape_change_detection_pipeline.scenes.zarr_store.read_sensor_group_attrs`),
    and excludes any ``(tile_id, sensor, scene_id)`` already annotated under
    ``mask_root`` (see
    :func:`landscape_change_detection_pipeline.features.training_cache.discover_annotated_scenes`).
    ``sensors``, when given, restricts discovery to those sensor groups only
    (e.g. to pseudo-label only the sensors already well represented in the
    annotated corpus).
    """
    from landscape_change_detection_pipeline.features.training_cache import discover_annotated_scenes
    from landscape_change_detection_pipeline.scenes.zarr_store import read_sensor_group_attrs

    tile_dir = Path(tile_dir)
    annotated_keys = {s.scene_key for s in discover_annotated_scenes(mask_root)}

    known_sensors = ("l5", "l7", "l8", "l9", "s2")
    allowed_sensors = {s.lower() for s in sensors} if sensors else None

    scenes: list[UnlabeledScene] = []
    for zarr_path in sorted(tile_dir.glob("*.zarr")):
        tile_id = zarr_path.stem
        for sensor in known_sensors:
            if allowed_sensors is not None and sensor not in allowed_sensors:
                continue
            try:
                attrs = read_sensor_group_attrs(tile_dir, tile_id, sensor)
            except KeyError:
                continue
            for scene_id in attrs.get("scene_ids", []):
                candidate = UnlabeledScene(tile_id=tile_id, sensor=sensor, scene_id=str(scene_id))
                if candidate.scene_key not in annotated_keys:
                    scenes.append(candidate)

    scenes.sort(key=lambda s: (s.tile_id, s.sensor, s.scene_id))
    return scenes


def _pseudo_cache_is_current(cache_dir: Path, expected_signature: str) -> bool:
    """Same up-to-date check as the real annotated cache (see
    :func:`landscape_change_detection_pipeline.features.training_cache.cache_is_current`) --
    duplicated rather than imported since a pseudo-label cache's signature
    additionally depends on the checkpoint and confidence threshold used to
    generate it, not just its source scene."""
    from landscape_change_detection_pipeline.features.training_cache import META_FILENAME
    import json

    npz_path = cache_dir / CACHE_FILENAME
    meta_path = cache_dir / META_FILENAME
    if not npz_path.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return meta.get("signature") == expected_signature


def generate_pseudo_labels(
    tile_dir: str | Path,
    mask_root: str | Path,
    pseudo_label_root: str | Path,
    model,
    checkpoint_id: str,
    mean: np.ndarray,
    std: np.ndarray,
    num_classes: int,
    index_names: tuple[str, ...],
    dem_layer_names: tuple[str, ...],
    include_doy_features: bool,
    include_latlon_features: bool,
    confidence_threshold: float = 0.9,
    max_scenes: Optional[int] = None,
    sensors: Optional[Sequence[str]] = None,
    patch_size: int = 256,
    stride: int = 128,
    inference_batch_size: int = 32,
    min_valid_ratio: float = 0.30,
    min_kept_pixel_ratio: float = 0.01,
    use_amp: bool = False,
    rare_classes: Sequence[int] = (),
    rare_confidence_threshold: float = 0.7,
    min_rare_pixels: int = 500,
    prefetch_workers: int = 3,
    label_remap: Optional[Sequence[int]] = None,
) -> list[Path]:
    """Score every unlabeled scene, threshold by confidence, and write the
    kept pixels to ``pseudo_label_root`` in the same per-scene ``.npz``
    schema real annotated caches use.

    One scene at a time: :func:`~landscape_change_detection_pipeline.features.training_cache.build_feature_stack`
    -> :func:`~landscape_change_detection_pipeline.inference.engine.predict_scene`
    (``return_probabilities=True``) -> threshold -> write, then the stack is
    dropped before the next scene -- see the module docstring's memory
    discipline. A scene whose kept-pixel ratio falls below
    ``min_kept_pixel_ratio`` after thresholding is skipped entirely (not
    worth a cache entry with almost no usable pixels).

    ``checkpoint_id`` (e.g. the checkpoint file's own signature/mtime-based
    id) is folded into each scene's cache signature alongside
    ``confidence_threshold``, so pseudo-labels regenerate automatically when
    either changes, exactly like a real cache invalidates on a source-file
    change.

    Returns the list of ``.npz`` paths actually (re)written -- an
    up-to-date pseudo-label cache is skipped, same contract as
    :func:`~landscape_change_detection_pipeline.features.training_cache.export_all_annotated_scenes`.
    """
    from landscape_change_detection_pipeline.features.training_cache import (
        build_feature_stack,
        scene_cache_signature,
        scene_year,
    )
    from landscape_change_detection_pipeline.inference.engine import predict_scene, scene_passes_precheck
    from landscape_change_detection_pipeline.scenes.zarr_store import zarr_path_for_tile

    unlabeled = discover_unlabeled_scenes(tile_dir, mask_root, sensors=sensors)
    logger.info("[pseudo-label] found %d unlabeled scenes under %s", len(unlabeled), tile_dir)
    if max_scenes is not None:
        unlabeled = unlabeled[: int(max_scenes)]

    feature_names = [
        *index_names,
        *dem_layer_names,
        *(("doy_sin", "doy_cos") if include_doy_features else ()),
        *(("lat_norm", "lon_norm") if include_latlon_features else ()),
    ]

    written: list[Path] = []
    total = len(unlabeled)

    def _prepare(scene: UnlabeledScene):
        """CPU/IO-bound half (cache check, zarr read, indices, precheck) --
        runs in a worker thread so the GPU is not left waiting on it."""
        cache_dir = Path(pseudo_label_root) / scene.tile_id / scene.sensor / scene.scene_id
        signature = scene_cache_signature(
            scene.scene_key,
            [zarr_path_for_tile(tile_dir, scene.tile_id)],
            [
                *feature_names, checkpoint_id, f"conf={confidence_threshold}",
                f"rare={sorted(rare_classes)}@{rare_confidence_threshold}", f"stride={stride}",
                *([f"merge={hash(tuple(label_remap))}"] if label_remap is not None else []),
            ],
        )
        if _pseudo_cache_is_current(cache_dir, signature):
            upgrade_to_memmap(cache_dir)
            return "current", cache_dir, signature, None
        try:
            built = build_feature_stack(
                tile_dir, scene.tile_id, scene.sensor, scene.scene_id,
                index_names, dem_layer_names, include_doy_features, include_latlon_features,
            )
        except Exception:  # noqa: BLE001 -- one bad scene must not abort the whole streaming run
            logger.warning("[pseudo-label] skipping %s: feature-stack build failed", scene.scene_key, exc_info=True)
            return "failed", cache_dir, signature, None
        if not scene_passes_precheck(built[0], min_valid_ratio=min_valid_ratio):
            return "invalid", cache_dir, signature, None
        return "ok", cache_dir, signature, built

    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    pending_writes: deque = deque()

    def _drain_writes(limit: int) -> None:
        while len(pending_writes) > limit:
            path, msg = pending_writes.popleft()
            written.append(path.result())
            print(msg, flush=True)

    with ThreadPoolExecutor(max_workers=max(1, prefetch_workers)) as loader, ThreadPoolExecutor(max_workers=max(2, prefetch_workers)) as writer:
        futures: deque = deque()
        it = iter(unlabeled)

        def _refill() -> None:
            while len(futures) < max(1, prefetch_workers) + 1:
                nxt = next(it, None)
                if nxt is None:
                    return
                futures.append((nxt, loader.submit(_prepare, nxt)))

        _refill()
        i = 0
        while futures:
            scene, fut = futures.popleft()
            _refill()
            i += 1
            print(f"[pseudo-label] [{i}/{total}] {scene.scene_key} ...", flush=True)
            status, cache_dir, signature, built = fut.result()
            if status == "current":
                print("    cache à jour, ignorée", flush=True)
                continue
            if status == "invalid":
                print("    ignorée : trop de pixels invalides", flush=True)
                continue
            if status == "failed":
                continue
            features, actual_feature_names, _provenance, transform, crs_wkt = built

            class_map, probabilities = predict_scene(
                model, features, num_classes, mean, std,
                patch_size=patch_size, stride=stride, batch_size=inference_batch_size,
                return_probabilities=True, nodata=NODATA_LABEL, use_amp=use_amp,
            )

            if label_remap is not None:
                class_map = np.asarray(label_remap, dtype=np.uint8)[class_map]
            confidence = probabilities.max(axis=0)
            threshold_map = np.full(confidence.shape, confidence_threshold, dtype=np.float32)
            is_rare = np.isin(class_map, list(rare_classes)) if rare_classes else np.zeros(class_map.shape, dtype=bool)
            threshold_map[is_rare] = rare_confidence_threshold
            keep = (confidence >= threshold_map) & (class_map != NODATA_LABEL)
            kept_ratio = float(keep.mean())
            rare_kept = int((keep & is_rare).sum())
            del probabilities
            if kept_ratio < min_kept_pixel_ratio and rare_kept < min_rare_pixels:
                logger.info(
                    "[pseudo-label] skipping %s: only %.2f%% of pixels pass the confidence threshold",
                    scene.scene_key, kept_ratio * 100.0,
                )
                print(f"    ignorée : {kept_ratio * 100:.2f}% de pixels confiants", flush=True)
                del features
                continue

            labels = np.where(keep, class_map, NODATA_LABEL).astype(np.uint8)
            fut_w = writer.submit(
                write_scene_cache, cache_dir, features, labels, actual_feature_names, signature,
                sensor=scene.sensor, tile_id=scene.tile_id,
                year=scene_year(scene.sensor, scene.scene_id), scene_id=scene.scene_id,
                transform=transform, crs_wkt=crs_wkt, memmap=True,
            )
            pending_writes.append((fut_w, f"    écrite {scene.scene_key} ({kept_ratio * 100:.1f}% gardés, {rare_kept} px classes rares)"))
            del features
            _drain_writes(limit=max(2, prefetch_workers) * 2)  # bounds memory held by queued writes
        _drain_writes(limit=0)

    return written
