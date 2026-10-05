#!/usr/bin/env python3
"""Stage 6 -- sliding-window inference over stored scenes.

Purpose
-------
For every scene stored in each tile's zarr store, build the same feature
stack Stage 4 builds for training, run the trained checkpoint through
sliding-window inference, and write the per-scene class map. Idempotent:
a scene whose output already exists is skipped unless ``--overwrite``.

Usage
-----
    python scripts/06_run_inference.py --config configs/config.yaml

Every run parameter (checkpoint path, tiles/sensors to restrict to, device,
worker-process count, overwrite) is read from ``config.inference`` -- see
:class:`landscape_change_detection_pipeline.config.InferenceConfig` -- not
from CLI flags, so a run is fully reproducible from ``config.yaml`` alone.
Only ``--config``/``--env-file`` (which config file to read) stay as flags.
"""

from __future__ import annotations

import os

# Windows/conda: torch (libiomp5md) and numpy/MKL (libomp) both load an OpenMP
# runtime, which kills the spawned pool workers with "OMP: Error #15". Must be
# set before numpy/torch are imported; workers inherit it.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
import threading
from collections import Counter, OrderedDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landscape_change_detection_pipeline.classes.class_config import (  # noqa: E402
    label_remap_table,
    load_class_config,
)
from landscape_change_detection_pipeline.config import (  # noqa: E402
    all_resolved_feature_names,
    load_config,
    resolved_feature_names,
)
from landscape_change_detection_pipeline.inference.engine import (  # noqa: E402
    default_priority_order,
    predict_scene,
    predict_scene_sklearn,
    scene_output_paths,
    scene_passes_precheck,
    write_class_map,
)
from landscape_change_detection_pipeline.inference.model_loader import load_model_for_inference  # noqa: E402
from landscape_change_detection_pipeline.scenes.zarr_store import read_sensor_group_attrs  # noqa: E402

SENSORS = ("l5", "l7", "l8", "l9", "s2")

#: Per-process globals set once by _worker_init, so each worker loads its own
#: model/config a single time (not once per scene) -- see _worker_init's own
#: docstring for why this can't just be a closure captured by the
#: ProcessPoolExecutor submit calls (the worker function must be
#: picklable/importable at module level).
_worker_state: dict = {}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    return parser.parse_args(argv)


#: Small LRU of (tile_id, layer_name) -> read_tile_dem result, guarded by
#: _dem_cache_lock. In the ProcessPoolExecutor path (torch models) each
#: process has its own copy of this module, so the cache is effectively
#: per-worker and per-tile (scene_plan groups scenes by tile, so a single
#: entry per layer is normally enough). In the ThreadPoolExecutor path
#: (non-torch models) all threads share this one dict, and different threads
#: can genuinely be working different tiles at once -- hence an LRU with a
#: few slots (one per in-flight worker/thread would suffice, a bit of slack
#: costs little next to a 716MB CatBoost checkpoint) and a lock, rather than
#: the single-slot cache a purely per-process design could get away with.
_DEM_CACHE_MAX_ENTRIES = 32
_dem_cache: "OrderedDict[tuple[str, str], object]" = OrderedDict()
_dem_cache_lock = threading.Lock()


def _cached_read_tile_dem(tile_dir, tile_id: str, name: str):
    """``read_tile_dem`` result, memoized (see ``_dem_cache`` above) instead
    of re-reading the *entire* tile DEM (every layer, full ``(H, W)`` array)
    from zarr on every single scene -- it never changes across a tile's
    scenes, so this turns "one full DEM read per layer per scene" into "one
    per layer per tile" -- a major avoidable driver of this script's
    memory/IO footprint, since a worker/thread typically processes many
    scenes per tile back to back."""
    key = (tile_id, name)
    with _dem_cache_lock:
        cached = _dem_cache.get(key)
        if cached is not None:
            _dem_cache.move_to_end(key)
            return cached

    from landscape_change_detection_pipeline.dem.zarr_store import read_tile_dem

    result = read_tile_dem(tile_dir, tile_id, name)
    with _dem_cache_lock:
        _dem_cache[key] = result
        _dem_cache.move_to_end(key)
        while len(_dem_cache) > _DEM_CACHE_MAX_ENTRIES:
            _dem_cache.popitem(last=False)
    return result


def _build_feature_stack_for_scene(
    tile_dir,
    tile_id: str,
    sensor: str,
    scene_id: str,
    feature_names: tuple[str, ...],
    index_names: tuple[str, ...],
    dem_layer_names: tuple[str, ...],
    include_doy_features: bool,
    include_latlon_features: bool,
):
    """Mirrors
    ``landscape_change_detection_pipeline.features.training_cache.build_feature_stack``
    (spectral indices, then DEM layers, then optional DOY/lat-lon channels,
    in that fixed order), duplicated here rather than calling it directly so
    the DEM layers go through ``_cached_read_tile_dem`` instead of
    ``training_cache``'s own ``read_tile_dem`` call -- see that cache's own
    docstring for why (avoids re-reading the whole tile DEM per scene). A
    previous version of this function monkeypatched
    ``training_cache.read_tile_dem`` for the call's duration instead; that is
    not thread-safe (the ThreadPoolExecutor path below runs this
    concurrently from several threads sharing the same module object), so
    the assembly is inlined here instead.
    ``index_names``/``dem_layer_names``/the DOY/lat-lon flags come from
    ``config.features`` and must match what the checkpoint was trained
    with, or the mismatch check below raises."""
    import numpy as np

    from landscape_change_detection_pipeline.features.scene_context import scene_context_vector
    from landscape_change_detection_pipeline.features.spectral_indices import (
        DOY_FEATURE_NAMES,
        LATLON_FEATURE_NAMES,
        compute_indices,
        dem_layer_to_channels,
        doy_cyclical_stack,
        expand_dem_feature_names,
        latlon_stack,
    )
    from landscape_change_detection_pipeline.features.training_cache import (
        _resample_dem_to_sensor_grid,
        bands_for_scene,
        scene_date,
    )

    bands, band_provenance = bands_for_scene(tile_dir, tile_id, sensor, scene_id)
    index_stack, _index_provenance = compute_indices(bands, band_provenance, names=index_names, sensor=sensor)
    dst_shape = index_stack.shape[1:]

    attrs = read_sensor_group_attrs(tile_dir, tile_id, sensor)
    dst_transform = attrs["transform"]
    dst_crs = attrs["crs_wkt"]

    dem_arrays = []
    for name in dem_layer_names:
        arr, dem_transform, dem_crs = _cached_read_tile_dem(tile_dir, tile_id, name)
        if tuple(arr.shape) != tuple(dst_shape):
            arr = _resample_dem_to_sensor_grid(arr, dem_transform, dem_crs, dst_shape, dst_transform, dst_crs)
        dem_arrays.extend(dem_layer_to_channels(name, arr))
    dem_stack = np.stack(dem_arrays, axis=0).astype(np.float32) if dem_arrays else np.zeros(
        (0, *dst_shape), dtype=np.float32
    )

    stacks = [index_stack, dem_stack]
    built_names = [*index_names, *expand_dem_feature_names(dem_layer_names)]

    if include_doy_features:
        acquisition_date = scene_date(sensor, scene_id)
        stacks.append(doy_cyclical_stack(acquisition_date, dst_shape))
        built_names.extend(DOY_FEATURE_NAMES)

    if include_latlon_features:
        stacks.append(latlon_stack(dst_transform, dst_crs, dst_shape))
        built_names.extend(LATLON_FEATURE_NAMES)

    stack = np.concatenate(stacks, axis=0).astype(np.float32)
    if tuple(built_names) != tuple(feature_names):
        raise ValueError(
            f"Feature order mismatch for {tile_id}/{sensor}/{scene_id}: checkpoint expects "
            f"{feature_names}, got {built_names}"
        )
    return stack, scene_context_vector(sensor, scene_id, dst_transform, dst_crs, tuple(dst_shape))


def _worker_init(
    config_path: Optional[str],
    env_file: Optional[str],
    checkpoint_path: str,
    device: Optional[str],
) -> None:
    """Runs once per worker process (``ProcessPoolExecutor(initializer=...)``):
    loads the config and checkpoint a single time and stashes them in this
    process's own module-global, so ``_process_one_scene`` below never
    reloads either per scene. Each worker gets its own full copy of the
    model (no cross-process sharing) -- fine for a CPU model, and the reason
    ``--workers`` should stay small for a GPU one (see its own help text)."""
    config = load_config(config_path, env_file)
    class_config = load_class_config()
    loaded = load_model_for_inference(checkpoint_path, device=device)
    if not loaded.is_torch:
        loaded.feature_names = all_resolved_feature_names(config.features)
    index_names, dem_layer_names = resolved_feature_names(config.features)
    _worker_state.update(
        config=config,
        loaded=loaded,
        index_names=index_names,
        dem_layer_names=dem_layer_names,
        class_config=class_config,
    )


def _process_one_scene_impl(
    config,
    loaded,
    index_names: tuple[str, ...],
    dem_layer_names: tuple[str, ...],
    class_config,
    tile_id: str,
    sensor: str,
    scene_id: str,
    transform,
    crs_wkt: str,
    overwrite: bool,
) -> tuple[str, str, str, str, str]:
    """One scene's full build-features -> infer -> write pipeline. Shared by
    both execution modes below: the ``ProcessPoolExecutor`` path (torch
    models) wraps this to pull ``config``/``loaded``/... from this worker
    process's own ``_worker_state``, and the ``ThreadPoolExecutor`` path
    (non-torch models) calls it directly with the single model instance
    loaded once in the main process and shared read-only across threads.
    Returns ``(tile_id, sensor, scene_id, status, detail)`` with ``status``
    one of ``"ok"``, ``"skip"``, ``"error"`` -- the caller does all
    printing/counting, so this stays a plain, picklable/thread-safe result
    rather than interleaved prints from several workers at once."""
    inf_cfg = config.inference

    out_path = scene_output_paths(inf_cfg.output_root, tile_id, sensor, scene_id)
    if out_path.is_file() and not overwrite:
        return tile_id, sensor, scene_id, "skip", "already done"

    try:
        features, context = _build_feature_stack_for_scene(
            config.dem.tile_dir, tile_id, sensor, scene_id, loaded.feature_names,
            index_names, dem_layer_names,
            config.features.include_doy_features, config.features.include_latlon_features,
        )
        if not scene_passes_precheck(features, inf_cfg.scene_precheck_min_valid_ratio):
            return tile_id, sensor, scene_id, "skip", "failed precheck"

        if loaded.is_torch:
            class_map = predict_scene(
                loaded.model,
                features,
                loaded.num_classes,
                loaded.mean,
                loaded.std,
                patch_size=inf_cfg.patch_size,
                stride=inf_cfg.stride,
                batch_size=inf_cfg.batch_size,
                ambiguity_threshold=inf_cfg.ambiguity_threshold or 0.0,
                priority_order=tuple(inf_cfg.class_priority_order) or default_priority_order([c.name for c in class_config.classes]),
                context=context,
            )
        else:
            class_map = predict_scene_sklearn(
                loaded.model,
                features,
                loaded.num_classes,
                ambiguity_threshold=inf_cfg.ambiguity_threshold or 0.0,
                priority_order=tuple(inf_cfg.class_priority_order) or default_priority_order([c.name for c in class_config.classes]),
            )
        # predict_scene[_sklearn] return dense class ids (0..N-1, model
        # output-channel order) -- class_map.npz is a boundary artifact read
        # by other tools (MaskForge's copy-inference, QA scripts) that only
        # know the yaml's stable raw ids (see ClassConfig's docstring), so
        # convert back to raw ids before writing. NODATA_VALUE (255) is not
        # a valid dense id and must pass through unchanged.
        nodata_value = 255
        merge_table = label_remap_table([c.name for c in class_config.classes], config.training.class_merge)
        if merge_table is not None:
            # Safety net: fold a merged-away class (e.g. from a checkpoint
            # trained before the merge) onto its target class.
            class_map = np.asarray(merge_table, dtype=np.uint8)[class_map]
        raw_class_map =class_config.remap_dense_to_raw(class_map, nodata_value=nodata_value)
        write_class_map(out_path, raw_class_map, transform, crs_wkt, nodata=nodata_value)
        return tile_id, sensor, scene_id, "ok", ""
    except Exception as exc:  # noqa: BLE001 -- reported per scene, never fatal
        return tile_id, sensor, scene_id, "error", f"{type(exc).__name__}: {exc}"


def _process_one_scene(
    tile_id: str, sensor: str, scene_id: str, transform, crs_wkt: str, overwrite: bool
) -> tuple[str, str, str, str, str]:
    """``ProcessPoolExecutor`` entry point (torch models): pulls this
    worker's own ``config``/``loaded``/... -- set once by ``_worker_init`` --
    out of ``_worker_state`` and delegates to ``_process_one_scene_impl``."""
    config = _worker_state["config"]
    loaded = _worker_state["loaded"]
    index_names = _worker_state["index_names"]
    dem_layer_names = _worker_state["dem_layer_names"]
    class_config = _worker_state["class_config"]
    return _process_one_scene_impl(
        config, loaded, index_names, dem_layer_names, class_config,
        tile_id, sensor, scene_id, transform, crs_wkt, overwrite,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv if argv is not None else sys.argv[1:]))
    config = load_config(args.config, args.env_file)
    class_config = load_class_config()
    inf_cfg = config.inference

    non_torch_types = ("threshold", "random_forest", "catboost", "lightgbm")
    default_ext = "joblib" if config.model.type in non_torch_types else "pt"
    checkpoint_path = inf_cfg.checkpoint_path or str(
        Path(config.training.checkpoint_dir) / f"{config.model.type}_best.{default_ext}"
    )
    if not Path(checkpoint_path).is_file():
        print(
            f"[inference] no checkpoint at '{checkpoint_path}' -- train one first "
            f"(scripts/05_train_model.py) or set inference.checkpoint_path explicitly"
        )
        return 1

    loaded = load_model_for_inference(checkpoint_path, device=inf_cfg.device)
    index_names, dem_layer_names = resolved_feature_names(config.features)
    if not loaded.is_torch:
        # Non-torch checkpoints (.joblib) carry no embedded feature-name
        # metadata (see model_loader._load_joblib_checkpoint), so the
        # feature order to validate scenes against comes from this run's
        # own config instead of the checkpoint -- correct as long as the
        # config used for inference matches the one training was run with.
        loaded.feature_names = all_resolved_feature_names(config.features)
    print(f"[inference] loaded checkpoint with {loaded.num_classes} classes, features={loaded.feature_names}")

    if inf_cfg.tiles:
        tile_ids = list(inf_cfg.tiles)
    else:
        # Not the tile registry: the registry only ever holds the tiles from
        # the most recent scripts/01_build_tiles.py run (one pilot_bbox/AOI
        # at a time, overwritten each run), so a workflow that downloads
        # several bboxes one at a time would silently skip every tile from
        # an earlier bbox. Discover tiles instead from what's actually on
        # disk in config.dem.tile_dir (one <tile_id>.zarr store per tile
        # scripts/03_download_scenes.py has written), so inference always
        # covers everything downloaded so far, regardless of registry state.
        tile_dir = Path(config.dem.tile_dir)
        tile_ids = sorted(p.stem for p in tile_dir.glob("*.zarr") if p.is_dir())
    sensors = tuple(inf_cfg.sensors) or SENSORS

    # Pre-count every (tile, sensor, scene) this run will visit, so progress
    # can be reported as "i/total" below -- a second, cheap pass over the
    # same zarr attrs (no feature stacks built yet), not a rescan of any
    # heavy data.
    scene_plan: list[tuple[str, str, str, object, str]] = []
    for tile_id in tile_ids:
        for sensor in sensors:
            try:
                attrs = read_sensor_group_attrs(config.dem.tile_dir, tile_id, sensor)
            except KeyError:
                continue
            for scene_id in attrs.get("scene_ids", []):
                scene_plan.append((tile_id, sensor, scene_id, attrs["transform"], attrs["crs_wkt"]))
    total = len(scene_plan)
    print(f"[inference] {total} scenes to visit across {len(tile_ids)} tile(s), workers={inf_cfg.workers}")

    # Remaining-scene counters per (tile, sensor) and per tile, so "finished
    # sensor/tile" can be reported correctly regardless of completion order
    # -- required once --workers > 1, since parallel completions don't
    # arrive in scene_plan order the way the old sequential loop's
    # next-item lookahead assumed.
    remaining_by_sensor: Counter = Counter()
    remaining_by_tile: Counter = Counter()
    for tile_id, sensor, _scene_id, _t, _c in scene_plan:
        remaining_by_sensor[(tile_id, sensor)] += 1
        remaining_by_tile[tile_id] += 1

    n_ok = n_skip = n_error = 0
    n_done = 0

    def _report(tile_id: str, sensor: str, scene_id: str, status: str, detail: str) -> None:
        nonlocal n_ok, n_skip, n_error, n_done
        n_done += 1
        if status == "skip":
            n_skip += 1
            print(f"[inference] ({n_done}/{total}) skip ({detail}) {tile_id}/{sensor}/{scene_id}")
        elif status == "error":
            n_error += 1
            print(f"[inference] ({n_done}/{total}) ERROR {tile_id}/{sensor}/{scene_id}: {detail}")
        else:
            n_ok += 1
            print(f"[inference] ({n_done}/{total}) ok {tile_id}/{sensor}/{scene_id}")

        remaining_by_sensor[(tile_id, sensor)] -= 1
        if remaining_by_sensor[(tile_id, sensor)] == 0:
            print(f"[inference] finished sensor {sensor} for tile {tile_id}")
        remaining_by_tile[tile_id] -= 1
        if remaining_by_tile[tile_id] == 0:
            print(f"[inference] finished tile {tile_id}")

    if inf_cfg.workers <= 1:
        for tile_id, sensor, scene_id, transform, crs_wkt in scene_plan:
            _, _, _, status, detail = _process_one_scene_impl(
                config, loaded, index_names, dem_layer_names, class_config,
                tile_id, sensor, scene_id, transform, crs_wkt, inf_cfg.overwrite,
            )
            _report(tile_id, sensor, scene_id, status, detail)
    elif not loaded.is_torch:
        # Non-torch checkpoints (catboost/random_forest/lightgbm/threshold)
        # can be hundreds of MB to GBs once deserialized (the catboost
        # checkpoint that motivated this: ~700MB on disk, more in memory).
        # ProcessPoolExecutor would give every one of inf_cfg.workers
        # processes its own full copy of that model (loaded independently by
        # _worker_init in each), so RAM scales with workers before a single
        # scene is even processed. These model types' predict_proba runs in
        # C++ and releases the GIL, so a ThreadPoolExecutor gets the same
        # scene-level parallelism from the *one* `loaded` model instance
        # already loaded in this process above, shared read-only across
        # threads -- no duplication.
        prefetch_margin = 2
        max_in_flight = inf_cfg.workers + prefetch_margin
        with ThreadPoolExecutor(max_workers=inf_cfg.workers) as pool:
            pending: set = set()
            scene_iter = iter(scene_plan)

            def _submit_next() -> bool:
                item = next(scene_iter, None)
                if item is None:
                    return False
                tile_id, sensor, scene_id, transform, crs_wkt = item
                pending.add(
                    pool.submit(
                        _process_one_scene_impl,
                        config, loaded, index_names, dem_layer_names, class_config,
                        tile_id, sensor, scene_id, transform, crs_wkt, inf_cfg.overwrite,
                    )
                )
                return True

            for _ in range(max_in_flight):
                if not _submit_next():
                    break

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    tile_id, sensor, scene_id, status, detail = future.result()
                    _report(tile_id, sensor, scene_id, status, detail)
                    _submit_next()
    else:
        # Torch models: each worker process needs its own model instance
        # anyway (CPU tensors aren't safely shared across processes without
        # extra plumbing, and a GPU device context is per-process), so this
        # stays a ProcessPoolExecutor with _worker_init loading the
        # checkpoint once per process -- keep inf_cfg.workers small for a
        # GPU model, per _worker_init's own docstring.
        #
        # Cap how many scenes are in flight at once instead of submitting the
        # whole scene_plan up front: submitting everything immediately queues
        # every scene's feature-stack build (each one several GB in memory)
        # behind the pool, not just the ones workers are actually crunching
        # on right now, and RAM usage balloons well past what inf_cfg.workers
        # implies. A sliding window of "workers + small prefetch margin"
        # in-flight tasks keeps each worker busy on its current scene plus a
        # couple queued ahead, without ever materializing more than that in
        # memory at once.
        prefetch_margin = 2
        max_in_flight = inf_cfg.workers + prefetch_margin
        with ProcessPoolExecutor(
            max_workers=inf_cfg.workers,
            initializer=_worker_init,
            initargs=(args.config, args.env_file, checkpoint_path, inf_cfg.device),
        ) as pool:
            pending: set = set()
            scene_iter = iter(scene_plan)

            def _submit_next() -> bool:
                item = next(scene_iter, None)
                if item is None:
                    return False
                tile_id, sensor, scene_id, transform, crs_wkt = item
                pending.add(
                    pool.submit(_process_one_scene, tile_id, sensor, scene_id, transform, crs_wkt, inf_cfg.overwrite)
                )
                return True

            for _ in range(max_in_flight):
                if not _submit_next():
                    break

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    tile_id, sensor, scene_id, status, detail = future.result()
                    _report(tile_id, sensor, scene_id, status, detail)
                    _submit_next()

    print(f"[inference] {n_ok} ok, {n_skip} skipped, {n_error} errored")
    return 1 if n_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
