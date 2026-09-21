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
    python scripts/06_run_inference.py --config configs/config.yaml --checkpoint models/checkpoints/unet_best.pt
    python scripts/06_run_inference.py --tiles tile_0000_0000 --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landscape_change_detection_pipeline.config import load_config, resolved_feature_names  # noqa: E402
from landscape_change_detection_pipeline.inference.engine import (  # noqa: E402
    DEFAULT_CLASS_PRIORITY_ORDER,
    predict_scene,
    scene_output_paths,
    scene_passes_precheck,
    write_class_map,
)
from landscape_change_detection_pipeline.inference.model_loader import load_model_for_inference  # noqa: E402
from landscape_change_detection_pipeline.scenes.zarr_store import read_sensor_group_attrs  # noqa: E402
from landscape_change_detection_pipeline.tiles.registry import read_registry  # noqa: E402

SENSORS = ("l5", "l7", "l8", "l9", "s2")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to a checkpoint (default: "
            "<training.checkpoint_dir>/<model.type>_best.pt, i.e. whatever "
            "scripts/05_train_model.py last wrote for the configured model type)"
        ),
    )
    parser.add_argument("--tiles", default=None, help="Comma-separated tile_ids to restrict to")
    parser.add_argument("--sensors", default=None, help="Comma-separated sensor keys (default: all)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def _build_feature_stack_for_scene(
    tile_dir,
    tile_id: str,
    sensor: str,
    scene_id: str,
    feature_names: tuple[str, ...],
    index_names: tuple[str, ...],
    dem_layer_names: tuple[str, ...],
):
    """Same feature-stack construction as
    ``landscape_change_detection_pipeline.features.training_cache.build_feature_stack``,
    duplicated here (rather than imported) only to avoid re-deriving
    ``index_names``/``dem_layer_names`` split from ``feature_names`` -- the
    checkpoint's own ``feature_names`` list is the source of truth for what
    order inference must reproduce; ``index_names``/``dem_layer_names`` come
    from ``config.features`` and must match what the checkpoint was trained
    with, or the mismatch check below raises."""
    from landscape_change_detection_pipeline.features.training_cache import build_feature_stack

    stack, built_names, _provenance, _transform, _crs_wkt = build_feature_stack(
        tile_dir, tile_id, sensor, scene_id, index_names=index_names, dem_layer_names=dem_layer_names
    )
    if tuple(built_names) != tuple(feature_names):
        raise ValueError(
            f"Feature order mismatch for {tile_id}/{sensor}/{scene_id}: checkpoint expects "
            f"{feature_names}, got {built_names}"
        )
    return stack


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv if argv is not None else sys.argv[1:]))
    config = load_config(args.config, args.env_file)
    inf_cfg = config.inference

    checkpoint_path = args.checkpoint or str(Path(config.training.checkpoint_dir) / f"{config.model.type}_best.pt")
    if not Path(checkpoint_path).is_file():
        print(
            f"[inference] no checkpoint at '{checkpoint_path}' -- train one first "
            f"(scripts/05_train_model.py) or pass --checkpoint explicitly"
        )
        return 1

    loaded = load_model_for_inference(checkpoint_path, device=args.device)
    print(f"[inference] loaded checkpoint with {loaded.num_classes} classes, features={loaded.feature_names}")
    index_names, dem_layer_names = resolved_feature_names(config.features)

    registry = read_registry(config.tiles.registry_path)
    tile_ids = args.tiles.split(",") if args.tiles else registry["tile_id"].tolist()
    sensors = tuple(args.sensors.split(",")) if args.sensors else SENSORS

    n_ok = n_skip = n_error = 0
    for tile_id in tile_ids:
        for sensor in sensors:
            try:
                attrs = read_sensor_group_attrs(config.dem.tile_dir, tile_id, sensor)
            except KeyError:
                continue
            scene_ids = list(attrs.get("scene_ids", []))
            transform = attrs["transform"]
            crs_wkt = attrs["crs_wkt"]

            for scene_id in scene_ids:
                out_path = scene_output_paths(inf_cfg.output_root, tile_id, sensor, scene_id)
                if out_path.is_file() and not args.overwrite:
                    n_skip += 1
                    continue
                try:
                    features = _build_feature_stack_for_scene(
                        config.dem.tile_dir, tile_id, sensor, scene_id, loaded.feature_names,
                        index_names, dem_layer_names,
                    )
                    if not scene_passes_precheck(features, inf_cfg.scene_precheck_min_valid_ratio):
                        n_skip += 1
                        continue

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
                        priority_order=tuple(inf_cfg.class_priority_order) or DEFAULT_CLASS_PRIORITY_ORDER,
                    )
                    write_class_map(out_path, class_map, transform, crs_wkt)
                    n_ok += 1
                except Exception as exc:  # noqa: BLE001 -- reported per scene, never fatal
                    print(f"[inference] ERROR {tile_id}/{sensor}/{scene_id}: {type(exc).__name__}: {exc}")
                    n_error += 1

    print(f"[inference] {n_ok} ok, {n_skip} skipped, {n_error} errored")
    return 1 if n_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
