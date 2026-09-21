#!/usr/bin/env python3
"""Stage 4 -- export MaskForge-annotated scenes into the training cache.

Purpose
-------
For every mask MaskForge has written back under ``config.features.mask_root``
(``{tile_id}_{sensor}_{scene_id}/mask.tif``), assemble the aligned
spectral-index + DEM feature stack and cache it as a per-scene ``.npz``
under ``config.features.train_root``, skipping any scene whose cache is
already up to date (see
``landscape_change_detection_pipeline.features.training_cache.cache_is_current``).

Usage
-----
    python scripts/04_export_training_cache.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landscape_change_detection_pipeline.classes.class_config import load_class_config  # noqa: E402
from landscape_change_detection_pipeline.config import load_config, resolved_feature_names  # noqa: E402
from landscape_change_detection_pipeline.features.training_cache import (  # noqa: E402
    discover_annotated_scenes,
    export_all_annotated_scenes,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument("--classes", default=None, help="Path to classes.yaml (default: configs/classes.yaml)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv if argv is not None else sys.argv[1:]))
    config = load_config(args.config, args.env_file)
    class_config = load_class_config(args.classes)

    annotated = discover_annotated_scenes(config.features.mask_root)
    print(f"[training-cache] found {len(annotated)} annotated scenes under {config.features.mask_root}")
    if not annotated:
        print("[training-cache] nothing to export")
        return 0

    index_names, dem_layer_names = resolved_feature_names(config.features)
    written = export_all_annotated_scenes(
        tile_dir=config.dem.tile_dir,
        mask_root=config.features.mask_root,
        train_root=config.features.train_root,
        class_config=class_config,
        index_names=index_names,
        dem_layer_names=dem_layer_names,
    )
    print(f"[training-cache] wrote {len(written)} scene caches to {config.features.train_root}")
    print(f"[training-cache] {len(annotated) - len(written)} scenes already up to date, skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
