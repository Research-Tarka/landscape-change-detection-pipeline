#!/usr/bin/env python3
"""Optional utility -- export the training cache as standalone GeoTIFFs.

Not a numbered pipeline stage: run any time after
``scripts/04_export_training_cache.py``, on your own cache or on a
``train_root`` someone else sent you.

Purpose
-------
``config.features.train_root`` (written by ``scripts/04_export_training_cache.py``)
holds one ``features.npz``/``features.json`` pair per annotated scene --
light, fast to load for training, but not directly openable in a GIS tool
and not self-explanatory to someone who does not know this pipeline's cache
format. This script reads every cached scene and writes it back out as two
plain, georeferenced GeoTIFFs (``features.tif`` -- one band per feature,
band descriptions set to the feature names -- and ``labels.tif`` -- the
class-id mask), so the training data can be handed to anyone (a
collaborator, another tool, QGIS for a visual sanity check) without them
needing to know anything about this pipeline.

This is a one-way export for sharing/inspection, not a cache format --
training itself (``scripts/05_train_model.py``) always reads
``features.npz`` directly.

Usage
-----
    python scripts/export_georeferenced_cache.py --config configs/config.yaml
    python scripts/export_georeferenced_cache.py --config configs/config.yaml --out-dir some/other/folder

Writes to ``config.features.geotiff_export_root`` by default (override with
``--out-dir``).

To inspect a cache someone else sent you without merging it into your own
``data/train_cache/`` first, point ``--config`` at a config whose
``features.train_root`` is that received folder directly (or a copy of it),
e.g. a throwaway ``configs/config_inspect.yaml`` with just
``features.train_root: path/to/their/train_cache`` overridden.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landscape_change_detection_pipeline.config import load_config  # noqa: E402
from landscape_change_detection_pipeline.features.training_cache import (  # noqa: E402
    export_train_root_as_geotiff,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write <tile_id>/<sensor>/<scene_id>/{features,labels}.tif into "
        "(default: config.features.geotiff_export_root)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv if argv is not None else sys.argv[1:]))
    config = load_config(args.config, args.env_file)
    out_dir = args.out_dir or config.features.geotiff_export_root

    print(f"[geotiff-export] reading cache from {config.features.train_root}")
    written = export_train_root_as_geotiff(config.features.train_root, out_dir)
    print(f"[geotiff-export] wrote {len(written)} scenes to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
