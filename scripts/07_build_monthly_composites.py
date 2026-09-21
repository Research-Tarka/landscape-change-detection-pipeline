#!/usr/bin/env python3
"""Stage 7 -- build monthly categorical composites from per-scene inference.

Purpose
-------
For every tile, group its Stage 6 per-scene class maps by
(year, month), reproject every sensor's scenes that month onto the finest
grid available (``composites.sensor_resolution_priority``), and reduce to
one composite per the configured per-class rule (median vs. any-occurrence,
``composites.class_rules``).

Usage
-----
    python scripts/07_build_monthly_composites.py --config configs/config.yaml
    python scripts/07_build_monthly_composites.py --tiles tile_0000_0000 --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landscape_change_detection_pipeline.classes.class_config import load_class_config  # noqa: E402
from landscape_change_detection_pipeline.config import load_config  # noqa: E402
from landscape_change_detection_pipeline.inference.composites import build_all_monthly_composites  # noqa: E402
from landscape_change_detection_pipeline.tiles.registry import read_registry  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument("--classes", default=None, help="Path to classes.yaml")
    parser.add_argument("--tiles", default=None, help="Comma-separated tile_ids to restrict to")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv if argv is not None else sys.argv[1:]))
    config = load_config(args.config, args.env_file)
    class_config = load_class_config(args.classes)
    num_classes = len(class_config.classes)
    comp_cfg = config.composites

    class_rules = {rule.class_id: (rule.rule, rule.priority) for rule in comp_cfg.class_rules}

    registry = read_registry(config.tiles.registry_path)
    tile_ids = args.tiles.split(",") if args.tiles else registry["tile_id"].tolist()

    total_written = 0
    for tile_id in tile_ids:
        written = build_all_monthly_composites(
            inference_root=config.inference.output_root,
            output_root=comp_cfg.output_root,
            tile_id=tile_id,
            num_classes=num_classes,
            sensor_priority=tuple(comp_cfg.sensor_resolution_priority),
            default_rule=comp_cfg.default_rule,
            class_rules=class_rules,
            overwrite=args.overwrite,
        )
        if written:
            print(f"[composites] {tile_id}: wrote {len(written)} monthly composites")
        total_written += len(written)

    print(f"[composites] {total_written} composites written across {len(tile_ids)} tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
