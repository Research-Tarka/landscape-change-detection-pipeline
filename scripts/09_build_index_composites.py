#!/usr/bin/env python3
"""Monthly per-tile spectral-index composites, for change-detection analysis.

Purpose
-------
For every tile, computes NDVI/NDSI/NBR/NDWI/Tasseled-Cap/etc. (see
:mod:`landscape_change_detection_pipeline.change.spectral_composites`) per scene from
this pipeline's own stored reflectance (never the classification outputs),
excludes cloud/shadow pixels via each scene's own Stage 6 classification,
and reduces each index to median/min/max/n_obs per tile-month. This is the
analysis-only counterpart of Stage 7's categorical class composites --
nothing here feeds the land-cover model. Output:
``outputs/index_composites/<tile_id>/<year>-<month>/indices.npz``.

Usage
-----
    python scripts/09_build_index_composites.py --config configs/config.yaml
    python scripts/09_build_index_composites.py --tiles tile_0000_0000 --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landscape_change_detection_pipeline.classes.class_config import load_class_config  # noqa: E402
from landscape_change_detection_pipeline.config import load_config  # noqa: E402
from landscape_change_detection_pipeline.change.spectral_composites import build_all_month_index_composites  # noqa: E402
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

    registry = read_registry(config.tiles.registry_path)
    tile_ids = args.tiles.split(",") if args.tiles else registry["tile_id"].tolist()

    total_written = 0
    for tile_id in tile_ids:
        written = build_all_month_index_composites(
            tile_dir=config.dem.tile_dir,
            inference_root=config.inference.output_root,
            output_root=config.index_composites.output_root,
            tile_id=tile_id,
            class_config=class_config,
            overwrite=args.overwrite,
        )
        if written:
            print(f"[index_composites] {tile_id}: wrote {len(written)} monthly index composites")
        total_written += len(written)

    print(f"[index_composites] {total_written} index composites written across {len(tile_ids)} tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
