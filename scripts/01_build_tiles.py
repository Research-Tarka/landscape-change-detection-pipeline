#!/usr/bin/env python3
"""Stage 1 -- tile the AOI and write the tile registry + split assignment (Prompts 01, 06).

Purpose
-------
Cover the AOI (``config.tiles.aoi_path``) with a regular grid of square
tiles, write the tile registry, then distribute tiles across
``config.tiles.n_splits`` batches (one batch per parallel GEE
project/worker used by later download/inference stages) via the
area-balanced LPT splitter.

Usage
-----
    python scripts/01_build_tiles.py --config configs/config.yaml
    python scripts/01_build_tiles.py --pilot-bbox 550000 6100000 560000 6110000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landscape_change_detection_pipeline.config import load_config  # noqa: E402
from landscape_change_detection_pipeline.tiles.registry import build_registry, write_registry  # noqa: E402
from landscape_change_detection_pipeline.tiles.splitter import (  # noqa: E402
    assign_lpt,
    assignment_table,
    balance_report,
    write_assignment,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument(
        "--pilot-bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="Restrict tiling to this bbox (aoi_crs units), overriding config.tiles.pilot_bbox",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv if argv is not None else sys.argv[1:]))
    config = load_config(args.config, args.env_file)
    tiles_cfg = config.tiles

    pilot_bbox = args.pilot_bbox if args.pilot_bbox is not None else tiles_cfg.pilot_bbox

    print(f"[tiles] building registry from AOI '{tiles_cfg.aoi_path}' (layer={tiles_cfg.aoi_layer})")
    registry = build_registry(
        tiles_cfg.aoi_path,
        tiles_cfg.aoi_layer,
        tiles_cfg.aoi_crs,
        tile_size_m=tiles_cfg.tile_size_m,
        buffer_m=tiles_cfg.buffer_m,
        pilot_bbox=pilot_bbox,
        tile_id_prefix=tiles_cfg.tile_id_prefix,
    )
    out_path = write_registry(registry, tiles_cfg.registry_path)
    print(f"[tiles] wrote {len(registry)} tiles to {out_path}")

    loads = assign_lpt(
        registry["tile_id"].tolist(),
        registry["area_km2"].tolist(),
        n_splits=tiles_cfg.n_splits,
    )
    assignment = assignment_table(loads)
    assignment_path = write_assignment(assignment, tiles_cfg.split_assignment_path)
    print(f"[tiles] wrote split assignment ({tiles_cfg.n_splits} splits) to {assignment_path}")
    print(balance_report(loads).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
