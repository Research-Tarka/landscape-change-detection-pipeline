#!/usr/bin/env python3
"""Stage 8 -- mosaic per-tile monthly composites into one AOI-wide raster.

Purpose
-------
Every stage through Stage 7 stays tile-scoped: each tile's monthly composite
covers its own buffered analysis window, and neighbouring tiles' windows
overlap. This stage crops every tile's composite down to its own unbuffered
``search_bbox`` (discarding the buffer) and places the crops side by side
into one continuous, georeferenced raster per period across the whole AOI,
stored as ``.npz`` like every other intermediate array in this pipeline --
not a GeoTIFF; use ``scripts/export_gui.py`` on demand to get a
GIS-readable file for a specific period. See
:mod:`landscape_change_detection_pipeline.mosaic.mosaic` for the full design.

Usage
-----
    python scripts/08_build_mosaics.py --config configs/config.yaml
    python scripts/08_build_mosaics.py --periods 2000-04,2000-12 --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landscape_change_detection_pipeline.config import load_config  # noqa: E402
from landscape_change_detection_pipeline.mosaic.mosaic import (  # noqa: E402
    build_period_mosaic,
    discover_periods,
    load_period_tile_composites,
    mosaic_output_path,
    write_mosaic,
)
from landscape_change_detection_pipeline.tiles.registry import read_registry  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument("--periods", default=None, help="Comma-separated 'YYYY-MM' periods to restrict to")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv if argv is not None else sys.argv[1:]))
    config = load_config(args.config, args.env_file)
    mosaic_cfg = config.mosaic
    comp_cfg = config.composites

    registry = read_registry(config.tiles.registry_path)
    tile_ids = registry["tile_id"].tolist()

    periods = args.periods.split(",") if args.periods else discover_periods(comp_cfg.output_root, tile_ids)

    written = 0
    for period in periods:
        out_path = mosaic_output_path(mosaic_cfg.output_root, period)
        if out_path.is_file() and not args.overwrite:
            continue
        tiles = load_period_tile_composites(comp_cfg.output_root, registry, period)
        if not tiles:
            print(f"[mosaic] {period}: no tile composites found, skipping")
            continue
        result = build_period_mosaic(tiles, registry)
        write_mosaic(out_path, result)
        print(
            f"[mosaic] {period}: wrote mosaic from {len(result['tiles_present'])} tiles "
            f"at {result['resolution_m']}m -> {out_path}"
        )
        written += 1

    print(f"[mosaic] {written} mosaics written across {len(periods)} periods")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
