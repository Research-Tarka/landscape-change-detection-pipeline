#!/usr/bin/env python3
"""Stage 2 -- fetch and store the per-tile DEM products.

Purpose
-------
For every tile in the registry (Stage 1), fetch the best available source
DEM over the tile's buffered analysis window (MRDEM-30 DTM, Copernicus
GLO-30 fallback), derive slope/aspect, and write everything into the tile's
zarr store. The Copernicus fallback is served through Earth Engine
(``COPERNICUS/DEM/GLO30_2024_1``), so a GEE project is initialized once
up front even though most tiles are expected to resolve via the direct
MRDEM-30 COG path and never touch it.

Usage
-----
    python scripts/02_download_dem.py --config configs/config.yaml
    python scripts/02_download_dem.py --tiles tile_0000_0000,tile_0000_0001
    python scripts/02_download_dem.py --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landscape_change_detection_pipeline.config import load_config  # noqa: E402
from landscape_change_detection_pipeline.dem.engine import run_dem_extraction  # noqa: E402
from landscape_change_detection_pipeline.scenes.gee_auth import initialize_ee  # noqa: E402
from landscape_change_detection_pipeline.tiles.registry import read_registry  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument("--tiles", default=None, help="Comma-separated tile_ids to restrict to")
    parser.add_argument("--overwrite", action="store_true", help="Recompute tiles already marked done")
    parser.add_argument("--ee-project", default=None, help="GEE project for the Copernicus GLO-30 fallback")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv if argv is not None else sys.argv[1:]))
    config = load_config(args.config, args.env_file)
    dem_cfg = config.dem

    project = args.ee_project or (config.gee.projects[0] if config.gee.projects else None) or config.gee.project
    initialize_ee(project=project, config_default=config.gee.project, verify=True)

    registry = read_registry(config.tiles.registry_path)
    tile_ids = args.tiles.split(",") if args.tiles else None

    results = run_dem_extraction(
        registry,
        dem_cfg.tile_dir,
        tile_ids=tile_ids,
        overwrite=args.overwrite,
        copernicus_fallback=dem_cfg.use_copernicus_fallback,
        force_copernicus=dem_cfg.force_copernicus,
    )

    n_ok = sum(1 for _, status in results if status == "ok")
    n_skip = sum(1 for _, status in results if status.startswith("skip"))
    n_error = sum(1 for _, status in results if status.startswith("error"))
    for tile_id, status in results:
        if status.startswith("error"):
            print(f"[dem] {tile_id}: {status}")

    print(f"[dem] {n_ok} ok, {n_skip} skipped, {n_error} errored (of {len(results)} tiles)")
    return 1 if n_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
