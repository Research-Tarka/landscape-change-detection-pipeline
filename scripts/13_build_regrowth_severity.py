#!/usr/bin/env python3
"""NDVI regrowth trajectory and dNBR burn severity.

Purpose
-------
For every tile, reads a break-detection result (CCDC or BFAST, per
``regrowth_severity.break_source``), and for every pixel with a detected
break: dNBR between the monthly composite immediately before and after that
break, and the NDVI trajectory across every monthly composite from the
break onward. See
:mod:`landscape_change_detection_pipeline.change.regrowth_severity` for the full
design.
Output: ``outputs/regrowth_severity/<tile_id>/regrowth_severity_<break_source>.npz``.

Usage
-----
    python scripts/13_build_regrowth_severity.py --config configs/config.yaml
    python scripts/13_build_regrowth_severity.py --tiles tile_0000_0000 --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landscape_change_detection_pipeline.config import load_config  # noqa: E402
from landscape_change_detection_pipeline.change.ccdc import ccdc_output_path, read_ccdc_result  # noqa: E402
from landscape_change_detection_pipeline.change.bfast import bfast_output_path, read_bfast_result  # noqa: E402
from landscape_change_detection_pipeline.change.regrowth_severity import (  # noqa: E402
    build_tile_regrowth_severity,
    read_regrowth_severity_result,
)
from landscape_change_detection_pipeline.change.spectral_composites import discover_periods_for_tile  # noqa: E402
from landscape_change_detection_pipeline.tiles.registry import read_registry  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument("--tiles", default=None, help="Comma-separated tile_ids to restrict to")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv if argv is not None else sys.argv[1:]))
    config = load_config(args.config, args.env_file)
    rs_cfg = config.regrowth_severity

    registry = read_registry(config.tiles.registry_path)
    tile_ids = args.tiles.split(",") if args.tiles else registry["tile_id"].tolist()

    total_dnbr = 0
    for tile_id in tile_ids:
        if rs_cfg.break_source == "ccdc":
            break_path = ccdc_output_path(config.ccdc.output_root, tile_id)
            reader = read_ccdc_result
        else:
            break_path = bfast_output_path(config.bfast.output_root, tile_id)
            reader = read_bfast_result

        if not break_path.is_file():
            print(f"[regrowth_severity] {tile_id}: no {rs_cfg.break_source} result at '{break_path}', skipping")
            continue

        break_result = reader(break_path)
        all_months = discover_periods_for_tile(config.index_composites.output_root, tile_id)
        if not all_months:
            print(f"[regrowth_severity] {tile_id}: no index composites found, skipping")
            continue

        out_path = build_tile_regrowth_severity(
            index_composites_root=config.index_composites.output_root,
            output_root=rs_cfg.output_root,
            tile_id=tile_id,
            break_result=break_result,
            break_source=rs_cfg.break_source,
            transform=break_result["transform"],
            crs_wkt=break_result["crs_wkt"],
            resolution_m=break_result["resolution_m"],
            dst_shape=break_result["shape"],
            all_months=all_months,
            overwrite=args.overwrite,
        )
        if out_path is None:
            print(f"[regrowth_severity] {tile_id}: already built, skipping (use --overwrite to rerun)")
            continue

        result = read_regrowth_severity_result(out_path)
        n_dnbr = int(result["has_dnbr"].sum())
        n_trajectory_points = len(result["trajectory_row"])
        print(
            f"[regrowth_severity] {tile_id}: {n_dnbr} pixels with dNBR, "
            f"{n_trajectory_points} NDVI trajectory points -> {out_path}"
        )
        total_dnbr += n_dnbr

    print(f"[regrowth_severity] {total_dnbr} total dNBR pixels across {len(tile_ids)} tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
