#!/usr/bin/env python3
"""BFAST-Monitor fire cross-check -- per-tile, per-pixel NBR break detection.

Purpose
-------
For every tile, reuses the same shared-grid time series
:mod:`landscape_change_detection_pipeline.change.ccdc` builds, computes NBR per pixel,
and runs BFAST-Monitor (a pure-Python reimplementation) to test whether the
monitoring period (``bfast.monitor_start`` onward) shows a statistically
significant break relative to the stable history period before it. Fire is
this cross-check's specific target (BFAST's own documented strength, ~96%
vs ~73% for CCDC/LandTrendr-family methods on fire per a 2025 multi-algorithm
comparison). Output: ``outputs/bfast/<tile_id>/bfast.npz``.

Usage
-----
    python scripts/12_run_bfast.py --config configs/config.yaml
    python scripts/12_run_bfast.py --tiles tile_0000_0000 --overwrite
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landscape_change_detection_pipeline.classes.class_config import load_class_config  # noqa: E402
from landscape_change_detection_pipeline.config import load_config  # noqa: E402
from landscape_change_detection_pipeline.change.bfast import build_tile_bfast, read_bfast_result  # noqa: E402
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
    bfast_cfg = config.bfast

    monitor_start_ordinal = date.fromisoformat(bfast_cfg.monitor_start).toordinal()

    registry = read_registry(config.tiles.registry_path)
    tile_ids = args.tiles.split(",") if args.tiles else registry["tile_id"].tolist()

    total_breaks = 0
    for tile_id in tile_ids:
        out_path = build_tile_bfast(
            tile_dir=config.dem.tile_dir,
            inference_root=config.inference.output_root,
            output_root=bfast_cfg.output_root,
            tile_id=tile_id,
            class_config=class_config,
            monitor_start_ordinal=monitor_start_ordinal,
            order=bfast_cfg.order,
            alpha=bfast_cfg.alpha,
            min_history_obs=bfast_cfg.min_history_obs,
            min_monitoring_obs=bfast_cfg.min_monitoring_obs,
            n_workers=bfast_cfg.n_workers,
            overwrite=args.overwrite,
        )
        if out_path is None:
            print(f"[bfast] {tile_id}: already built, skipping (use --overwrite to rerun)")
            continue
        result = read_bfast_result(out_path)
        n_tested = len(result["row"])
        n_breaks = int(result["has_break"].sum())
        print(f"[bfast] {tile_id}: {n_tested} pixels tested, {n_breaks} breaks detected -> {out_path}")
        total_breaks += n_breaks

    print(f"[bfast] {total_breaks} total breaks detected across {len(tile_ids)} tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
