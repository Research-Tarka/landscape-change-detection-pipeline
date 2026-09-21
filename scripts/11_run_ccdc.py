#!/usr/bin/env python3
"""Local CCDC/COLD change detection -- per-tile, per-pixel break dates.

Purpose
-------
For every tile, builds one shared-grid, multi-decade per-pixel time series
from this pipeline's own stored reflectance and classification (never Earth
Engine's raw collections), then fits COLD
(via ``pyxccd``) independently per pixel to detect breakpoints: date,
duration, and per-band spectral magnitude of each disturbance. Output:
``outputs/ccdc/<tile_id>/ccdc.npz``, a flattened segment table (one row per
detected segment across every pixel).

Usage
-----
    python scripts/11_run_ccdc.py --config configs/config.yaml
    python scripts/11_run_ccdc.py --tiles tile_0000_0000 --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landscape_change_detection_pipeline.classes.class_config import load_class_config  # noqa: E402
from landscape_change_detection_pipeline.config import load_config  # noqa: E402
from landscape_change_detection_pipeline.change.ccdc import build_tile_ccdc, read_ccdc_result  # noqa: E402
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
    ccdc_cfg = config.ccdc

    registry = read_registry(config.tiles.registry_path)
    tile_ids = args.tiles.split(",") if args.tiles else registry["tile_id"].tolist()

    total_segments = 0
    for tile_id in tile_ids:
        out_path = build_tile_ccdc(
            tile_dir=config.dem.tile_dir,
            inference_root=config.inference.output_root,
            output_root=ccdc_cfg.output_root,
            tile_id=tile_id,
            class_config=class_config,
            lam=ccdc_cfg.lam,
            p_cg=ccdc_cfg.p_cg,
            conse=ccdc_cfg.conse,
            min_clear_obs=ccdc_cfg.min_clear_obs,
            n_workers=ccdc_cfg.n_workers,
            overwrite=args.overwrite,
        )
        if out_path is None:
            print(f"[ccdc] {tile_id}: already built, skipping (use --overwrite to rerun)")
            continue
        result = read_ccdc_result(out_path)
        n_segments = len(result["row"])
        n_breaks = int((result["t_break"] > 0).sum())
        print(f"[ccdc] {tile_id}: {n_segments} segments across {len(set(zip(result['row'].tolist(), result['col'].tolist())))} pixels, {n_breaks} breaks detected -> {out_path}")
        total_segments += n_segments

    print(f"[ccdc] {total_segments} total segments written across {len(tile_ids)} tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
