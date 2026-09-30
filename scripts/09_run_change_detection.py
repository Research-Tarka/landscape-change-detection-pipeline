#!/usr/bin/env python3
"""Per-pixel temporal segmentation -- break dates, trends and seasonality.

Purpose
-------
For every tile, builds each pixel's full time series (from
``change_detection.start_year`` on, every stored scene with a classification)
of the configured spectral indices, and segments it into stable stretches
separated by detected breaks (see
:mod:`landscape_change_detection_pipeline.change.segmentation` for the
method). Each segment records its start/end/break dates, the size and
direction of the break per feature, the fitted level/trend/seasonal
amplitude/peak day per feature, and the dominant land-cover class before and
at the break. Output: ``outputs/change_detection/<tile_id>/segments.npz``, a
flattened segment table (one row per detected segment across every pixel).

Usage
-----
    python scripts/09_run_change_detection.py --config configs/config.yaml

``tiles``/``overwrite`` and every detector parameter are read from
``config.change_detection`` (see
:class:`landscape_change_detection_pipeline.config.ChangeDetectionConfig`) --
not CLI flags, so a run is reproducible from ``config.yaml`` alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landscape_change_detection_pipeline.classes.class_config import load_class_config  # noqa: E402
from landscape_change_detection_pipeline.config import load_config  # noqa: E402
from landscape_change_detection_pipeline.change.change_detection import run_tile, read_segments  # noqa: E402
from landscape_change_detection_pipeline.tiles.registry import read_registry  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument("--classes", default=None, help="Path to classes.yaml")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv if argv is not None else sys.argv[1:]))
    config = load_config(args.config, args.env_file)
    class_config = load_class_config(args.classes)
    cfg = config.change_detection

    registry = read_registry(config.tiles.registry_path)
    tile_ids = cfg.tiles or registry["tile_id"].tolist()

    total_segments = 0
    for tile_id in tile_ids:
        out_path = run_tile(
            tile_dir=config.dem.tile_dir,
            inference_root=config.inference.output_root,
            output_root=cfg.output_root,
            tile_id=tile_id,
            class_config=class_config,
            cfg=cfg,
            overwrite=cfg.overwrite,
        )
        if out_path is None:
            print(f"[change_detection] {tile_id}: already built, skipping (set change_detection.overwrite: true to rerun)")
            continue
        result = read_segments(out_path)
        n_segments = len(result["row"])
        n_breaks = int((result["t_break"] > 0).sum())
        n_truncated = int(result["truncated"].sum())
        print(
            f"[change_detection] {tile_id}: {n_segments} segments across "
            f"{int((result['n_segments'] > 0).sum())} pixels, {n_breaks} breaks detected"
            + (f", {n_truncated} pixels truncated at max_segments" if n_truncated else "")
            + f" -> {out_path}"
        )
        total_segments += n_segments

    print(f"[change_detection] {total_segments} total segments written across {len(tile_ids)} tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
