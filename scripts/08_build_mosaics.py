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

``periods``/``overwrite``/``workers`` are read from ``config.mosaic`` (see
:class:`landscape_change_detection_pipeline.config.MosaicConfig`), not CLI
flags -- a run must be reproducible from ``config.yaml`` alone. Only
``--config``/``--env-file`` (which config file to read) stay as flags.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd

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
    return parser.parse_args(argv)


def _build_one_period(
    period: str,
    composites_root: str,
    output_root: str,
    registry: pd.DataFrame,
    overwrite: bool,
) -> tuple[str, Optional[dict]]:
    """One period's full load-composites -> mosaic -> write pipeline, run in
    a worker process. Returns ``(period, result_summary_or_None)`` --
    ``None`` when skipped (already written, or no tile composite found for
    this period), so the caller does all printing/counting."""
    out_path = mosaic_output_path(output_root, period)
    if out_path.is_file() and not overwrite:
        return period, None
    tiles = load_period_tile_composites(composites_root, registry, period)
    if not tiles:
        return period, None
    result = build_period_mosaic(tiles, registry)
    write_mosaic(out_path, result)
    return period, {"n_tiles": len(result["tiles_present"]), "resolution_m": result["resolution_m"]}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv if argv is not None else sys.argv[1:]))
    config = load_config(args.config, args.env_file)
    mosaic_cfg = config.mosaic
    comp_cfg = config.composites

    registry = read_registry(config.tiles.registry_path)
    tile_ids = registry["tile_id"].tolist()

    periods = mosaic_cfg.periods or discover_periods(comp_cfg.output_root, tile_ids)

    common_args = (comp_cfg.output_root, mosaic_cfg.output_root, registry, mosaic_cfg.overwrite)

    def _report(period: str, summary: Optional[dict]) -> bool:
        if summary is None:
            return False
        print(
            f"[mosaic] {period}: wrote mosaic from {summary['n_tiles']} tiles "
            f"at {summary['resolution_m']}m"
        )
        return True

    written = 0
    if mosaic_cfg.workers <= 1:
        for period in periods:
            period, summary = _build_one_period(period, *common_args)
            written += int(_report(period, summary))
    else:
        with ProcessPoolExecutor(max_workers=mosaic_cfg.workers) as pool:
            futures = [pool.submit(_build_one_period, period, *common_args) for period in periods]
            for future in as_completed(futures):
                period, summary = future.result()
                written += int(_report(period, summary))

    print(f"[mosaic] {written} mosaics written across {len(periods)} periods")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
