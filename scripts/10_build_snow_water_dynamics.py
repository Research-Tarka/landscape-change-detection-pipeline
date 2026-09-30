#!/usr/bin/env python3
"""Per-pixel, per-water-year snow, ice and open-water summaries.

Purpose
-------
For every tile, reads Stage 7's monthly class composites and summarises,
per pixel per water year (default: October-September): how many observed
months were snow-covered/ice-covered/open-water, and the water-year-
relative month index of the first and last observed snow month. This is a
categorical, classifier-driven signal -- deliberately distinct from
:mod:`landscape_change_detection_pipeline.change.change_detection`'s
continuous vegetation-index trend model, which excludes these same classes
because mixing snow-onset/melt into a NBR/NDVI harmonic fit would read every
seasonal snow event as a spurious break. See
:mod:`landscape_change_detection_pipeline.change.snow_water_dynamics` for
the full design. Output:
``outputs/snow_water_dynamics/<tile_id>/snow_water_dynamics.npz``.

Usage
-----
    python scripts/10_build_snow_water_dynamics.py --config configs/config.yaml

``tiles``/``overwrite``/``water_year_start_month``/``workers`` are read from
``config.snow_water_dynamics`` (see
:class:`landscape_change_detection_pipeline.config.SnowWaterDynamicsConfig`)
-- not CLI flags, so a run is reproducible from ``config.yaml`` alone.
``workers>1`` builds separate tiles' summaries in parallel worker processes
-- tiles are fully independent here (each reads only its own composites and
writes its own output subtree), and unlike Stage 09 this stage does no
internal multi-core work of its own, so it benefits from inter-tile
parallelism the way Stage 08's mosaicking does.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landscape_change_detection_pipeline.classes.class_config import load_class_config  # noqa: E402
from landscape_change_detection_pipeline.config import load_config  # noqa: E402
from landscape_change_detection_pipeline.change.snow_water_dynamics import (  # noqa: E402
    build_tile_snow_water_dynamics,
    read_snow_water_dynamics,
)
from landscape_change_detection_pipeline.tiles.registry import read_registry  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument("--classes", default=None, help="Path to classes.yaml")
    return parser.parse_args(argv)


def _build_one_tile(
    tile_id: str,
    composites_root: str,
    output_root: str,
    class_config,
    water_year_start_month: int,
    overwrite: bool,
) -> tuple[str, Optional[dict]]:
    """One tile's full build, run in a worker process. Returns ``(tile_id,
    summary_or_None)`` -- ``None`` when skipped (already built, or no
    composites found), so the caller does all printing/counting."""
    out_path = build_tile_snow_water_dynamics(
        composites_root=composites_root,
        output_root=output_root,
        tile_id=tile_id,
        class_config=class_config,
        water_year_start_month=water_year_start_month,
        overwrite=overwrite,
    )
    if out_path is None:
        return tile_id, None
    result = read_snow_water_dynamics(out_path)
    return tile_id, {
        "n_years": len(result["water_years"]),
        "year_min": int(result["water_years"].min()),
        "year_max": int(result["water_years"].max()),
        "mean_snow_months": float(result["n_months_snow"].mean()),
        "out_path": out_path,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv if argv is not None else sys.argv[1:]))
    config = load_config(args.config, args.env_file)
    class_config = load_class_config(args.classes)
    swd_cfg = config.snow_water_dynamics

    registry = read_registry(config.tiles.registry_path)
    tile_ids = swd_cfg.tiles or registry["tile_id"].tolist()

    common_args = (
        config.composites.output_root, swd_cfg.output_root, class_config,
        swd_cfg.water_year_start_month, swd_cfg.overwrite,
    )

    def _report(tile_id: str, summary: Optional[dict]) -> int:
        if summary is None:
            print(f"[snow_water_dynamics] {tile_id}: already built or no monthly composites found, skipping "
                  "(set snow_water_dynamics.overwrite: true to rerun, or run "
                  "scripts/07_build_monthly_composites.py first)")
            return 0
        print(
            f"[snow_water_dynamics] {tile_id}: {summary['n_years']} water years "
            f"({summary['year_min']}-{summary['year_max']}), "
            f"mean {summary['mean_snow_months']:.1f} snow months/pixel/year -> {summary['out_path']}"
        )
        return summary["n_years"]

    total_years = 0
    if swd_cfg.workers <= 1:
        for tile_id in tile_ids:
            tile_id, summary = _build_one_tile(tile_id, *common_args)
            total_years += _report(tile_id, summary)
    else:
        with ProcessPoolExecutor(max_workers=swd_cfg.workers) as pool:
            futures = [pool.submit(_build_one_tile, tile_id, *common_args) for tile_id in tile_ids]
            for future in as_completed(futures):
                tile_id, summary = future.result()
                total_years += _report(tile_id, summary)

    print(f"[snow_water_dynamics] {total_years} tile-years written across {len(tile_ids)} tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
