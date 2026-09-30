#!/usr/bin/env python3
"""NDVI regrowth trajectory and dNBR burn severity.

Purpose
-------
For every tile, reads its per-pixel break dates from
``change_detection.output_root`` (Stage 09's ``segments.npz``), and for
every pixel with a detected break: dNBR between the monthly NBR composite
immediately before and after that break, and the NDVI trajectory across
every month from the break onward. Every month's NBR/NDVI composite is
computed directly from this tile's stored scenes on demand (never a
separate persisted index-composite store -- see
:func:`landscape_change_detection_pipeline.change.regrowth_severity.build_month_composite_cache`).
See :mod:`landscape_change_detection_pipeline.change.regrowth_severity` for
the full design.
Output: ``outputs/regrowth_severity/<tile_id>/regrowth_severity.npz``.

Usage
-----
    python scripts/11_build_regrowth_severity.py --config configs/config.yaml

``tiles``/``overwrite``/``workers`` are read from ``config.regrowth_severity``
(see :class:`landscape_change_detection_pipeline.config.RegrowthSeverityConfig`)
-- not CLI flags, so a run is reproducible from ``config.yaml`` alone.
``workers>1`` builds separate tiles in parallel worker processes -- tiles
are fully independent here, and unlike Stage 09 this stage does no internal
multi-core work of its own, so it benefits from inter-tile parallelism the
way Stage 08's mosaicking does.
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
from landscape_change_detection_pipeline.change.change_detection import (  # noqa: E402
    read_segments,
    segments_output_path,
)
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
    parser.add_argument("--classes", default=None, help="Path to classes.yaml")
    return parser.parse_args(argv)


def _build_one_tile(
    tile_id: str,
    tile_dir: str,
    inference_root: str,
    change_detection_output_root: str,
    composites_root: str,
    output_root: str,
    class_config,
    overwrite: bool,
    month_threads: int = 1,
) -> tuple[str, Optional[str], Optional[dict]]:
    """One tile's full build, run in a worker process. Returns ``(tile_id,
    skip_reason_or_None, summary_or_None)`` -- exactly one of the last two
    is non-``None``, so the caller does all printing/counting."""
    segments_path = segments_output_path(change_detection_output_root, tile_id)
    if not segments_path.is_file():
        return tile_id, f"no segments at '{segments_path}', skipping (run scripts/09_run_change_detection.py first)", None

    segments_result = read_segments(segments_path)
    all_months = discover_periods_for_tile(composites_root, tile_id, filename="composite.npz")
    if not all_months:
        return tile_id, "no monthly composites found, skipping (run scripts/07_build_monthly_composites.py first)", None

    print(f"[regrowth_severity] {tile_id}: starting ({len(all_months)} months available)", flush=True)
    out_path = build_tile_regrowth_severity(
        tile_dir=tile_dir,
        inference_root=inference_root,
        output_root=output_root,
        tile_id=tile_id,
        class_config=class_config,
        segments_result=segments_result,
        transform=segments_result["transform"],
        crs_wkt=segments_result["crs_wkt"],
        resolution_m=segments_result["resolution_m"],
        dst_shape=segments_result["shape"],
        all_months=all_months,
        overwrite=overwrite,
        month_threads=month_threads,
    )
    if out_path is None:
        return tile_id, "already built, skipping (set regrowth_severity.overwrite: true to rerun)", None

    result = read_regrowth_severity_result(out_path)
    return tile_id, None, {
        "n_dnbr": int(result["has_dnbr"].sum()),
        "n_trajectory_points": len(result["trajectory_row"]),
        "out_path": out_path,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv if argv is not None else sys.argv[1:]))
    config = load_config(args.config, args.env_file)
    class_config = load_class_config(args.classes)
    rs_cfg = config.regrowth_severity

    registry = read_registry(config.tiles.registry_path)
    tile_ids = rs_cfg.tiles or registry["tile_id"].tolist()

    common_args = (
        config.dem.tile_dir, config.inference.output_root, config.change_detection.output_root,
        config.composites.output_root, rs_cfg.output_root, class_config, rs_cfg.overwrite,
        rs_cfg.month_threads,
    )

    def _report(tile_id: str, skip_reason: Optional[str], summary: Optional[dict]) -> int:
        if skip_reason is not None:
            print(f"[regrowth_severity] {tile_id}: {skip_reason}")
            return 0
        print(
            f"[regrowth_severity] {tile_id}: {summary['n_dnbr']} pixels with dNBR, "
            f"{summary['n_trajectory_points']} NDVI trajectory points -> {summary['out_path']}"
        )
        return summary["n_dnbr"]

    total_dnbr = 0
    if rs_cfg.workers <= 1:
        for tile_id in tile_ids:
            tile_id, skip_reason, summary = _build_one_tile(tile_id, *common_args)
            total_dnbr += _report(tile_id, skip_reason, summary)
    else:
        with ProcessPoolExecutor(max_workers=rs_cfg.workers) as pool:
            futures = [pool.submit(_build_one_tile, tile_id, *common_args) for tile_id in tile_ids]
            for future in as_completed(futures):
                tile_id, skip_reason, summary = future.result()
                total_dnbr += _report(tile_id, skip_reason, summary)

    print(f"[regrowth_severity] {total_dnbr} total dNBR pixels across {len(tile_ids)} tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
