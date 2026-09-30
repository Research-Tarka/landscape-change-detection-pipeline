#!/usr/bin/env python3
"""Combined annual / month-to-month change maps.

Purpose
-------
For every tile, builds both an annual change map (same calendar month
across consecutive years) and a month-to-month change map (consecutive
stored months, any season), each with three classification-change
confidence bands (raw, persistent, corroborated by the per-pixel
segmentation's break dates) plus dNBR where available -- see
:mod:`landscape_change_detection_pipeline.change.change_maps` for the full design
and why these bands are kept separate rather than collapsed into one
opaque "changed" bit. Output:
``outputs/change_maps/<tile_id>/<from_month>_vs_<to_month>/change_map.npz``.

Usage
-----
    python scripts/13_build_change_maps.py --config configs/config.yaml

``tiles``/``overwrite``/``workers`` are read from ``config.change_maps``
(see :class:`landscape_change_detection_pipeline.config.ChangeMapsConfig`)
-- not CLI flags, so a run is reproducible from ``config.yaml`` alone.
``workers>1`` builds separate tiles in parallel worker processes (every
comparison pair for one tile still runs sequentially within its own worker)
-- tiles are fully independent here, and unlike Stage 09 this stage does no
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

from affine import Affine  # noqa: E402

from landscape_change_detection_pipeline.classes.class_config import load_class_config  # noqa: E402
from landscape_change_detection_pipeline.config import load_config  # noqa: E402
from landscape_change_detection_pipeline.change.change_detection import (  # noqa: E402
    read_segments,
    segments_output_path,
)
from landscape_change_detection_pipeline.change.change_maps import (  # noqa: E402
    annual_month_pairs,
    build_change_map,
    change_map_output_path,
    month_to_month_pairs,
    write_change_map,
)
from landscape_change_detection_pipeline.change.regrowth_severity import (  # noqa: E402
    read_regrowth_severity_result,
    regrowth_severity_output_path,
)
from landscape_change_detection_pipeline.change.spectral_composites import discover_periods_for_tile  # noqa: E402
from landscape_change_detection_pipeline.tiles.registry import read_registry  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument("--classes", default=None, help="Path to classes.yaml")
    return parser.parse_args(argv)


def _dnbr_grid(regrowth_severity_output_root: str, tile_id: str, dst_shape):
    """dNBR per-pixel array from the regrowth_severity output, if built for
    this tile; ``None`` otherwise (build_change_map fills in NaN when this
    is missing)."""
    path = regrowth_severity_output_path(regrowth_severity_output_root, tile_id)
    if path.is_file():
        result = read_regrowth_severity_result(path)
        if result["dnbr"].shape == tuple(dst_shape):
            return result["dnbr"]
    return None


def _build_one_tile(
    tile_id: str,
    composites_root: str,
    change_detection_output_root: str,
    regrowth_severity_output_root: str,
    output_root: str,
    class_config,
    persistence_periods: int,
    corroboration_window_days: int,
    overwrite: bool,
) -> tuple[str, Optional[str], list[dict]]:
    """Every comparison pair's change map for one tile, run in a worker
    process. Returns ``(tile_id, skip_reason_or_None, summaries)`` --
    ``summaries`` is a list of per-comparison dicts (empty list if skipped
    or nothing new to write, so the caller does all printing/counting)."""
    all_months = discover_periods_for_tile(composites_root, tile_id, filename="composite.npz")
    if not all_months:
        return tile_id, "no monthly composites found, skipping", []

    segments_path = segments_output_path(change_detection_output_root, tile_id)
    if not segments_path.is_file():
        return tile_id, (
            f"no segments at '{segments_path}', skipping "
            "(corroboration needs a grid -- run scripts/09_run_change_detection.py first)"
        ), []
    segments_result = read_segments(segments_path)
    dst_transform = Affine(*segments_result["transform"])
    dst_crs_wkt = segments_result["crs_wkt"]
    dst_shape = segments_result["shape"]
    resolution_m = segments_result["resolution_m"]
    dnbr = _dnbr_grid(regrowth_severity_output_root, tile_id, dst_shape)

    pairs = annual_month_pairs(all_months) + month_to_month_pairs(all_months)
    sorted_months = sorted(all_months)

    print(f"[change_maps] {tile_id}: start, {len(pairs)} comparisons", flush=True)
    summaries: list[dict] = []
    for i, (from_month, to_month) in enumerate(pairs, start=1):
        comparison_key = f"{from_month}_vs_{to_month}"
        out_path = change_map_output_path(output_root, tile_id, comparison_key)
        if out_path.is_file() and not overwrite:
            print(f"[change_maps] {tile_id} [{i}/{len(pairs)}] {comparison_key}: exists, skipped", flush=True)
            continue
        print(f"[change_maps] {tile_id} [{i}/{len(pairs)}] {comparison_key}: building...", flush=True)

        to_index = sorted_months.index(to_month) if to_month in sorted_months else -1
        next_months = sorted_months[to_index + 1 :] if to_index >= 0 else []

        result = build_change_map(
            composites_root=composites_root,
            tile_id=tile_id,
            from_month=from_month,
            to_month=to_month,
            next_months=next_months,
            dst_transform=dst_transform,
            dst_crs_wkt=dst_crs_wkt,
            dst_shape=dst_shape,
            class_config=class_config,
            resolution_m=resolution_m,
            segments_result=segments_result,
            dnbr=dnbr,
            persistence_periods=persistence_periods,
            corroboration_window_days=corroboration_window_days,
        )
        if result is None:
            print(f"[change_maps] {tile_id} {comparison_key}: no result, skipped", flush=True)
            continue
        write_change_map(out_path, result)
        summaries.append({
            "comparison_key": comparison_key,
            "n_raw": int(result["changed_raw"].sum()),
            "n_persistent": int(result["changed_persistent"].sum()),
            "n_corroborated": int(result["changed_corroborated"].sum()),
            "out_path": out_path,
        })
        print(
            f"[change_maps] {tile_id} {comparison_key}: raw={summaries[-1]['n_raw']} "
            f"persistent={summaries[-1]['n_persistent']} corroborated={summaries[-1]['n_corroborated']} "
            f"-> {out_path}",
            flush=True,
        )

    return tile_id, None, summaries


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv if argv is not None else sys.argv[1:]))
    config = load_config(args.config, args.env_file)
    class_config = load_class_config(args.classes)
    cm_cfg = config.change_maps

    registry = read_registry(config.tiles.registry_path)
    tile_ids = cm_cfg.tiles or registry["tile_id"].tolist()

    common_args = (
        config.composites.output_root, config.change_detection.output_root,
        config.regrowth_severity.output_root, cm_cfg.output_root, class_config,
        cm_cfg.persistence_periods, cm_cfg.corroboration_window_days, cm_cfg.overwrite,
    )

    def _report(tile_id: str, skip_reason: Optional[str], summaries: list[dict]) -> int:
        if skip_reason is not None:
            print(f"[change_maps] {tile_id}: {skip_reason}")
            return 0
        print(f"[change_maps] {tile_id}: done, {len(summaries)} maps written", flush=True)
        return len(summaries)

    total_written = 0
    if cm_cfg.workers <= 1:
        for tile_id in tile_ids:
            tile_id, skip_reason, summaries = _build_one_tile(tile_id, *common_args)
            total_written += _report(tile_id, skip_reason, summaries)
    else:
        with ProcessPoolExecutor(max_workers=cm_cfg.workers) as pool:
            futures = [pool.submit(_build_one_tile, tile_id, *common_args) for tile_id in tile_ids]
            for future in as_completed(futures):
                tile_id, skip_reason, summaries = future.result()
                total_written += _report(tile_id, skip_reason, summaries)

    print(f"[change_maps] {total_written} change maps written across {len(tile_ids)} tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
