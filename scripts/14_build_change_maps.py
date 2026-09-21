#!/usr/bin/env python3
"""Combined annual / month-to-month change maps.

Purpose
-------
For every tile, builds both an annual change map (same calendar month
across consecutive years) and a month-to-month change map (consecutive
stored months, any season), each with three classification-change
confidence bands (raw, persistent, corroborated by CCDC/BFAST) plus dNBR
where available -- see
:mod:`landscape_change_detection_pipeline.change.change_maps` for the full design
and why these bands are kept separate rather than collapsed into one
opaque "changed" bit. Output:
``outputs/change_maps/<tile_id>/<from_month>_vs_<to_month>/change_map.npz``.

Usage
-----
    python scripts/14_build_change_maps.py --config configs/config.yaml
    python scripts/14_build_change_maps.py --tiles tile_0000_0000 --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from affine import Affine  # noqa: E402

from landscape_change_detection_pipeline.classes.class_config import load_class_config  # noqa: E402
from landscape_change_detection_pipeline.config import load_config  # noqa: E402
from landscape_change_detection_pipeline.change.ccdc import ccdc_output_path, read_ccdc_result  # noqa: E402
from landscape_change_detection_pipeline.change.bfast import bfast_output_path, read_bfast_result  # noqa: E402
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
    parser.add_argument("--tiles", default=None, help="Comma-separated tile_ids to restrict to")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _load_break_results(config, tile_id: str):
    """Both CCDC and BFAST results for this tile, whichever exist -- either
    may be ``None`` (see change_maps.compute_corroborated_change's own
    tolerance for a missing detector)."""
    ccdc_result = None
    ccdc_path = ccdc_output_path(config.ccdc.output_root, tile_id)
    if ccdc_path.is_file():
        ccdc_result = read_ccdc_result(ccdc_path)

    bfast_result = None
    bfast_path = bfast_output_path(config.bfast.output_root, tile_id)
    if bfast_path.is_file():
        bfast_result = read_bfast_result(bfast_path)

    return ccdc_result, bfast_result


def _dnbr_grid(config, tile_id: str, dst_shape):
    """dNBR per-pixel array from the regrowth_severity output, if built for
    this tile; ``None`` otherwise (build_change_map fills in NaN when this
    is missing)."""
    for break_source in ("ccdc", "bfast"):
        path = regrowth_severity_output_path(config.regrowth_severity.output_root, tile_id, break_source)
        if path.is_file():
            result = read_regrowth_severity_result(path)
            if result["dnbr"].shape == tuple(dst_shape):
                return result["dnbr"]
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv if argv is not None else sys.argv[1:]))
    config = load_config(args.config, args.env_file)
    class_config = load_class_config(args.classes)
    cm_cfg = config.change_maps

    registry = read_registry(config.tiles.registry_path)
    tile_ids = args.tiles.split(",") if args.tiles else registry["tile_id"].tolist()

    total_written = 0
    for tile_id in tile_ids:
        all_months = discover_periods_for_tile(config.composites.output_root, tile_id, filename="composite.npz")
        if not all_months:
            print(f"[change_maps] {tile_id}: no monthly composites found, skipping")
            continue

        ccdc_result, bfast_result = _load_break_results(config, tile_id)
        grid_source = ccdc_result or bfast_result
        if grid_source is None:
            print(f"[change_maps] {tile_id}: no CCDC or BFAST result found, skipping (corroboration needs a grid)")
            continue
        dst_transform = Affine(*grid_source["transform"])
        dst_crs_wkt = grid_source["crs_wkt"]
        dst_shape = grid_source["shape"]
        resolution_m = grid_source["resolution_m"]
        dnbr = _dnbr_grid(config, tile_id, dst_shape)

        pairs = annual_month_pairs(all_months) + month_to_month_pairs(all_months)
        sorted_months = sorted(all_months)

        for from_month, to_month in pairs:
            comparison_key = f"{from_month}_vs_{to_month}"
            out_path = change_map_output_path(cm_cfg.output_root, tile_id, comparison_key)
            if out_path.is_file() and not args.overwrite:
                continue

            to_index = sorted_months.index(to_month) if to_month in sorted_months else -1
            next_months = sorted_months[to_index + 1 :] if to_index >= 0 else []

            result = build_change_map(
                composites_root=config.composites.output_root,
                tile_id=tile_id,
                from_month=from_month,
                to_month=to_month,
                next_months=next_months,
                dst_transform=dst_transform,
                dst_crs_wkt=dst_crs_wkt,
                dst_shape=dst_shape,
                class_config=class_config,
                resolution_m=resolution_m,
                ccdc_result=ccdc_result,
                bfast_result=bfast_result,
                dnbr=dnbr,
                persistence_periods=cm_cfg.persistence_periods,
                corroboration_window_days=cm_cfg.corroboration_window_days,
            )
            if result is None:
                continue
            write_change_map(out_path, result)
            n_raw = int(result["changed_raw"].sum())
            n_persistent = int(result["changed_persistent"].sum())
            n_corroborated = int(result["changed_corroborated"].sum())
            print(
                f"[change_maps] {tile_id} {comparison_key}: raw={n_raw} persistent={n_persistent} "
                f"corroborated={n_corroborated} -> {out_path}"
            )
            total_written += 1

    print(f"[change_maps] {total_written} change maps written across {len(tile_ids)} tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
