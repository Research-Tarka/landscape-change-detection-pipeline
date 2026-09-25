"""Stage 3 -- orchestrate scene download across splits and tiles.

Usage
-----
Run one split sequentially::

    python scripts/03_download_scenes.py --config configs/config.yaml --split 0

Run every configured split in parallel (one subprocess per split, each
against its own GEE project)::

    python scripts/03_download_scenes.py --config configs/config.yaml --split all --parallel

Within one split, process tiles with up to 4 concurrent worker processes::

    python scripts/03_download_scenes.py --config configs/config.yaml --split 0 --tile-workers 4

Quick smoke test (a handful of recent-year scenes per sensor, not the full
historical archive)::

    python scripts/03_download_scenes.py --config configs/config.yaml --split all --parallel --first-year 2022

See ``landscape_change_detection_pipeline.scenes.orchestration`` for why both
parallelism layers use real OS-level isolation (subprocesses /
multiprocessing.Process), never threads.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landscape_change_detection_pipeline.config import load_config
from landscape_change_detection_pipeline.scenes.gee_auth import initialize_ee
from landscape_change_detection_pipeline.scenes.orchestration import (
    resolve_splits,
    run_splits_parallel,
    run_tiles_multi_process,
    tiles_for_split,
)
from landscape_change_detection_pipeline.tiles.registry import read_registry
from landscape_change_detection_pipeline.tiles.splitter import read_assignment


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument(
        "--split",
        default=None,
        help=(
            "Split index to run ('all' to run every configured split). "
            "Defaults to config.yaml's scene_download.run_all_splits (all "
            "splits in parallel if true, else split 0)."
        ),
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=None,
        help=(
            "With --split all: run every split concurrently, one subprocess each. "
            "Defaults to config.yaml's scene_download.run_all_splits."
        ),
    )
    parser.add_argument(
        "--tile-workers",
        type=int,
        default=None,
        help="Concurrent tile workers within this split (overrides config)",
    )
    parser.add_argument(
        "--ee-project",
        default=None,
        help="Explicit GEE project id (overrides this split's configured project)",
    )
    parser.add_argument(
        "--sensors",
        default=None,
        help="Comma-separated sensor keys to restrict to (e.g. L8,L9,S2)",
    )
    parser.add_argument("--until-year", type=int, default=None, help="Last year to search (default: current year)")
    parser.add_argument(
        "--first-year",
        type=int,
        default=1984,
        help="First year to search (default: 1984). Narrow this for a quick smoke test.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    args = parse_args(argv)

    config = load_config(args.config, args.env_file)
    sd = config.scene_download
    n_splits = config.tiles.n_splits

    # sd.ee_projects is the authoritative per-split project assignment
    # (index-aligned with tiles.n_splits). When left unset in config.yaml,
    # fall back to round-robining over the shared gee.projects pool (e.g.
    # this pipeline's full set of Earth Engine Cloud projects), so a single
    # place (gee.projects) can drive every split without repeating the list.
    ee_projects = list(sd.ee_projects)
    if not ee_projects and config.gee.projects:
        pool = config.gee.projects
        ee_projects = [pool[i % len(pool)] for i in range(n_splits)]

    # CLI flags win when given explicitly; otherwise scene_download.run_all_splits
    # (config.yaml) decides between "all splits in parallel" and "split 0 only",
    # so a bare invocation with no CLI args is fully config-driven.
    if args.split is None:
        args.split = "all" if sd.run_all_splits else "0"
    if args.parallel is None:
        args.parallel = sd.run_all_splits

    if args.split == "all" and args.parallel:
        splits = resolve_splits(args.split, n_splits)
        return run_splits_parallel(argv, splits)

    splits = resolve_splits(args.split, n_splits)

    registry = read_registry(config.tiles.registry_path)
    assignment = read_assignment(config.tiles.split_assignment_path)
    sensors = args.sensors.split(",") if args.sensors else None
    tile_workers = args.tile_workers if args.tile_workers is not None else sd.tile_workers

    harmonization_coefficients = None
    if config.harmonization.enabled:
        harmonization_coefficients = {
            sensor_key: {label: (c.scale, c.offset) for label, c in bands.items()}
            for sensor_key, bands in config.harmonization.coefficients.items()
        }

    rgb_cfg = config.rgb_composites
    rgb_enabled_views = {
        view
        for view, enabled in (
            ("rgb_true_color", rgb_cfg.true_color_enabled),
            ("rgb_true_color_shadow", rgb_cfg.true_color_shadow_enabled),
            ("rgb_natural_color", rgb_cfg.natural_color_enabled),
            ("rgb_color_infrared", rgb_cfg.color_infrared_enabled),
        )
        if enabled
    }

    exit_code = 0
    for split_index in splits:
        project = args.ee_project or (
            ee_projects[split_index] if split_index < len(ee_projects) else None
        )
        if not project:
            project = config.gee.project

        initialize_ee(project=project, config_default=config.gee.project, verify=True)

        tiles = tiles_for_split(registry, assignment, split_index)
        print(f"[split {split_index}] {len(tiles)} tiles, project={project}, tile_workers={tile_workers}")

        stats = run_tiles_multi_process(
            tiles,
            project=project,
            tile_dir=config.dem.tile_dir,
            cache_dir=sd.cache_dir,
            split=split_index,
            date_start_mmdd=sd.date_start_mmdd,
            date_end_mmdd=sd.date_end_mmdd,
            max_cloud_pct=sd.max_cloud_pct,
            min_aoi_coverage_pct=sd.min_aoi_coverage_pct,
            max_scenes_per_tile_month=sd.max_scenes_per_tile_month,
            min_plausible_reflectance=sd.min_plausible_reflectance,
            until_year=args.until_year,
            first_year=args.first_year,
            sensors=sensors,
            max_workers=tile_workers,
            timeout_s=sd.tile_work_timeout_s,
            harmonization_coefficients=harmonization_coefficients,
            topo_correction_enabled=config.topographic_correction.enabled,
            topo_correction_min_sun_elevation_deg=config.topographic_correction.min_sun_elevation_deg,
            topo_correction_reference_band=config.topographic_correction.reference_band,
            topo_correction_ratio_clip_min=config.topographic_correction.ratio_clip_min,
            topo_correction_ratio_clip_max=config.topographic_correction.ratio_clip_max,
            rgb_enabled_views=rgb_enabled_views,
            rgb_asinh_k=rgb_cfg.asinh_k,
            rgb_gamma=rgb_cfg.gamma,
        )
        print(f"[split {split_index}] {stats.summary()}")
        if stats.tiles_failed or stats.tiles_timed_out:
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
