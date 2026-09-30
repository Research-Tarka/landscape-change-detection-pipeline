#!/usr/bin/env python3
"""Persistent land-cover object tracking (agriculture, cutblock, built-up, burn/regrowth).

Purpose
-------
For every tile, walks Stage 7's monthly classification composites and turns
each tracked class (by default: ``cultivated_agriculture``,
``cutblock_harvest``, ``built_up_infrastructure``, ``burned_disturbed``)
from a per-month state into dated intervals per pixel: when it appeared, and
when (if ever) it ended -- e.g. a cutblock that appeared in 2015 and had
regrown back to forest by 2021, or a field still cultivated as of the last
observed month. A sliding-window majority vote over observed months absorbs
single-month classification noise without requiring a strict run of
consecutive confirming months (see
:mod:`landscape_change_detection_pipeline.change.landcover_persistence` for
the full design, including why ``burned_disturbed`` reuses Stage 09's own
break date rather than being detected purely from the monthly-composite
vote). Output:
``outputs/landcover_persistence/<tile_id>/landcover_persistence.npz``.

The same run also names the change events of the tile (cropland, cutblock with
its duration and forest recovery, fire, permanent clearing, canopy decline --
cultivated agriculture, cutblock and built-up are uses, not classifier
states, so they are read from the behaviour of the states; see
:mod:`landscape_change_detection_pipeline.change.event_typing`). Output:
``outputs/change_events/<tile_id>/change_events.npz``, parameters in
``config.event_typing``.

Usage
-----
    python scripts/12_build_landcover_persistence.py --config configs/config.yaml

``tiles``/``overwrite``/``tracked_classes``/``window_size``/``min_fraction``/
``workers`` are read from ``config.landcover_persistence`` (see
:class:`landscape_change_detection_pipeline.config.LandcoverPersistenceConfig`)
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

from affine import Affine  # noqa: E402

from landscape_change_detection_pipeline.classes.class_config import load_class_config  # noqa: E402
from landscape_change_detection_pipeline.config import load_config  # noqa: E402
from landscape_change_detection_pipeline.change.change_detection import (  # noqa: E402
    read_segments,
    segments_output_path,
)
from landscape_change_detection_pipeline.change.landcover_persistence import (  # noqa: E402
    build_tile_landcover_persistence,
    read_landcover_persistence_result,
)
from landscape_change_detection_pipeline.change.event_typing import (  # noqa: E402
    EVENT_NAMES,
    change_events_output_path,
    detect_cropland,
    type_change_events,
    write_change_events,
)
from landscape_change_detection_pipeline.change.landcover_persistence import build_monthly_class_stack  # noqa: E402
from landscape_change_detection_pipeline.change.spectral_composites import discover_periods_for_tile  # noqa: E402
from landscape_change_detection_pipeline.tiles.registry import read_registry  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument("--classes", default=None, help="Path to classes.yaml")
    return parser.parse_args(argv)


def _type_events(tile_id: str, composites_root: str, all_months: list, segments_result: dict, event_args: dict) -> None:
    """Cropland / cutblock / fire ... events of one tile (see
    :mod:`change.event_typing`); needs Stage 09's segments."""
    out_path = change_events_output_path(event_args["output_root"], tile_id)
    if out_path.is_file() and not event_args["overwrite"]:
        print(f"[change_events] {tile_id}: already built, skipping (set event_typing.overwrite: true to rerun)")
        return
    stack = build_monthly_class_stack(
        composites_root, tile_id, all_months, Affine(*segments_result["transform"]),
        segments_result["crs_wkt"], segments_result["shape"],
    )
    events = type_change_events(segments_result, stack, event_args["class_ids"], event_args["masked_ids"], event_args["params"])
    cropland = detect_cropland(stack, event_args["class_ids"], event_args["masked_ids"], event_args["params"])
    write_change_events(out_path, events, cropland, segments_result)
    counts = {EVENT_NAMES[c]: int((events["event_type"] == c).sum()) for c in range(1, len(EVENT_NAMES))}
    print(f"[change_events] {tile_id}: {counts}, cropland pixels {len(cropland['row'])} -> {out_path}", flush=True)


def _build_one_tile(
    tile_id: str,
    composites_root: str,
    change_detection_output_root: str,
    output_root: str,
    class_config,
    tracked_classes: tuple[str, ...],
    window_size: int,
    min_fraction: float,
    overwrite: bool,
    options: Optional[dict] = None,
    event_args: Optional[dict] = None,
) -> tuple[str, Optional[str], Optional[str], Optional[dict]]:
    """One tile's full build, run in a worker process. Returns ``(tile_id,
    skip_reason_or_None, fallback_note_or_None, summary_or_None)`` --
    exactly one of ``skip_reason``/``summary`` is non-``None``;
    ``fallback_note`` can accompany a summary (informational, not a skip)."""
    all_months = discover_periods_for_tile(composites_root, tile_id, filename="composite.npz")
    if not all_months:
        return tile_id, "no monthly composites found, skipping (run scripts/07_build_monthly_composites.py first)", None, None

    print(f"[landcover_persistence] {tile_id}: starting ({len(all_months)} months available)", flush=True)
    segments_path = segments_output_path(change_detection_output_root, tile_id)
    segments_result = None
    fallback_note: Optional[str] = None
    if segments_path.is_file():
        segments_result = read_segments(segments_path)
        dst_transform = Affine(*segments_result["transform"])
        dst_crs_wkt = segments_result["crs_wkt"]
        dst_shape = segments_result["shape"]
        resolution_m = segments_result["resolution_m"]
    else:
        # No segmentation grid available for this tile -- fall back to its
        # own latest monthly composite's grid, and burned_disturbed simply
        # uses the generic monthly-composite detector for this tile instead
        # of Stage 09's break date (see module docstring).
        from landscape_change_detection_pipeline.inference.composites import composite_output_path, read_composite
        latest = read_composite(composite_output_path(composites_root, tile_id, sorted(all_months)[-1]))
        dst_transform = Affine(*latest["transform"])
        dst_crs_wkt = latest["crs_wkt"]
        dst_shape = latest["composite"].shape
        resolution_m = latest["resolution_m"]
        fallback_note = (
            f"no segments at '{segments_path}', burned_disturbed will use the generic "
            "monthly-composite detector (run scripts/09_run_change_detection.py first "
            "for the precise break-date variant)"
        )

    out_path = build_tile_landcover_persistence(
        composites_root=composites_root,
        output_root=output_root,
        tile_id=tile_id,
        class_config=class_config,
        all_months=all_months,
        dst_transform=dst_transform,
        dst_crs_wkt=dst_crs_wkt,
        dst_shape=dst_shape,
        resolution_m=resolution_m,
        segments_result=segments_result,
        tracked_classes=tracked_classes,
        window_size=window_size,
        min_fraction=min_fraction,
        overwrite=overwrite,
        **(options or {}),
    )
    if event_args is not None and segments_result is not None:
        _type_events(tile_id, composites_root, all_months, segments_result, event_args)
    if out_path is None:
        return tile_id, "already built, skipping (set landcover_persistence.overwrite: true to rerun)", None, None

    result = read_landcover_persistence_result(out_path)
    by_class_counts = {
        name: int((result["class_name"] == name).sum()) for name in sorted(set(result["class_name"].tolist()))
    }
    return tile_id, None, fallback_note, {
        "n_intervals": len(result["row"]),
        "n_still_active": int((result["end_month"] == "").sum()),
        "by_class_counts": by_class_counts,
        "out_path": out_path,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv if argv is not None else sys.argv[1:]))
    config = load_config(args.config, args.env_file)
    class_config = load_class_config(args.classes)
    lp_cfg = config.landcover_persistence

    registry = read_registry(config.tiles.registry_path)
    tile_ids = lp_cfg.tiles or registry["tile_id"].tolist()

    ev_cfg = config.event_typing
    from landscape_change_detection_pipeline.change.event_typing import EventTypingParams

    event_args = dict(
        output_root=ev_cfg.output_root, overwrite=ev_cfg.overwrite,
        class_ids={
            "forest": class_config.by_name(ev_cfg.forest_class).id, "grass": class_config.by_name(ev_cfg.grass_class).id,
            "bare": class_config.by_name(ev_cfg.bare_class).id, "burned": class_config.by_name(ev_cfg.burned_class).id,
        },
        masked_ids=tuple(class_config.by_name(n).id for n in ev_cfg.masked_classes),
        params=EventTypingParams(
            min_drop=ev_cfg.min_drop, min_fraction=ev_cfg.min_fraction, end_fraction=ev_cfg.end_fraction,
            burn_min_fraction=ev_cfg.burn_min_fraction, cutblock_season=tuple(ev_cfg.cutblock_season),
            min_season_obs=ev_cfg.min_season_obs, pre_seasons=ev_cfg.pre_seasons,
            min_cleared_years=ev_cfg.min_cleared_years, permanent_years=ev_cfg.permanent_years,
            permanent_bare_fraction=ev_cfg.permanent_bare_fraction, crop_season=tuple(ev_cfg.crop_season),
            crop_min_months_per_year=ev_cfg.crop_min_months_per_year,
            crop_min_run_years=ev_cfg.crop_min_run_years, crop_max_forest_fraction=ev_cfg.crop_max_forest_fraction,
            crop_allowed_gaps=ev_cfg.crop_allowed_gaps,
        ),
    )

    common_args = (
        config.composites.output_root, config.change_detection.output_root, lp_cfg.output_root,
        class_config, tuple(lp_cfg.tracked_classes), lp_cfg.window_size, lp_cfg.min_fraction, lp_cfg.overwrite,
        dict(
            masked_classes=tuple(lp_cfg.masked_classes), end_fraction=lp_cfg.end_fraction,
            min_duration_months=dict(lp_cfg.min_duration_months),
            default_min_duration_months=lp_cfg.default_min_duration_months,
            require_break_classes=tuple(lp_cfg.require_break_classes),
            break_tolerance_months=lp_cfg.break_tolerance_months, break_after_months=lp_cfg.break_after_months,
            min_break_drop=lp_cfg.min_break_drop,
        ),
        event_args,
    )

    def _report(tile_id: str, skip_reason: Optional[str], fallback_note: Optional[str], summary: Optional[dict]) -> int:
        if skip_reason is not None:
            print(f"[landcover_persistence] {tile_id}: {skip_reason}")
            return 0
        if fallback_note is not None:
            print(f"[landcover_persistence] {tile_id}: {fallback_note}")
        print(
            f"[landcover_persistence] {tile_id}: {summary['n_intervals']} intervals "
            f"({summary['n_still_active']} still active) across classes {summary['by_class_counts']} -> {summary['out_path']}"
        )
        return summary["n_intervals"]

    total_intervals = 0
    if lp_cfg.workers <= 1:
        for tile_id in tile_ids:
            tile_id, skip_reason, fallback_note, summary = _build_one_tile(tile_id, *common_args)
            total_intervals += _report(tile_id, skip_reason, fallback_note, summary)
    else:
        with ProcessPoolExecutor(max_workers=lp_cfg.workers) as pool:
            futures = [pool.submit(_build_one_tile, tile_id, *common_args) for tile_id in tile_ids]
            for future in as_completed(futures):
                tile_id, skip_reason, fallback_note, summary = future.result()
                total_intervals += _report(tile_id, skip_reason, fallback_note, summary)

    print(f"[landcover_persistence] {total_intervals} total intervals written across {len(tile_ids)} tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
