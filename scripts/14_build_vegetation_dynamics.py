#!/usr/bin/env python3
"""Vegetation dynamics (phenology, trends, anomalies, variability) and event recovery / succession.

Purpose
-------
Two families of new data, both built from one monthly index cube per tile
(NDVI by default, one scene pass, temporary and on disk):

``vegetation_dynamics`` (what the vegetation does between breaks)
    - **phenology**: per year, peak month/value, amplitude, start / end /
      length of the growing season;
    - **trend**: Theil-Sen slope + Mann-Kendall of the yearly series (slow
      greening / browning), also since each pixel's latest break, and of the
      phenology series (is the season getting longer);
    - **anomalies**: monthly z-score against the pixel's own climatology
      (droughts, late snowmelt), reduced per year;
    - **variability**: interannual coefficient of variation / mean
      year-to-year change.
    Output: ``outputs/vegetation_dynamics/<tile_id>/vegetation_dynamics.npz``
    (+ ``monthly_anomaly_<index>.npz`` if ``anomalies.store_monthly``).

``recovery_analysis`` (how the land comes back after each named event)
    - **curves**: yearly index after each event relative to its pre-event
      baseline, recovery time (censoring explicit), trough, rate;
    - **succession**: dominant class year by year after each event, per-event
      summaries, and pooled Markov / class-share tables per event type;
    - **factors**: the event table joined with terrain (DEM), severity and
      pre-event level, plus a binned summary (event type x elevation / slope /
      aspect).
    Needs Stage 12's events (``event_typing``). Output:
    ``outputs/recovery_analysis/<tile_id>/recovery_analysis.npz``.

See :mod:`landscape_change_detection_pipeline.change.vegetation_dynamics` and
:mod:`landscape_change_detection_pipeline.change.recovery_analysis` for the
full definitions. Every element has its own ``enabled`` switch in
``config.vegetation_dynamics`` / ``config.recovery_analysis``; both files are
browsable and exportable (GeoTIFF / CSV) from ``scripts/export_gui.py``.

Usage
-----
    python scripts/14_build_vegetation_dynamics.py --config configs/config.yaml

``tiles``/``overwrite``/``workers`` are read from ``config.vegetation_dynamics``
(not CLI flags: a run is reproducible from ``config.yaml`` alone).
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from affine import Affine  # noqa: E402

from landscape_change_detection_pipeline.change.change_detection import read_segments, segments_output_path  # noqa: E402
from landscape_change_detection_pipeline.change.event_typing import change_events_output_path, read_change_events  # noqa: E402
from landscape_change_detection_pipeline.change.landcover_persistence import build_monthly_class_stack  # noqa: E402
from landscape_change_detection_pipeline.change.recovery_analysis import (  # noqa: E402
    analyse_events,
    recovery_analysis_output_path,
    write_recovery_analysis,
)
from landscape_change_detection_pipeline.change.spectral_composites import discover_periods_for_tile  # noqa: E402
from landscape_change_detection_pipeline.change.vegetation_dynamics import (  # noqa: E402
    build_index_cubes,
    cleanup_cubes,
    compute_tile_dynamics,
    latest_break_year_index,
    make_calendar,
    monthly_anomaly_output_path,
    vegetation_dynamics_output_path,
    write_monthly_anomaly,
    write_vegetation_dynamics,
)
from landscape_change_detection_pipeline.classes.class_config import load_class_config  # noqa: E402
from landscape_change_detection_pipeline.config import load_config  # noqa: E402
from landscape_change_detection_pipeline.tiles.registry import read_registry  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument("--classes", default=None, help="Path to classes.yaml")
    return parser.parse_args(argv)


def _build_one_tile(tile_id: str, config, class_config) -> tuple[str, Optional[str], dict]:
    """One tile, run in a worker process. Returns ``(tile_id, skip_reason_or_None,
    summary)``; ``summary`` maps each written product to its path."""
    vd, ra = config.vegetation_dynamics, config.recovery_analysis
    prefix = f"[vegetation_dynamics] {tile_id}: "
    t_start = time.time()

    vd_path = vegetation_dynamics_output_path(vd.output_root, tile_id)
    ra_path = recovery_analysis_output_path(ra.output_root, tile_id)
    do_vd = vd.enabled and (vd.overwrite or not vd_path.is_file())
    do_ra = ra.enabled and (vd.overwrite or ra.overwrite or not ra_path.is_file())
    if vd.enabled and not do_vd:
        print(f"{prefix}vegetation_dynamics already built, skipping (set vegetation_dynamics.overwrite: true to rerun)", flush=True)
    if ra.enabled and not do_ra:
        print(f"{prefix}recovery_analysis already built, skipping (set recovery_analysis.overwrite: true to rerun)", flush=True)
    if not (do_vd or do_ra):
        return tile_id, "nothing to do (disabled or already built)", {}

    all_months = discover_periods_for_tile(config.composites.output_root, tile_id, filename="composite.npz")
    if not all_months:
        return tile_id, "no monthly composites found, skipping (run scripts/07_build_monthly_composites.py first)", {}
    segments_path = segments_output_path(config.change_detection.output_root, tile_id)
    if not segments_path.is_file():
        return tile_id, f"no segments at '{segments_path}', skipping (run scripts/09_run_change_detection.py first)", {}

    events = None
    if do_ra:
        events_path = change_events_output_path(config.event_typing.output_root, tile_id)
        if events_path.is_file():
            events = read_change_events(events_path)
        else:
            print(f"{prefix}no change events at '{events_path}': recovery_analysis skipped "
                  "(run scripts/12_build_landcover_persistence.py first)", flush=True)
            do_ra = False
    if do_ra and ra.index not in vd.indices:
        print(f"{prefix}recovery_analysis.index '{ra.index}' is not in vegetation_dynamics.indices {vd.indices}: "
              "recovery_analysis skipped", flush=True)
        do_ra = False
    if not (do_vd or do_ra):
        return tile_id, "nothing to do", {}

    print(f"{prefix}starting ({len(all_months)} months; "
          f"vegetation_dynamics={'on' if do_vd else 'off'}, recovery_analysis={'on' if do_ra else 'off'})", flush=True)
    segments = read_segments(segments_path)
    transform, crs_wkt, shape, resolution = segments["transform"], segments["crs_wkt"], segments["shape"], segments["resolution_m"]
    dst_transform = Affine(*transform[:6])
    calendar = make_calendar(all_months)

    need_stack = (bool(vd.mask_classes) and (do_vd or do_ra)) or (do_ra and ra.succession.enabled)
    class_stack = None
    if need_stack:
        print(f"{prefix}reading monthly class composites...", flush=True)
        class_stack = build_monthly_class_stack(config.composites.output_root, tile_id, all_months, dst_transform, crs_wkt, shape)
    mask_ids = tuple(class_config.by_name(n).id for n in vd.mask_classes)
    indices = list(vd.indices)

    work_dir = Path(vd.output_root) / tile_id / "_work"
    cubes = build_index_cubes(
        config.dem.tile_dir, config.inference.output_root, tile_id, class_config, all_months, indices,
        dst_transform, crs_wkt, shape, work_dir, class_stack, mask_ids, vd.month_threads,
    )
    written: dict[str, str] = {}
    try:
        break_idx = latest_break_year_index(segments, calendar.year0)
        annual: dict = {}
        print(f"{prefix}computing per-pixel dynamics ({', '.join(indices)})...", flush=True)
        # recovery_analysis only needs the yearly series: switch the other elements off when vegetation_dynamics is
        cfg_run = vd if do_vd else vd.model_copy(update={
            k: getattr(vd, k).model_copy(update={"enabled": False}) for k in ("phenology", "trend", "anomalies", "variability")
        })
        arrays, monthly_z = compute_tile_dynamics(cubes, calendar, cfg_run, break_idx, tile_id, annual_out=annual)
        if do_vd:
            write_vegetation_dynamics(vd_path, arrays, calendar, transform, crs_wkt, resolution, vd)
            written["vegetation_dynamics"] = str(vd_path)
            print(f"{prefix}vegetation_dynamics -> {vd_path} ({len(arrays)} arrays)", flush=True)
            for name, z in (monthly_z or {}).items():
                zp = monthly_anomaly_output_path(vd.output_root, tile_id, name)
                write_monthly_anomaly(zp, z, calendar, transform, crs_wkt, resolution)
                written[f"monthly_anomaly_{name.lower()}"] = str(zp)
                print(f"{prefix}monthly anomalies ({name}) -> {zp}", flush=True)
        if do_ra:
            out = analyse_events(
                events, annual[ra.index], calendar.year0, class_stack,
                [c.id for c in class_config.classes], [c.name for c in class_config.classes], ra,
                (int(vd.growing_season[0]), int(vd.growing_season[1])), transform, crs_wkt, shape,
                config.dem.tile_dir, tile_id, mask_ids,
            )
            if out:
                write_recovery_analysis(ra_path, out, transform, crs_wkt, resolution, events["event_names"], calendar.year0)
                written["recovery_analysis"] = str(ra_path)
                print(f"{prefix}recovery_analysis -> {ra_path}", flush=True)
            else:
                print(f"{prefix}recovery_analysis: no event of the configured types, nothing written", flush=True)
    finally:
        cleanup_cubes(cubes, work_dir)
    print(f"{prefix}done in {time.time() - t_start:.0f}s", flush=True)
    return tile_id, None, written


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv if argv is not None else sys.argv[1:]))
    config = load_config(args.config, args.env_file)
    class_config = load_class_config(args.classes)
    vd = config.vegetation_dynamics

    if not (vd.enabled or config.recovery_analysis.enabled):
        print("[vegetation_dynamics] vegetation_dynamics.enabled and recovery_analysis.enabled are both false: nothing to do")
        return 0
    registry = read_registry(config.tiles.registry_path)
    tile_ids = vd.tiles or registry["tile_id"].tolist()
    print(f"[vegetation_dynamics] {len(tile_ids)} tiles, workers={vd.workers}, indices={vd.indices}", flush=True)

    n_done = 0

    def _report(tile_id: str, skip_reason: Optional[str], written: dict) -> int:
        if skip_reason is not None:
            print(f"[vegetation_dynamics] {tile_id}: {skip_reason}", flush=True)
            return 0
        return 1

    if vd.workers <= 1:
        for tile_id in tile_ids:
            n_done += _report(*_build_one_tile(tile_id, config, class_config))
    else:
        with ProcessPoolExecutor(max_workers=vd.workers) as pool:
            futures = [pool.submit(_build_one_tile, tile_id, config, class_config) for tile_id in tile_ids]
            for future in as_completed(futures):
                n_done += _report(*future.result())

    print(f"[vegetation_dynamics] {n_done}/{len(tile_ids)} tiles processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
