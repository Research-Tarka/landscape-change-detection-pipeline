"""Orchestrate scene discovery/download across many tiles.

Purpose
-------
This module wires together the per-(tile, sensor, year) pieces --
discovery/filtering (:mod:`.download`), fetch/enhance/store
(:mod:`.process_scene`) -- across a whole tile registry with two layers of
parallelism:

1. **Across GEE-project splits** -- one OS **subprocess** per split
   (:func:`run_splits_parallel`), each authenticating against its own
   configured GEE project. This is real quota spreading: two splits hammering
   the same GEE project would share one project's quota, not double it.
   (``subprocess.Popen`` per split, re-invoking the same
   script with ``--split N``.)

2. **Across tiles within one split** -- real OS **processes**
   (:func:`run_tiles_multi_process`), never threads. See "Why processes, not
   threads" below.

Why processes, not threads
---------------------------
A ``ThreadPoolExecutor``-based version of this exact kind of worker (one
worker per unit of work, each making Earth Engine/HTTP calls) was run live in
an earlier iteration of this project and hung
indefinitely partway through a real multi-year run: the whole process sat at
near-zero CPU with no exception, consistent with one thread's stalled network
call blocking every thread sharing that process (the GIL, plus shared
``ee``/HTTP client state, serializes what looks like concurrent work). Real
OS-level process isolation does not have that failure mode. A
``ProcessPoolExecutor`` was tried next and rejected for a different reason:
``Future.result(timeout=...)`` only stops the *parent* from waiting on a
hung future, it does not kill the still-hung worker process underneath --
leaking it forever. Real :class:`multiprocessing.Process` objects, spawned
via :func:`multiprocessing.get_context` ``"spawn"``, are used instead
precisely because a timed-out worker can be ``.terminate()``-d for real.

``spawn``, not the platform default ``fork``, is used explicitly: this
pipeline must run on Windows (no ``fork`` there at all), and ``spawn`` avoids
inheriting a parent process's already-initialized ``ee`` module / open zarr
file handles into the child in the first place -- each worker does its own
``initialize_ee()`` and opens its own cache connection.

Per-tile timeout and reassignment
-----------------------------------
:data:`TILE_WORK_TIMEOUT_S` mirrors the reference project's
``GLACIER_WORK_TIMEOUT_S`` -- a stalled GEE/network call has been observed,
live, to hang a worker indefinitely with near-zero CPU and no exception, so a
wall-clock ceiling per tile is required to make progress at all. Unlike the
reference project (which only kills a timed-out worker and moves on),
:func:`run_tiles_multi_process` additionally **requeues** a timed-out tile,
up to :data:`MAX_TILE_ATTEMPTS` attempts, before giving up on it -- a single
transient hang should not permanently drop a tile from a run.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

#: Per-tile wall-clock ceiling (seconds) before a worker is presumed hung and
#: killed (1800s): a
#: stalled earthengine-api/HTTP call has been observed to hang a worker
#: indefinitely with no exception and near-zero CPU use.
TILE_WORK_TIMEOUT_S = 1800

#: How many times a tile may be (re)launched before it is given up on and
#: recorded as failed. 1 = no retry beyond the first attempt.
MAX_TILE_ATTEMPTS = 2

#: How often the poll loop checks running workers, in seconds.
_POLL_INTERVAL_S = 0.5

#: How long to wait for a terminated/finished worker process to actually exit
#: before giving up on joining it (it is daemonic, so a leaked join here does
#: not keep the parent alive).
_JOIN_TIMEOUT_S = 10


@dataclass
class DownloadStats:
    """Aggregate counters across one or more tiles' scene downloads."""

    tiles_ok: int = 0
    tiles_failed: int = 0
    tiles_timed_out: int = 0
    scenes_ok: int = 0
    scenes_skipped: int = 0
    scenes_errored: int = 0
    failed_tile_ids: list[str] = field(default_factory=list)

    def merge(self, other: "DownloadStats") -> None:
        self.tiles_ok += other.tiles_ok
        self.tiles_failed += other.tiles_failed
        self.tiles_timed_out += other.tiles_timed_out
        self.scenes_ok += other.scenes_ok
        self.scenes_skipped += other.scenes_skipped
        self.scenes_errored += other.scenes_errored
        self.failed_tile_ids.extend(other.failed_tile_ids)

    def summary(self) -> str:
        return (
            f"tiles: {self.tiles_ok} ok, {self.tiles_failed} failed, "
            f"{self.tiles_timed_out} timed out -- scenes: {self.scenes_ok} ok, "
            f"{self.scenes_skipped} skipped, {self.scenes_errored} errored"
        )


def download_one_tile_all_sensors_years(
    tile_id: str,
    search_bbox: tuple[float, float, float, float],
    window_bbox: tuple[float, float, float, float],
    crs: str,
    tile_dir: str,
    cache_conn,
    date_start_mmdd: str,
    date_end_mmdd: str,
    max_cloud_pct: float,
    min_aoi_coverage_pct: float,
    max_scenes_per_tile_month: Optional[int],
    min_plausible_reflectance: float,
    until_year: Optional[int] = None,
    first_year: int = 1984,
    sensors: Optional[Sequence[str]] = None,
    progress_queue=None,
    harmonization_coefficients: Optional[dict[str, dict[str, tuple[float, float]]]] = None,
    topo_correction_enabled: bool = False,
    topo_correction_min_sun_elevation_deg: float = 5.0,
    topo_correction_reference_band: str = "nir",
    topo_correction_ratio_clip_min: float = 0.2,
    topo_correction_ratio_clip_max: float = 5.0,
    rgb_enabled_views: Optional[set[str]] = None,
    rgb_asinh_k: float = 8.0,
    rgb_gamma: float = 1.0 / 2.2,
) -> DownloadStats:
    """Discover and fetch every sensor/year's scenes for one tile.

    A plain top-level function (not a closure over ``ee``), so it is
    picklable and safe to reference by name across a process boundary.
    Earth Engine itself must already be initialized in-process
    (by the caller, e.g. :func:`_run_tile_worker`'s own
    :func:`~.gee_auth.initialize_ee` call) before this function is invoked;
    :mod:`.download` and :mod:`.process_scene` both do a bare ``import ee``
    internally and rely on that process-wide initialization.

    Sensors and years within one tile are handled strictly sequentially,
    deliberately: threading sensor/year downloads for one tile would mean
    concurrent appends into the same tile's zarr store from multiple workers
    (the store's own per-(path, sensor) lock in :mod:`.zarr_store` already
    serializes that anyway) and concurrent ``getInfo()``/fetch calls against
    one GEE project from one process -- no benefit, only contention.

    ``first_year`` narrows the scan's start (default 1984, the earliest any
    sensor here can have data) -- mainly useful for a test or a targeted
    rerun that already knows which year it cares about, since scanning every
    year from 1984 with lax coverage filters was confirmed live to make one
    ``compute_aoi_coverage_pct`` GEE call per historical candidate scene
    before ever filtering anything (see the live-tuning note in
    ``tests/integration/test_download_orchestration_live.py``).
    """
    from .download import discover_scenes_for_tile_sensor_year
    from .process_scene import process_and_store_scene
    from .sensors import SENSOR_ORDER, is_year_allowed

    stats = DownloadStats()
    sensor_keys = list(sensors) if sensors is not None else list(SENSOR_ORDER)
    current_year = until_year or time.localtime().tm_year

    for sensor_key in sensor_keys:
        for year in range(first_year, current_year + 1):
            if not is_year_allowed(sensor_key, year):
                continue

            result = discover_scenes_for_tile_sensor_year(
                sensor_key,
                tile_id,
                search_bbox,
                window_bbox,
                crs,
                year,
                cache_conn,
                date_start_mmdd=date_start_mmdd,
                date_end_mmdd=date_end_mmdd,
                max_cloud_pct=max_cloud_pct,
                min_aoi_coverage_pct=min_aoi_coverage_pct,
                max_scenes_per_tile_month=max_scenes_per_tile_month,
                min_plausible_reflectance=min_plausible_reflectance,
            )

            for candidate in result.kept:
                status = process_and_store_scene(
                    tile_dir, tile_id, sensor_key, candidate.scene_id, window_bbox, crs,
                    harmonization_coefficients=(harmonization_coefficients or {}).get(sensor_key),
                    topo_correction_enabled=topo_correction_enabled,
                    topo_correction_min_sun_elevation_deg=topo_correction_min_sun_elevation_deg,
                    topo_correction_reference_band=topo_correction_reference_band,
                    topo_correction_ratio_clip_min=topo_correction_ratio_clip_min,
                    topo_correction_ratio_clip_max=topo_correction_ratio_clip_max,
                    rgb_enabled_views=rgb_enabled_views,
                    rgb_asinh_k=rgb_asinh_k,
                    rgb_gamma=rgb_gamma,
                )
                if status == "ok":
                    stats.scenes_ok += 1
                elif status.startswith("skip"):
                    stats.scenes_skipped += 1
                else:
                    stats.scenes_errored += 1

            if progress_queue is not None:
                # Grouped by month (not just reported once per year), so the
                # progress log reflects the per-month budget cap: one line
                # per (year, month) that actually had kept scenes.
                kept_by_month: dict[int, int] = {}
                for candidate in result.kept:
                    kept_by_month[candidate.month] = kept_by_month.get(candidate.month, 0) + 1
                try:
                    for month in sorted(kept_by_month):
                        progress_queue.put((
                            "progress",
                            {
                                "tile_id": tile_id,
                                "sensor": sensor_key,
                                "year": year,
                                "month": month,
                                "kept": kept_by_month[month],
                                "scenes_ok": stats.scenes_ok,
                                "scenes_skipped": stats.scenes_skipped,
                                "scenes_errored": stats.scenes_errored,
                            },
                        ))
                except Exception:
                    pass

    return stats


def _run_tile_worker(
    result_queue,
    project: str,
    tile_id: str,
    search_bbox: tuple[float, float, float, float],
    window_bbox: tuple[float, float, float, float],
    crs: str,
    tile_dir: str,
    cache_dir: str,
    split: int,
    date_start_mmdd: str,
    date_end_mmdd: str,
    max_cloud_pct: float,
    min_aoi_coverage_pct: float,
    max_scenes_per_tile_month: Optional[int],
    min_plausible_reflectance: float,
    until_year: Optional[int],
    sensors: Optional[Sequence[str]],
    first_year: int = 1984,
    harmonization_coefficients: Optional[dict[str, dict[str, tuple[float, float]]]] = None,
    topo_correction_enabled: bool = False,
    topo_correction_min_sun_elevation_deg: float = 5.0,
    topo_correction_reference_band: str = "nir",
    topo_correction_ratio_clip_min: float = 0.2,
    topo_correction_ratio_clip_max: float = 5.0,
    rgb_enabled_views: Optional[set[str]] = None,
    rgb_asinh_k: float = 8.0,
    rgb_gamma: float = 1.0 / 2.2,
) -> None:
    """Entry point run inside each spawned :class:`multiprocessing.Process`.

    Does its own Earth Engine initialization and opens its own cache
    connection -- no state is inherited from the parent process (the whole
    point of ``spawn``). Puts ``("ok", DownloadStats)`` or
    ``("error", message)`` onto ``result_queue``; never lets an exception
    escape uncaught, since an uncaught exception in a daemon child would
    otherwise just look like a hang to the parent's poll loop.
    """
    try:
        from . import cache as cache_mod
        from .gee_auth import initialize_ee

        initialize_ee(project=project, verify=False)

        with cache_mod.open_cache(cache_dir, split) as conn:
            stats = download_one_tile_all_sensors_years(
                tile_id,
                search_bbox,
                window_bbox,
                crs,
                tile_dir,
                conn,
                date_start_mmdd=date_start_mmdd,
                date_end_mmdd=date_end_mmdd,
                max_cloud_pct=max_cloud_pct,
                min_aoi_coverage_pct=min_aoi_coverage_pct,
                max_scenes_per_tile_month=max_scenes_per_tile_month,
                min_plausible_reflectance=min_plausible_reflectance,
                until_year=until_year,
                first_year=first_year,
                sensors=sensors,
                progress_queue=result_queue,
                harmonization_coefficients=harmonization_coefficients,
                topo_correction_enabled=topo_correction_enabled,
                topo_correction_min_sun_elevation_deg=topo_correction_min_sun_elevation_deg,
                topo_correction_reference_band=topo_correction_reference_band,
                topo_correction_ratio_clip_min=topo_correction_ratio_clip_min,
                topo_correction_ratio_clip_max=topo_correction_ratio_clip_max,
                rgb_enabled_views=rgb_enabled_views,
                rgb_asinh_k=rgb_asinh_k,
                rgb_gamma=rgb_gamma,
            )
        result_queue.put(("ok", stats))
    except Exception as exc:  # noqa: BLE001 -- reported to the parent, never fatal
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def run_tiles_multi_process(
    tiles: Sequence[dict],
    project: str,
    tile_dir: str,
    cache_dir: str,
    split: int,
    date_start_mmdd: str = "01-01",
    date_end_mmdd: str = "12-31",
    max_cloud_pct: float = 20.0,
    min_aoi_coverage_pct: float = 50.0,
    max_scenes_per_tile_month: Optional[int] = None,
    min_plausible_reflectance: float = 0.15,
    until_year: Optional[int] = None,
    first_year: int = 1984,
    sensors: Optional[Sequence[str]] = None,
    max_workers: int = 1,
    timeout_s: int = TILE_WORK_TIMEOUT_S,
    max_attempts: int = MAX_TILE_ATTEMPTS,
    verbose: bool = True,
    worker_target=None,
    harmonization_coefficients: Optional[dict[str, dict[str, tuple[float, float]]]] = None,
    topo_correction_enabled: bool = False,
    topo_correction_min_sun_elevation_deg: float = 5.0,
    topo_correction_reference_band: str = "nir",
    topo_correction_ratio_clip_min: float = 0.2,
    topo_correction_ratio_clip_max: float = 5.0,
    rgb_enabled_views: Optional[set[str]] = None,
    rgb_asinh_k: float = 8.0,
    rgb_gamma: float = 1.0 / 2.2,
) -> DownloadStats:
    """Download scenes for many tiles in parallel within one split.

    ``tiles`` is a sequence of dicts, each with keys ``tile_id``,
    ``search_bbox``, ``window_bbox``, ``crs`` (one row of the tile registry,
    already restricted to this split). Real ``multiprocessing.Process``
    objects (``spawn`` context) are used -- see the module docstring for why
    threads and ``ProcessPoolExecutor`` were both rejected.

    Never assigns two workers to the same tile concurrently, so there is
    never a concurrent zarr append into one tile's store from two processes.
    A worker that exceeds ``timeout_s`` is ``.terminate()``-d and, unlike the
    reference project, requeued for another attempt (up to ``max_attempts``
    total) before being recorded as failed.

    ``worker_target`` defaults to :func:`_run_tile_worker` (the real,
    GEE-touching worker); tests substitute a fast, picklable stand-in with
    the same ``(result_queue, project, tile_id, ...)`` signature to exercise
    the poll loop's concurrency/timeout/requeue logic without any network
    access.
    """
    import multiprocessing as mp

    target = worker_target or _run_tile_worker
    ctx = mp.get_context("spawn")
    max_concurrent = max(1, min(max_workers, len(tiles) or 1))

    total = DownloadStats()
    pending: list[tuple[dict, int]] = [(tile, 1) for tile in tiles]
    running: dict[str, tuple] = {}  # tile_id -> (process, queue, attempt, started_at)
    final_results: dict[str, tuple] = {}  # tile_id -> ("ok"|"error", payload), drained early

    def _drain_progress(tile_id: str, queue) -> None:
        """Print progress messages and stash the final ("ok"/"error") result, if seen.

        A tile worker's queue carries zero or more ``"progress"`` messages
        followed by exactly one final ``"ok"``/``"error"`` message; this
        drains whatever is currently available without blocking, so progress
        prints as soon as it lands instead of only after the whole tile ends.
        """
        while True:
            try:
                item = queue.get_nowait()
            except Exception:
                break
            status, payload = item
            if status == "progress":
                if verbose:
                    p = payload
                    month_str = f"{p['year']}-{p['month']:02d}"
                    print(
                        f"[split {split}] {p['tile_id']} {p['sensor']} {month_str}: "
                        f"+{p['kept']} scenes (ok={p['scenes_ok']} skip={p['scenes_skipped']} err={p['scenes_errored']})"
                    )
            else:
                final_results[tile_id] = item

    def _launch(tile: dict, attempt: int) -> None:
        tile_id = tile["tile_id"]
        queue = ctx.Queue()
        process = ctx.Process(
            target=target,
            args=(
                queue,
                project,
                tile_id,
                tuple(tile["search_bbox"]),
                tuple(tile["window_bbox"]),
                tile["crs"],
                tile_dir,
                cache_dir,
                split,
                date_start_mmdd,
                date_end_mmdd,
                max_cloud_pct,
                min_aoi_coverage_pct,
                max_scenes_per_tile_month,
                min_plausible_reflectance,
                until_year,
                sensors,
                first_year,
                harmonization_coefficients,
                topo_correction_enabled,
                topo_correction_min_sun_elevation_deg,
                topo_correction_reference_band,
                topo_correction_ratio_clip_min,
                topo_correction_ratio_clip_max,
                rgb_enabled_views,
                rgb_asinh_k,
                rgb_gamma,
            ),
            daemon=True,
        )
        process.start()
        running[tile_id] = (process, queue, attempt, time.time())

    while pending or running:
        while pending and len(running) < max_concurrent:
            tile, attempt = pending.pop(0)
            _launch(tile, attempt)

        finished_ids = []
        for tile_id, (process, queue, attempt, started_at) in list(running.items()):
            _drain_progress(tile_id, queue)

            if process.is_alive():
                if time.time() - started_at > timeout_s:
                    process.terminate()
                    process.join(timeout=_JOIN_TIMEOUT_S)
                    finished_ids.append(tile_id)
                    final_results.pop(tile_id, None)
                    if attempt < max_attempts:
                        tile = next(t for t in tiles if t["tile_id"] == tile_id)
                        pending.append((tile, attempt + 1))
                        if verbose:
                            print(f"[TIMEOUT] {tile_id}: attempt {attempt} exceeded {timeout_s}s, requeuing")
                    else:
                        total.tiles_timed_out += 1
                        total.failed_tile_ids.append(tile_id)
                        if verbose:
                            print(f"[TIMEOUT] {tile_id}: giving up after {attempt} attempts")
                continue

            if tile_id in final_results:
                status, payload = final_results.pop(tile_id)
            else:
                try:
                    status, payload = queue.get_nowait()
                except Exception:
                    status, payload = "error", "worker process exited without a result"

            if status == "ok":
                total.merge(payload)
                total.tiles_ok += 1
            else:
                if verbose:
                    print(f"[ERROR] {tile_id}: {payload}")
                if attempt < max_attempts:
                    tile = next(t for t in tiles if t["tile_id"] == tile_id)
                    pending.append((tile, attempt + 1))
                else:
                    total.tiles_failed += 1
                    total.failed_tile_ids.append(tile_id)

            process.join(timeout=_JOIN_TIMEOUT_S)
            finished_ids.append(tile_id)

        for tile_id in finished_ids:
            del running[tile_id]

        if running and not finished_ids:
            time.sleep(_POLL_INTERVAL_S)

    return total


def resolve_splits(arg: str, n_configured: int) -> list[int]:
    """Turn a ``--split`` CLI argument (``"all"`` or an int string) into split indices.

    Splits are 0-based internally (matching ``tiles.splitter``'s
    ``split_index``), so the valid range is ``[0, n_configured)``.
    """
    if arg == "all":
        return list(range(n_configured))
    try:
        value = int(arg)
    except ValueError:
        raise ValueError(f"--split must be 'all' or an integer, got {arg!r}") from None
    if not (0 <= value < n_configured):
        raise ValueError(f"--split {value} out of range [0, {n_configured})")
    return [value]


def strip_split_args(argv: Sequence[str]) -> list[str]:
    """Remove ``--split``/``--split=N`` and ``--parallel`` from ``argv``.

    Used before re-invoking this script once per split via
    :func:`run_splits_parallel`: each child process gets its own explicit
    ``--split N``, and must not also see ``--parallel`` (which would recurse).
    """
    result = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token == "--split":
            skip_next = True
            continue
        if token.startswith("--split="):
            continue
        if token == "--parallel":
            continue
        result.append(token)
    return result


def run_splits_parallel(argv: Sequence[str], splits: Sequence[int], script: Optional[str] = None) -> int:
    """Run one OS subprocess per split, each re-invoking this same script.

    Real quota spreading: each subprocess authenticates against its own
    split's configured GEE project (resolved independently inside the child),
    so N splits draw on N separate projects' quotas
    concurrently rather than one project's quota shared across N "concurrent"
    callers.
    """
    import subprocess

    base_args = strip_split_args(argv)
    script_path = script or sys.argv[0]

    processes: dict[int, "subprocess.Popen"] = {}
    for split_num in splits:
        cmd = [sys.executable, script_path, "--split", str(split_num), *base_args]
        processes[split_num] = subprocess.Popen(cmd)

    exit_code = 0
    for split_num, proc in processes.items():
        code = proc.wait()
        if code != 0:
            exit_code = exit_code or code
    return exit_code


def tiles_for_split(registry, assignment, split_index: int) -> list[dict]:
    """Join the tile registry against a split assignment, for one split index.

    Returns a list of plain dicts (picklable, no DataFrame/Series objects
    crossing the process boundary) with the keys
    :func:`run_tiles_multi_process` expects.
    """
    tile_ids = set(assignment.loc[assignment["split_index"] == split_index, "tile_id"])
    subset = registry[registry["tile_id"].isin(tile_ids)]

    rows = []
    for _, row in subset.iterrows():
        rows.append(
            {
                "tile_id": str(row["tile_id"]),
                "search_bbox": (
                    float(row["search_minx"]),
                    float(row["search_miny"]),
                    float(row["search_maxx"]),
                    float(row["search_maxy"]),
                ),
                "window_bbox": (
                    float(row["window_minx"]),
                    float(row["window_miny"]),
                    float(row["window_maxx"]),
                    float(row["window_maxy"]),
                ),
                "crs": str(row["crs"]),
            }
        )
    return rows
