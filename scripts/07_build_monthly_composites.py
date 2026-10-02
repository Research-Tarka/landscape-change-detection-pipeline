#!/usr/bin/env python3
"""Stage 7 -- build monthly categorical composites from per-scene inference.

Purpose
-------
For every tile, group its Stage 6 per-scene class maps by
(year, month), reproject every sensor's scenes that month onto the finest
grid available (``composites.sensor_resolution_priority``), and reduce to
one composite per the configured per-class rule (median vs. any-occurrence,
``composites.class_rules``).

Usage
-----
    python scripts/07_build_monthly_composites.py --config configs/config.yaml

``tiles``/``overwrite`` are read from ``config.composites`` (see
:class:`landscape_change_detection_pipeline.config.CompositesConfig`), not
CLI flags -- a run must be reproducible from ``config.yaml`` alone. Only
``--config``/``--env-file``/``--classes`` (which files to read) stay as
flags.

Progress reporting
-------------------
A tile can have hundreds of months; printing only once a whole tile
finishes gives no signal for a long time on a real run. Every month
processed (written or already-built and skipped) prints one line, whether
``composites.workers`` is 1 (direct call, same process) or >1 (each worker
process reports through a shared ``multiprocessing.Queue`` -- a child
process's own ``print`` is not reliably visible from the parent, especially
on Windows, so progress must be relayed explicitly rather than printed
in-worker).
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landscape_change_detection_pipeline.classes.class_config import load_class_config  # noqa: E402
from landscape_change_detection_pipeline.config import load_config  # noqa: E402
from landscape_change_detection_pipeline.inference.composites import build_all_monthly_composites  # noqa: E402
from landscape_change_detection_pipeline.tiles.registry import read_registry  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument("--classes", default=None, help="Path to classes.yaml")
    return parser.parse_args(argv)


def _print_progress(tile_id: str, month: str, month_index: int, total_months: int, wrote_this_month: bool) -> None:
    status = "wrote" if wrote_this_month else "already built, skipped"
    print(f"[composites] {tile_id} {month} ({month_index}/{total_months}): {status}")


def _queue_progress(queue: "multiprocessing.Queue", tile_id: str, month: str, month_index: int, total_months: int, wrote_this_month: bool) -> None:
    """``progress_callback`` for a worker process: relays the same event
    through ``queue`` instead of printing directly (see module docstring)."""
    queue.put((tile_id, month, month_index, total_months, wrote_this_month))


def _build_one_tile(
    tile_id: str,
    inference_root: str,
    output_root: str,
    num_classes: int,
    sensor_priority: tuple[str, ...],
    default_rule: str,
    class_rules: dict[int, tuple[str, int]],
    ignore_class_ids: tuple[int, ...],
    overwrite: bool,
    progress_queue: "multiprocessing.Queue | None",
) -> tuple[str, int]:
    """One tile's full ``build_all_monthly_composites`` call, run in a worker
    process when ``composites.workers > 1`` -- tiles never share state (each
    reads only its own ``inference_root/<tile_id>`` subtree and writes only
    its own ``output_root/<tile_id>`` subtree), so this parallelizes safely.
    ``progress_queue`` is ``None`` when called directly (workers<=1) --
    progress is then printed in this same process without an intermediate
    queue."""
    from functools import partial

    callback = _print_progress if progress_queue is None else partial(_queue_progress, progress_queue)
    written = build_all_monthly_composites(
        inference_root=inference_root,
        output_root=output_root,
        tile_id=tile_id,
        num_classes=num_classes,
        sensor_priority=sensor_priority,
        default_rule=default_rule,
        class_rules=class_rules,
        overwrite=overwrite,
        progress_callback=callback,
        ignore_class_ids=ignore_class_ids,
    )
    return tile_id, len(written)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv if argv is not None else sys.argv[1:]))
    config = load_config(args.config, args.env_file)
    class_config = load_class_config(args.classes)
    # The class maps hold RAW ids (06 converts dense -> raw), which are not
    # contiguous (e.g. 11/12 retired, 16/17 appended): the vote arrays must be
    # sized by the highest raw id, not by the class count, or a class whose id
    # is >= len(classes) silently casts no vote.
    num_classes = max(c.id for c in class_config.classes) + 1
    comp_cfg = config.composites
    ignore_class_ids = tuple(class_config.by_name(n).id for n in comp_cfg.ignore_classes)

    class_rules = {rule.class_id: (rule.rule, rule.priority) for rule in comp_cfg.class_rules}

    registry = read_registry(config.tiles.registry_path)
    tile_ids = comp_cfg.tiles or registry["tile_id"].tolist()

    base_args = (
        config.inference.output_root,
        comp_cfg.output_root,
        num_classes,
        tuple(comp_cfg.sensor_resolution_priority),
        comp_cfg.default_rule,
        class_rules,
        ignore_class_ids,
        comp_cfg.overwrite,
    )

    total_written = 0
    if comp_cfg.workers <= 1:
        for tile_id in tile_ids:
            tile_id, n_written = _build_one_tile(tile_id, *base_args, None)
            total_written += n_written
    else:
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        with ProcessPoolExecutor(max_workers=comp_cfg.workers) as pool:
            futures = [pool.submit(_build_one_tile, tile_id, *base_args, progress_queue) for tile_id in tile_ids]
            pending = set(futures)
            while pending:
                # Drain whatever progress events have arrived so far, then
                # check for finished tiles -- interleaving these two polls
                # (rather than blocking on either alone) is what keeps
                # progress lines appearing live instead of arriving all at
                # once when the last tile finishes.
                while not progress_queue.empty():
                    tile_id, month, month_index, total_months, wrote_this_month = progress_queue.get()
                    _print_progress(tile_id, month, month_index, total_months, wrote_this_month)
                done_now = {f for f in pending if f.done()}
                for future in done_now:
                    tile_id, n_written = future.result()
                    total_written += n_written
                    pending.discard(future)
                if not done_now:
                    time.sleep(0.5)  # nothing finished this pass -- avoid busy-looping
            # Final drain in case events arrived after the last future settled.
            while not progress_queue.empty():
                tile_id, month, month_index, total_months, wrote_this_month = progress_queue.get()
                _print_progress(tile_id, month, month_index, total_months, wrote_this_month)

    print(f"[composites] {total_written} composites written across {len(tile_ids)} tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
