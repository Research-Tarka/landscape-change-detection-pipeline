"""Balanced distribution of tiles across N processing batches (splits).

Purpose
-------
The scene-download and inference stages are run as N parallel batches, each
against its own Google Earth Engine project and output root. This module
decides which tile goes in which batch, balancing both the **tile count** and
the **total tile area** across batches.

Uses an area-weighted LPT (Longest Processing Time) load-balancing approach.

Inputs
------
- The tile registry (:mod:`.registry`), or an existing on-disk split layout.

Outputs
-------
- A split assignment table (``tile_id`` -> split index), optionally applied by
  moving per-tile directories on disk.

Algorithm
---------
Longest Processing Time (LPT) greedy: sort tiles by descending area, then
assign each in turn to the currently least-loaded batch. LPT is the classic
4/3-approximation for makespan on identical machines, and area is the best
available proxy for per-tile processing cost (scene footprint and pixel count
both scale with it). In this pipeline all tiles share the same
``tile_size_m``, so area balance and count balance mostly coincide -- but the
combined score still avoids drift when tiles differ in size (e.g. after a
pilot-area/full-AOI mix).

"Least loaded" uses a combined score that balances both criteria at once::

    score = n_tiles + total_area / mean_tile_area

The second term expresses area in units of "average tiles", so a batch that
is light on count but heavy on area is not treated as free capacity.

Design note
-----------
The number of batches is a **parameter** (``config.tiles.n_splits``) rather
than a hardcoded constant, so it can be tuned to however many parallel GEE
projects/output roots are actually available.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class SplitLoad:
    """Running load of one batch during LPT assignment."""

    index: int
    name: str
    path: Optional[Path] = None
    count: int = 0
    total_area: float = 0.0
    tile_ids: list[str] = field(default_factory=list)

    def score(self, mean_area: float) -> float:
        """Combined count + area-normalised load (lower is less loaded)."""
        area_term = (self.total_area / mean_area) if mean_area > 0 else 0.0
        return self.count + area_term

    def add(self, tile_id: str, area: float) -> None:
        self.count += 1
        self.total_area += float(area)
        self.tile_ids.append(tile_id)


def assign_lpt(
    tile_ids: Sequence[str],
    areas: Sequence[float],
    n_splits: int,
    split_names: Optional[Sequence[str]] = None,
    initial_loads: Optional[Sequence[SplitLoad]] = None,
) -> list[SplitLoad]:
    """Assign tiles to ``n_splits`` batches by descending-area LPT.

    ``initial_loads`` lets an incremental run start from the batches' existing
    occupancy, so newly added tiles land where they even things out rather
    than being distributed as if the batches were empty.

    Ties in score are broken by split index, making the assignment fully
    deterministic for a given input ordering.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if len(tile_ids) != len(areas):
        raise ValueError(
            f"tile_ids and areas must be the same length ({len(tile_ids)} vs {len(areas)})"
        )

    if initial_loads is not None:
        loads = list(initial_loads)
        if len(loads) != n_splits:
            raise ValueError(
                f"initial_loads has {len(loads)} entries but n_splits is {n_splits}"
            )
    else:
        names = list(split_names) if split_names else [f"Split{i + 1}" for i in range(n_splits)]
        loads = [SplitLoad(index=i, name=names[i]) for i in range(n_splits)]

    area_arr = np.asarray(areas, dtype=float)
    if area_arr.size == 0:
        return loads

    mean_area = float(area_arr.mean()) if area_arr.size else 1.0

    # Descending area (LPT). argsort on the negated array is a stable
    # descending sort, so equal areas keep their input order.
    order = np.argsort(-area_arr, kind="stable")

    for idx in order:
        target = min(loads, key=lambda s: (s.score(mean_area), s.index))
        target.add(str(tile_ids[idx]), float(area_arr[idx]))

    return loads


def assignment_table(loads: Sequence[SplitLoad]) -> pd.DataFrame:
    """Flatten LPT loads into a ``tile_id`` -> split assignment table."""
    rows = []
    for load in loads:
        for tile_id in load.tile_ids:
            rows.append({"tile_id": tile_id, "split_index": load.index, "split_name": load.name})
    return pd.DataFrame(rows, columns=["tile_id", "split_index", "split_name"])


def write_assignment(assignment: pd.DataFrame, path: str | Path) -> Path:
    """Write the split assignment table to Parquet."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    assignment.to_parquet(out, index=False)
    return out


def read_assignment(path: str | Path) -> pd.DataFrame:
    """Read a split assignment table written by :func:`write_assignment`."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Split assignment not found at '{p}'.")
    return pd.read_parquet(p)


def balance_report(loads: Sequence[SplitLoad]) -> pd.DataFrame:
    """Per-split count/area summary, for logging and for balance assertions."""
    return pd.DataFrame(
        [
            {
                "split_index": load.index,
                "split_name": load.name,
                "n_tiles": load.count,
                "total_area_km2": load.total_area,
            }
            for load in loads
        ]
    ).sort_values("split_index", ignore_index=True)


def balance_metrics(loads: Sequence[SplitLoad]) -> dict:
    """Spread statistics used to judge (and test) balance quality."""
    counts = np.array([load.count for load in loads], dtype=float)
    areas = np.array([load.total_area for load in loads], dtype=float)

    def _spread(values: np.ndarray) -> dict:
        if values.size == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "range": 0.0, "rel_range": 0.0}
        mean = float(values.mean())
        value_range = float(values.max() - values.min())
        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": mean,
            "range": value_range,
            "rel_range": (value_range / mean) if mean > 0 else 0.0,
        }

    return {"count": _spread(counts), "area_km2": _spread(areas)}


def move_tile(src: Path, dst_dir: Path, dry_run: bool = False) -> None:
    """Move a tile directory into ``dst_dir``, merging if it already exists.

    When the target already exists, only files missing from the target are
    copied across (never overwriting newer work already done in the target),
    and the source is then removed once nothing is left to move -- so
    re-running a split assignment after a partial download never destroys
    work already completed for a tile at its new location.
    """
    target = dst_dir / src.name
    if not target.exists():
        if not dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst_dir))
        return

    for item in src.rglob("*"):
        destination = target / item.relative_to(src)
        if item.is_dir():
            if not dry_run:
                destination.mkdir(parents=True, exist_ok=True)
        elif not destination.exists():
            if not dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, destination)
    if not dry_run:
        shutil.rmtree(src)


def plan_moves(current: dict[str, Path], target: dict[str, Path]) -> list[tuple[str, Path, Path]]:
    """Return ``(tile_id, src_split_dir, dst_split_dir)`` for tiles that move."""
    moves = []
    for tile_id, src_dir in current.items():
        dst_dir = target.get(tile_id)
        if dst_dir is not None and Path(dst_dir) != Path(src_dir):
            moves.append((tile_id, Path(src_dir), Path(dst_dir)))
    return moves


def apply_moves(
    moves: Iterable[tuple[str, Path, Path]],
    dry_run: bool = False,
    verbose: bool = True,
) -> int:
    """Execute planned moves. Returns the number of failures."""
    errors = 0
    for tile_id, src_dir, dst_dir in moves:
        if verbose:
            print(f"  {tile_id}: {src_dir.name} -> {dst_dir.name}")
        try:
            move_tile(src_dir / tile_id, dst_dir, dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] {tile_id}: {exc}")
            errors += 1
    return errors
