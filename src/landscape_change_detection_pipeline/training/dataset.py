"""Train/val/test split over annotated scenes, plus train-only normalization
statistics and class weights.

The leakage-preventing spatial unit is the **tile**: every
scene belonging to one tile lands in exactly one partition. Scenes from the
same tile are spatially and temporally correlated (same footprint,
overlapping years), so splitting them across partitions would let the model
recognise a tile it has already seen and report that as generalisation.

Key design points:

**Determinism is enforced by sorting, not by seeding alone.** Python
randomises string hashing per process, so iterating a ``set`` of tile ids and
taking the first match yields a different tile on every run even with the RNG
seeded -- a documented class of reproducibility bug. Sets here are used for
membership tests only; iteration order always comes from a sorted list.

**The split is tile-level**, stratified by scene count (three strata: 1
scene, 2-4 scenes, 5+ scenes) so a partition cannot end up holding all the
heavily-imaged tiles purely by chance ordering.

**Three repair passes run in sequence** after the initial stratified
assignment: each partition must see every sensor, every class, and every
sensor-class combination present in the corpus. Training keeps priority over
val/test when they conflict (a sensor/class absent from training is one the
model can never learn; absent from val/test only costs a measurement). Each
pass moves *whole tiles*, never individual scenes, so tile-level exclusivity
survives every pass. The hard exclusivity invariant is checked explicitly
afterwards, and a proportion-drift warning (not a failure) is raised if a
partition's actual scene share drifts far from its target -- tiles differ
hugely in scene count, so a split correct by tile count can still skew scene
proportions.

**Train-only normalization statistics.** :func:`compute_mean_std` and
:func:`compute_class_counts` must only ever be called with train-partition
scenes -- leaking val/test into either would let held-out information
influence every training example.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from landscape_change_detection_pipeline.features.training_cache import CACHE_FILENAME, load_scene_cache

logger = logging.getLogger(__name__)

PARTITIONS: tuple[str, ...] = ("train", "val", "test")

#: Default target ratios, (train, val, test).
DEFAULT_RATIOS: tuple[float, float, float] = (0.7, 0.15, 0.15)


class SplitError(Exception):
    """Raised when the split cannot be built, or when the hard
    tile-exclusivity invariant is violated."""


@dataclass(frozen=True)
class SceneRecord:
    """One annotated scene discovered under a ``train_root`` cache tree."""

    tile_id: str
    sensor: str
    scene_id: str
    cache_dir: Path

    @property
    def scene_key(self) -> str:
        return f"{self.tile_id}_{self.sensor}_{self.scene_id}"

    @property
    def sort_key(self) -> tuple[str, str, str]:
        return (self.tile_id, self.sensor, self.scene_id)


@dataclass(frozen=True)
class SplitResult:
    """The outcome of :func:`split_scenes`: a partition assignment per tile,
    plus the scene records grouped by partition."""

    tile_partition: dict[str, str]
    scenes_by_partition: dict[str, list[SceneRecord]]
    warnings: list[str]

    def partition_of(self, tile_id: str) -> str:
        return self.tile_partition[tile_id]


# -- discovery -----------------------------------------------------------


def discover_scene_records(train_root: str | Path) -> list[SceneRecord]:
    """Scan a ``train_root`` cache tree
    (``<train_root>/<tile_id>/<sensor>/<scene_id>/features.npz``) and return
    every scene found, sorted by ``(tile_id, sensor, scene_id)``.

    ``os.walk``/directory listing order is never trusted: every level is
    sorted explicitly, and the result is sorted again as a whole.
    """
    root = Path(train_root)
    records: list[SceneRecord] = []
    if not root.is_dir():
        return records

    for tile_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for sensor_dir in sorted(p for p in tile_dir.iterdir() if p.is_dir()):
            for scene_dir in sorted(p for p in sensor_dir.iterdir() if p.is_dir()):
                if not (scene_dir / CACHE_FILENAME).is_file():
                    continue
                records.append(
                    SceneRecord(
                        tile_id=tile_dir.name,
                        sensor=sensor_dir.name,
                        scene_id=scene_dir.name,
                        cache_dir=scene_dir,
                    )
                )

    records.sort(key=lambda r: r.sort_key)
    return records


def group_scenes_by_tile(records: Sequence[SceneRecord]) -> dict[str, list[int]]:
    """Map tile id to its scene indices. Insertion order follows the sorted
    scene list; keys are sorted again wherever they drive a decision."""
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.tile_id, []).append(index)
    return groups


def scene_class_counts(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Per-class pixel counts for one scene's ``(H, W)`` label array."""
    flat = labels.reshape(-1)
    valid = flat[(flat >= 0) & (flat < num_classes)]
    if valid.size == 0:
        return np.zeros(num_classes, dtype=np.int64)
    return np.bincount(valid, minlength=num_classes)[:num_classes].astype(np.int64)


# -- the split -------------------------------------------------------------
#
# Every function below takes/returns *sorted lists* of tile ids, never sets,
# for the reason given in the module docstring. Membership tests may use a
# set built locally; iteration order never comes from one.


def _stratify_by_size(
    tile_groups: Mapping[str, list[int]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Optional[tuple[list[str], list[str], list[str]]]:
    """Split tiles into train/val/test, stratified by scene count.

    Three strata -- 1 scene, 2-4 scenes, 5+ scenes -- each split
    independently, so a partition cannot end up holding all the
    heavily-imaged tiles. Each stratum's tile list is sorted before
    shuffling, so a given seed always produces the same assignment.
    """
    if not tile_groups:
        return None

    sizes = {tid: len(indices) for tid, indices in tile_groups.items()}
    strata = {
        "small": sorted(t for t, n in sizes.items() if n <= 1),
        "medium": sorted(t for t, n in sizes.items() if 2 <= n <= 4),
        "large": sorted(t for t, n in sizes.items() if n >= 5),
    }

    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    rng = random.Random(seed)

    for name in ("small", "medium", "large"):
        members = list(strata[name])
        if not members:
            continue
        rng.shuffle(members)
        n = len(members)

        n_test = int(round(n * test_ratio)) if test_ratio > 0 else 0
        n_val = int(round(n * val_ratio)) if val_ratio > 0 else 0

        # Never empty a stratum's training share.
        if n_test + n_val >= n:
            n_val = max(0, n - 1 - n_test)
        if n_test + n_val >= n:
            n_test = max(0, n - 1 - n_val)
        if val_ratio > 0 and n_val == 0 and n > n_test + 1:
            n_val = 1
        if test_ratio > 0 and n_test == 0 and n > n_val + 1:
            n_test = 1

        test.extend(members[:n_test])
        val.extend(members[n_test : n_test + n_val])
        train.extend(members[n_test + n_val :])

    if not train:
        return None
    return sorted(train), sorted(val), sorted(test)


def _enforce_sensor_coverage(
    tile_groups: Mapping[str, list[int]],
    scene_sensors: Sequence[str],
    train: list[str],
    val: list[str],
    test: list[str],
    val_ratio: float,
    test_ratio: float,
) -> None:
    """Move tiles so each partition sees every available sensor.

    Modifies the three lists in place, keeping them sorted. Training has
    priority: a sensor missing there is recovered from val/test, since a
    sensor absent from training is one the model can never learn.
    """
    sensors_per_tile = {
        tid: {scene_sensors[i] for i in indices} for tid, indices in tile_groups.items()
    }
    available = sorted({sensor for sensors in sensors_per_tile.values() for sensor in sensors})

    def _has(partition: list[str], sensor: str) -> bool:
        return any(sensor in sensors_per_tile.get(tid, ()) for tid in partition)

    def _move(sensor: str, donor: list[str], destination: list[str]) -> bool:
        if len(donor) <= 1:
            return False
        for tid in sorted(donor):
            if sensor in sensors_per_tile.get(tid, ()):
                donor.remove(tid)
                destination.append(tid)
                destination.sort()
                return True
        return False

    if val_ratio > 0:
        for sensor in available:
            if not _has(val, sensor):
                _move(sensor, train, val)
    if test_ratio > 0:
        for sensor in available:
            if not _has(test, sensor):
                _move(sensor, train, test)

    for sensor in available:
        if not _has(train, sensor):
            _move(sensor, val, train) or _move(sensor, test, train)


def _enforce_class_coverage(
    tile_groups: Mapping[str, list[int]],
    class_counts: Sequence[np.ndarray],
    num_classes: int,
    train: list[str],
    val: list[str],
    test: list[str],
    val_ratio: float,
    test_ratio: float,
) -> None:
    """Move tiles so each partition contains every class.

    When a class is missing, the donor is the tile holding the most pixels
    of it -- giving the receiving partition enough of the class to measure
    rather than a token handful. Ties break on tile id.
    """
    counts_per_tile = {
        tid: sum(
            (class_counts[i] for i in indices),
            start=np.zeros(num_classes, dtype=np.int64),
        )
        for tid, indices in tile_groups.items()
    }

    def _has(partition: list[str], class_id: int) -> bool:
        return any(counts_per_tile[tid][class_id] > 0 for tid in partition)

    def _move(class_id: int, donor: list[str], destination: list[str]) -> bool:
        if len(donor) <= 1:
            return False
        candidates = [tid for tid in sorted(donor) if counts_per_tile[tid][class_id] > 0]
        if not candidates:
            return False
        best = max(candidates, key=lambda tid: (int(counts_per_tile[tid][class_id]), tid))
        donor.remove(best)
        destination.append(best)
        destination.sort()
        return True

    if val_ratio > 0:
        for class_id in range(num_classes):
            if not _has(val, class_id):
                _move(class_id, train, val)
    if test_ratio > 0:
        for class_id in range(num_classes):
            if not _has(test, class_id):
                _move(class_id, train, test)

    for class_id in range(num_classes):
        if not _has(train, class_id):
            _move(class_id, val, train) or _move(class_id, test, train)


def _enforce_sensor_class_coverage(
    tile_groups: Mapping[str, list[int]],
    class_counts: Sequence[np.ndarray],
    scene_sensors: Sequence[str],
    num_classes: int,
    train: list[str],
    val: list[str],
    test: list[str],
    val_ratio: float,
    test_ratio: float,
) -> list[str]:
    """Try to give each partition every sensor-class combination present in
    the corpus. Finer than the two passes above: a partition can hold
    sensor A scenes and class X pixels while containing no sensor-A/class-X
    pixel. Donor selection prefers a tile whose removal costs the donor
    partition the fewest sensor-class combinations, so repairing one gap
    does not open another.

    Some combinations may be structurally unsatisfiable (a rare
    sensor/class pair concentrated in a single tile). Returns a list of
    human-readable warnings rather than failing.
    """
    tile_pairs: dict[str, set[tuple[str, int]]] = {}
    for tid, indices in tile_groups.items():
        pairs: set[tuple[str, int]] = set()
        for i in indices:
            sensor = scene_sensors[i]
            present = np.nonzero(class_counts[i])[0]
            pairs.update((sensor, int(c)) for c in present)
        tile_pairs[tid] = pairs

    all_pairs = sorted({pair for pairs in tile_pairs.values() for pair in pairs})

    def _has(partition: list[str], pair: tuple[str, int]) -> bool:
        return any(pair in tile_pairs[tid] for tid in partition)

    def _move(pair: tuple[str, int], donor: list[str], destination: list[str]) -> bool:
        if len(donor) <= 1:
            return False
        candidates = [tid for tid in sorted(donor) if pair in tile_pairs[tid]]
        if not candidates:
            return False

        def _cost(tid: str) -> tuple[int, int, str]:
            retained: set[tuple[str, int]] = set()
            for other in donor:
                if other != tid:
                    retained |= tile_pairs[other]
            lost = len({p for g in donor for p in tile_pairs[g]} - retained)
            return (lost, len(tile_pairs[tid]), tid)

        best = min(candidates, key=_cost)
        if _cost(best)[0] > 0 and len(candidates) == len(donor):
            return False
        donor.remove(best)
        destination.append(best)
        destination.sort()
        return True

    if val_ratio > 0:
        for pair in all_pairs:
            if not _has(val, pair):
                _move(pair, train, val)
    if test_ratio > 0:
        for pair in all_pairs:
            if not _has(test, pair):
                _move(pair, train, test)

    for pair in all_pairs:
        if not _has(train, pair):
            _move(pair, val, train) or _move(pair, test, train)

    warnings: list[str] = []
    for name, partition, enabled in (
        ("train", train, True),
        ("val", val, val_ratio > 0),
        ("test", test, test_ratio > 0),
    ):
        if not enabled:
            continue
        missing = [pair for pair in all_pairs if not _has(partition, pair)]
        warnings.extend(
            f"{name} has no {sensor} pixels of class {class_id}" for sensor, class_id in missing
        )
    return warnings


def _proportion_warnings(
    train: Sequence[str],
    val: Sequence[str],
    test: Sequence[str],
    tile_groups: Mapping[str, list[int]],
    val_ratio: float,
    test_ratio: float,
) -> list[str]:
    """Flag a split whose scene proportions drift far from the configured
    ones (not a failure -- the split stays tile-exclusive regardless).

    The split targets tile *counts*, but what a partition actually measures
    is its *scenes*, and tiles differ hugely in scene count -- one
    heavily-imaged tile can dominate a partition meant to be a small
    fraction of the corpus. That is worth flagging even though it is not
    leakage.
    """
    scene_counts = {
        "train": sum(len(tile_groups[t]) for t in train),
        "val": sum(len(tile_groups[t]) for t in val),
        "test": sum(len(tile_groups[t]) for t in test),
    }
    total = sum(scene_counts.values())
    if total == 0:
        return []

    messages: list[str] = []
    for name, target in (("val", val_ratio), ("test", test_ratio)):
        if target <= 0:
            continue
        actual = scene_counts[name] / total
        # Trigger only on a substantial drift, so ordinary rounding is quiet.
        if actual > 2 * target or actual < 0.4 * target:
            messages.append(
                f"{name} holds {actual:.1%} of scenes but was configured for "
                f"{target:.0%}; one or a few tiles with many scenes dominate "
                f"this partition, so its score is not comparable to a run with "
                f"different proportions (try another split_seed)"
            )

    # Name a single tile that dominates a held-out partition.
    for name, partition in (("val", val), ("test", test)):
        n_scenes = sum(len(tile_groups[t]) for t in partition)
        if n_scenes < 5:
            continue
        counts = {t: len(tile_groups[t]) for t in partition}
        tile_id, count = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
        share = count / n_scenes
        if share > 0.5:
            messages.append(
                f"{name} is {share:.0%} scenes from the single tile {tile_id}; "
                f"its score largely measures that one tile"
            )

    return messages


def _check_tile_exclusivity(train: Sequence[str], val: Sequence[str], test: Sequence[str]) -> None:
    """Hard invariant: no tile id may appear in more than one partition.
    Structurally this should already be guaranteed by construction (every
    move removes a tile from its donor before appending to its
    destination), but it is checked explicitly since silent violation of
    this invariant is the one failure mode that must never pass unnoticed --
    it would mean scenes of the same tile landing in two partitions, i.e.
    leakage."""
    overlap = (set(train) & set(val)) | (set(train) & set(test)) | (set(val) & set(test))
    if overlap:
        raise SplitError(
            f"tile-level split is not exclusive: {sorted(overlap)} appear in more than "
            f"one partition"
        )


def split_scenes(
    records: Sequence[SceneRecord],
    class_pixel_counts_by_scene: Mapping[str, np.ndarray],
    num_classes: int,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    split_seed: int = 0,
) -> SplitResult:
    """Split annotated scenes into train/val/test by whole tile.

    Parameters
    ----------
    records:
        Every annotated scene to split (see :func:`discover_scene_records`).
    class_pixel_counts_by_scene:
        ``{scene_key: (num_classes,) pixel counts}`` -- per-class pixel
        counts for each scene's label mask, used to drive class- and
        sensor-class-coverage repair (see :func:`collect_class_pixel_counts_by_scene`).
        ``scene_key`` is ``SceneRecord.scene_key``.
    num_classes:
        Total number of possible class ids (the class-coverage passes index
        arrays of this length).
    ratios:
        Target ``(train, val, test)`` ratios by tile count. Must sum to 1.0.
    split_seed:
        Seed for the stratified shuffle; the same seed always reproduces the
        same split for the same input records.

    Returns
    -------
    A :class:`SplitResult` with the resulting tile-to-partition assignment,
    the scenes grouped by partition, and any non-fatal warnings raised by
    the coverage-repair or proportion-drift checks.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise SplitError(f"ratios must sum to 1.0, got {ratios} (sum={sum(ratios)})")
    if not records:
        raise SplitError("No scene records to split.")

    _train_ratio, val_ratio, test_ratio = ratios
    records = sorted(records, key=lambda r: r.sort_key)
    scene_sensors = [r.sensor for r in records]
    empty_counts = np.zeros(num_classes, dtype=np.int64)
    class_counts = [
        np.asarray(class_pixel_counts_by_scene.get(r.scene_key, empty_counts), dtype=np.int64)
        for r in records
    ]

    tile_groups = group_scenes_by_tile(records)
    assignment = _stratify_by_size(tile_groups, val_ratio, test_ratio, split_seed)
    if assignment is None:
        raise SplitError("Could not build a tile-level split (no tiles?).")

    train, val, test = (list(part) for part in assignment)

    _enforce_sensor_coverage(tile_groups, scene_sensors, train, val, test, val_ratio, test_ratio)
    _enforce_class_coverage(
        tile_groups, class_counts, num_classes, train, val, test, val_ratio, test_ratio
    )
    warnings = _enforce_sensor_class_coverage(
        tile_groups, class_counts, scene_sensors, num_classes, train, val, test, val_ratio, test_ratio
    )

    _check_tile_exclusivity(train, val, test)
    warnings = warnings + _proportion_warnings(train, val, test, tile_groups, val_ratio, test_ratio)
    for warning in warnings:
        logger.warning("[split] %s", warning)

    tile_partition: dict[str, str] = {}
    for tid in train:
        tile_partition[tid] = "train"
    for tid in val:
        tile_partition[tid] = "val"
    for tid in test:
        tile_partition[tid] = "test"

    scenes_by_partition: dict[str, list[SceneRecord]] = {p: [] for p in PARTITIONS}
    for record in records:
        scenes_by_partition[tile_partition[record.tile_id]].append(record)
    for partition in scenes_by_partition:
        scenes_by_partition[partition].sort(key=lambda r: r.sort_key)

    return SplitResult(
        tile_partition=tile_partition, scenes_by_partition=scenes_by_partition, warnings=warnings
    )


# -- normalisation and counts (train-only) ---------------------------------


def compute_mean_std(train_records: Sequence[SceneRecord]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over every train-partition scene's feature
    stack, computed **train-only** -- val/test scenes must never be passed
    here, since leaking their statistics into normalization would let
    held-out information influence every training example.

    Accumulates sum and sum-of-squares per scene (never concatenating the
    whole corpus into memory at once).

    NaN-safe: a feature stack legitimately carries NaN at its edges (e.g. a
    resampled DEM layer's reprojection halo, see
    ``features.training_cache._resample_dem_to_sensor_grid``) or from a
    handful of degenerate pixels in a spectral-index ratio (division by a
    near-zero denominator). Every accumulation here excludes non-finite
    values and tracks **per-channel** valid counts -- a single stray NaN
    anywhere in the corpus must never propagate into every channel's
    mean/std (a plain unconditional ``.sum()`` would turn one NaN pixel into
    a fully-NaN result for that channel, and previously did exactly that).
    """
    if not train_records:
        raise SplitError("compute_mean_std requires at least one train scene.")

    n_channels = None
    total = None
    total_sq = None
    count = None

    for record in sorted(train_records, key=lambda r: r.sort_key):
        cache = load_scene_cache(record.cache_dir)
        features = cache["features"].astype(np.float64)
        c = features.shape[0]
        if n_channels is None:
            n_channels = c
            total = np.zeros(c, dtype=np.float64)
            total_sq = np.zeros(c, dtype=np.float64)
            count = np.zeros(c, dtype=np.int64)
        elif c != n_channels:
            raise SplitError(
                f"Scene '{record.scene_key}' has {c} feature channels, expected {n_channels} "
                f"(every scene must share the same feature stack)."
            )

        flat = features.reshape(c, -1)
        finite = np.isfinite(flat)
        safe = np.where(finite, flat, 0.0)
        total += safe.sum(axis=1)
        total_sq += np.square(safe).sum(axis=1)
        count += finite.sum(axis=1)

    if not np.any(count > 0):
        raise SplitError("no finite feature values available for normalisation")

    safe_count = np.maximum(count, 1)
    mean = total / safe_count
    variance = np.maximum(total_sq / safe_count - np.square(mean), 0.0)
    # Sample (n-1) convention, guarding single-pixel corpora, per channel.
    correction = safe_count / np.maximum(safe_count - 1, 1)
    std = np.sqrt(np.maximum(variance * correction, 1e-8))

    return mean.astype(np.float32), std.astype(np.float32)


def compute_class_counts(train_records: Sequence[SceneRecord], num_classes: int) -> np.ndarray:
    """Per-class pixel counts over every train-partition scene, **train
    only**. Returns an ``(num_classes,)`` int64 array indexed by class id."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for record in sorted(train_records, key=lambda r: r.sort_key):
        cache = load_scene_cache(record.cache_dir)
        counts += scene_class_counts(cache["labels"].astype(np.int64), num_classes)
    return counts


def compute_class_weights(class_counts: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Inverse-frequency class weights from train-only pixel counts:
    ``total / (n_classes * (count + eps))``, normalized so the mean weight
    is 1.0. ``eps`` avoids a division by zero for a class with zero train
    pixels (which class-coverage repair tries to prevent, but a class can
    still appear in a tile's mask with only a handful of pixels)."""
    counts = class_counts.astype(np.float64)
    n_classes = counts.shape[0]
    total = counts.sum()
    if total <= 0:
        raise SplitError("compute_class_weights requires at least one labeled pixel.")

    weights = total / (n_classes * (counts + eps))
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def collect_class_pixel_counts_by_scene(
    records: Sequence[SceneRecord], num_classes: int
) -> dict[str, np.ndarray]:
    """Load every scene's cached labels and return
    ``{scene_key: (num_classes,) pixel counts}``, used to drive
    :func:`split_scenes`'s class- and sensor-class-coverage repair passes.
    Loads each ``.npz`` once, in sorted order."""
    result: dict[str, np.ndarray] = {}
    for record in sorted(records, key=lambda r: r.sort_key):
        cache = load_scene_cache(record.cache_dir)
        result[record.scene_key] = scene_class_counts(cache["labels"].astype(np.int64), num_classes)
    return result
