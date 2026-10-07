"""Threshold classifier: per-class spectral-index thresholds tuned by Optuna.

Design choice -- why "one rule per class in priority order", not hand-designed
rules
-----------------------------------------------------------------------------
A hand-designed decision tree over a small number of spectral indices,
tuned with a coarse-then-fine manual grid search, is tractable when there
are only a few classes and indices, where a person can reason about the
whole rule set at once.

This project has 14 classes and 11 spectral-index/chromaticity channels plus
3 DEM layers (see :mod:`landscape_change_detection_pipeline.features.spectral_indices`'s
``INDEX_NAMES``/``DEM_LAYER_NAMES``) -- coordinating which channel best
separates each class, and at what threshold, by hand is exactly the kind of
combinatorial search a person is bad at and a sampler is good at. So rather
than writing 14 bespoke, domain-guessed rules (e.g. "cloud is high in all
three visible chromaticity channels AND low NDVI AND ..."), this module fixes
a single **generic rule shape** -- one channel, one threshold, one direction,
per class -- and lets Optuna's TPE sampler decide, for every class
independently, which channel and threshold actually separate it from the
rest in the real training data. This is deliberately not a claim that a
single-channel-per-class rule is the best possible threshold classifier; it
is the simplest rule shape that (a) generalises to any ``num_classes``
without new code, (b) has a small enough per-class search space (channel x
threshold x direction) for Optuna to explore effectively in a couple hundred
trials, and (c) is easy to inspect after the fact (a config-flat list of
``(class_id, channel, threshold, direction)``). Real hand-designed
rules should follow once training runs reveal actual confusion
patterns -- this module is the config-driven scaffold those rules would slot
into (as :class:`ThresholdRule` entries), not a hand-authored ruleset itself.

Rule evaluation order and priority
-----------------------------------
Rules are evaluated in list order; the **first** rule that matches a pixel
wins (later rules never overwrite an earlier match), and a pixel matching no
rule at all falls back to ``default_class``. This means class order in the
``rules`` list is itself part of the model -- :func:`search_thresholds` fixes
it to ascending class id, which is an arbitrary but reproducible convention,
not a claim that low-id classes are semantically higher priority.

Hard assignment, not a calibrated probability
-----------------------------------------------
:meth:`ThresholdModel.predict_proba` returns a one-hot ``(num_classes, H, W)``
array (1.0 at the predicted class, 0.0 elsewhere) purely so this model can
sit behind the same "features in, per-class probability out" evaluation path
as the learned models (:mod:`landscape_change_detection_pipeline.models.unet` and
friends). A threshold rule has no notion of confidence -- there is nothing
probabilistic backing that 1.0, and it must not be read as calibrated
confidence (e.g. fed into :class:`landscape_change_detection_pipeline.training.metrics.CalibrationTally`
and expected to look reasonable).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from landscape_change_detection_pipeline.training.losses import IGNORE_INDEX

__all__ = [
    "ThresholdRule",
    "ThresholdModel",
    "search_thresholds",
    "build_threshold",
]

logger = logging.getLogger(__name__)

#: Directions a rule's comparison can take -- "above" (feature > threshold)
#: or "below" (feature < threshold).
_DIRECTIONS: tuple[str, ...] = ("above", "below")


@dataclass(frozen=True)
class ThresholdRule:
    """One per-class rule: pixel matches ``class_id`` when
    ``feature[channel_index]`` is ``above`` (``>``) or ``below`` (``<``)
    ``threshold``.

    ``channel_index`` indexes the ``(C, H, W)`` feature stack in the same
    channel order the training cache stores (see
    :func:`landscape_change_detection_pipeline.features.training_cache.build_feature_stack`
    -- spectral indices followed by DEM layers).
    """

    class_id: int
    channel_index: int
    threshold: float
    direction: str = "above"

    def __post_init__(self) -> None:
        if self.direction not in _DIRECTIONS:
            raise ValueError(f"direction must be one of {_DIRECTIONS}, got {self.direction!r}")


class ThresholdModel:
    """One-vs-rest per-class threshold classifier, evaluated in a fixed
    priority order.

    ``.predict`` applies ``rules`` in list order; the first rule whose
    condition holds for a pixel assigns that pixel's class, and a later
    rule never revisits a pixel an earlier rule already claimed. A pixel
    matching no rule gets ``default_class``. See the module docstring for
    why this shape (one channel/threshold/direction per class, in a fixed
    order) was chosen over a hand-designed decision tree.
    """

    def __init__(
        self,
        num_classes: int,
        rules: Optional[list[ThresholdRule]] = None,
        default_class: int = 0,
    ) -> None:
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if not (0 <= default_class < num_classes):
            raise ValueError(
                f"default_class must be in [0, {num_classes}), got {default_class}"
            )
        self.num_classes = int(num_classes)
        self.rules: list[ThresholdRule] = list(rules) if rules is not None else []
        self.default_class = int(default_class)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """``(C, H, W)`` features -> ``(H, W)`` predicted class ids.

        Rules are applied in list order; only pixels not yet claimed by an
        earlier rule are considered by each subsequent rule (first match
        wins). Unmatched pixels get ``default_class``.
        """
        features = np.asarray(features)
        if features.ndim != 3:
            raise ValueError(f"features must be (C, H, W), got shape {features.shape}")

        _channels, height, width = features.shape
        out = np.full((height, width), self.default_class, dtype=np.int64)
        claimed = np.zeros((height, width), dtype=bool)

        for rule in self.rules:
            channel = features[rule.channel_index]
            if rule.direction == "above":
                matches = channel > rule.threshold
            else:
                matches = channel < rule.threshold
            newly_matched = matches & ~claimed
            out[newly_matched] = rule.class_id
            claimed |= newly_matched

        return out

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """``(C, H, W)`` features -> ``(num_classes, H, W)`` one-hot "hard"
        probabilities: 1.0 at the predicted class, 0.0 elsewhere. See the
        module docstring -- this is not a calibrated probability, only a
        shape-compatible stand-in so the threshold model can share an
        evaluation path with the learned models."""
        predicted = self.predict(features)
        proba = np.zeros((self.num_classes, *predicted.shape), dtype=np.float32)
        np.put_along_axis(proba, predicted[np.newaxis, :, :], 1.0, axis=0)
        return proba


# -- pooled confusion / mIoU, numpy only (no torch import needed here) ------


def _pooled_confusion(
    scenes: Sequence[tuple[np.ndarray, np.ndarray]],
    model: ThresholdModel,
    num_classes: int,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """Accumulate one ``(num_classes, num_classes)`` confusion matrix
    (indexed ``[true, predicted]``) over every ``(features, labels)`` scene,
    pooling first the same way ``training/dataset.py``/``training/metrics.py``
    do -- one matrix over every pixel of every scene, never an average of
    per-scene matrices."""
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for features, labels in scenes:
        predicted = model.predict(features)
        valid = labels != ignore_index
        if not np.any(valid):
            continue
        true_flat = labels[valid].reshape(-1).astype(np.int64)
        pred_flat = predicted[valid].reshape(-1).astype(np.int64)
        flat_index = true_flat * num_classes + pred_flat
        counts = np.bincount(flat_index, minlength=num_classes * num_classes)
        confusion += counts[: num_classes * num_classes].reshape(num_classes, num_classes)
    return confusion


def _macro_miou(confusion: np.ndarray) -> float:
    """Unweighted mean IoU over classes present (non-empty union), computed
    directly from a pooled confusion matrix -- mirrors
    :func:`landscape_change_detection_pipeline.training.metrics.compute_confusion_metrics`'s
    ``macro.miou`` without requiring a torch import. Returns ``nan`` when the
    matrix is empty or no class has a non-empty union (mirrors
    ``weighted_cross_entropy``'s all-ignored-pixels handling in
    ``training/losses.py``: a clean, non-crashing degenerate value rather
    than a NaN propagating from a 0/0 division)."""
    conf = confusion.astype(np.float64)
    total = conf.sum()
    if total <= 0:
        return float("nan")

    tp = np.diag(conf)
    union = conf.sum(axis=0) + conf.sum(axis=1) - tp
    present = union > 0
    if not np.any(present):
        return float("nan")
    return float((tp[present] / union[present]).mean())


# -- Optuna search ------------------------------------------------------------


def _load_train_scenes(
    train_records: Sequence,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[str]]:
    """Load every train scene's cached ``(features, labels)`` once, plus the
    feature-channel names recorded in the first scene's cache. All scenes
    must share the same channel order (the training cache format's
    invariant -- see ``training/dataset.py::compute_mean_std``, which
    enforces the same thing for channel *count*)."""
    from landscape_change_detection_pipeline.features.training_cache import load_scene_cache

    scenes: list[tuple[np.ndarray, np.ndarray]] = []
    feature_names: Optional[list[str]] = None

    for record in sorted(train_records, key=lambda r: r.sort_key):
        cache = load_scene_cache(record.cache_dir, label_remap=record.label_remap)
        features = cache["features"].astype(np.float32)
        labels = cache["labels"].astype(np.int64)
        names = cache.get("feature_names")
        if feature_names is None:
            feature_names = list(names) if names is not None else [
                str(i) for i in range(features.shape[0])
            ]
        scenes.append((features, labels))

    if feature_names is None:
        feature_names = []
    return scenes, feature_names


def search_thresholds(
    train_records: Sequence,
    feature_names: Sequence[str],
    num_classes: int,
    n_trials: int = 200,
    sampler: str = "tpe",
    timeout_s: Optional[int] = None,
    storage: Optional[str] = None,
    seed: int = 0,
) -> ThresholdModel:
    """Tune one ``(channel, threshold, direction)`` rule per class with
    Optuna, evaluated against every ``train_records`` scene pooled into one
    confusion matrix per trial, maximizing macro mIoU.

    ``optuna`` is imported lazily inside this function, so importing this
    module never requires it -- only calling ``search_thresholds`` (or
    :func:`build_threshold` with ``train_records`` supplied) does.

    Args:
        train_records: scenes to tune against (``Sequence[SceneRecord]``,
            see :mod:`landscape_change_detection_pipeline.training.dataset`). Loaded once
            up front and kept in memory for the whole search.
        feature_names: channel names in the training cache's stack order
            (used only for logging/diagnostics; the search itself works by
            channel *index*, sourced per-scene from the loaded caches so a
            mismatch against a stale ``feature_names`` argument cannot skew
            the search).
        num_classes: total number of classes; one rule is searched per
            class id in ``range(num_classes)``, in that fixed order.
        n_trials: number of Optuna trials.
        sampler: ``"tpe"`` (default, ``optuna.samplers.TPESampler``) or
            ``"random"``/``"cmaes"`` (mirrors the sampler choices available
            for the torch-model HPO search).
        timeout_s: optional wall-clock budget passed to ``study.optimize``.
        storage: optional Optuna storage URL for a persisted/resumable
            study; ``None`` runs in-memory.
        seed: RNG seed for the sampler, for reproducible searches.

    Returns:
        A :class:`ThresholdModel` built from the best trial's parameters.
        If every trial fails to produce a finite score (e.g. no train scene
        has any valid pixel), an untrained default-class-only model is
        returned rather than raising.
    """
    import optuna
    from optuna.samplers import BaseSampler, CmaEsSampler, RandomSampler, TPESampler

    scenes, loaded_feature_names = _load_train_scenes(train_records)
    channel_names = list(feature_names) if feature_names else loaded_feature_names
    if not scenes:
        logger.warning("[threshold] search_thresholds got no train scenes; returning untrained model")
        return ThresholdModel(num_classes=num_classes)

    n_channels = scenes[0][0].shape[0]
    if not channel_names or len(channel_names) != n_channels:
        channel_names = [str(i) for i in range(n_channels)]

    # Per-channel observed min/max across every loaded scene, bounding each
    # trial's threshold search to values the data can actually distinguish.
    channel_min = np.full(n_channels, np.inf, dtype=np.float64)
    channel_max = np.full(n_channels, -np.inf, dtype=np.float64)
    for features, _labels in scenes:
        flat = features.reshape(n_channels, -1)
        channel_min = np.minimum(channel_min, flat.min(axis=1))
        channel_max = np.maximum(channel_max, flat.max(axis=1))
    # A channel with no spread (or no finite values, e.g. an all-nodata
    # scene) gets a degenerate-but-valid [x, x+eps] range so trial.suggest_float
    # never raises on low==high.
    degenerate = ~np.isfinite(channel_min) | ~np.isfinite(channel_max) | (channel_max <= channel_min)
    channel_min = np.where(degenerate, 0.0, channel_min)
    channel_max = np.where(degenerate, 1.0, channel_max)

    channel_indices = list(range(n_channels))

    def objective(trial: "optuna.trial.Trial") -> float:
        rules: list[ThresholdRule] = []
        for class_id in range(num_classes):
            channel_index = trial.suggest_categorical(f"class_{class_id}_channel", channel_indices)
            direction = trial.suggest_categorical(f"class_{class_id}_direction", _DIRECTIONS)
            low = float(channel_min[channel_index])
            high = float(channel_max[channel_index])
            threshold = trial.suggest_float(f"class_{class_id}_threshold", low, high)
            rules.append(
                ThresholdRule(
                    class_id=class_id,
                    channel_index=channel_index,
                    threshold=threshold,
                    direction=direction,
                )
            )

        model = ThresholdModel(num_classes=num_classes, rules=rules, default_class=0)
        confusion = _pooled_confusion(scenes, model, num_classes)
        score = _macro_miou(confusion)
        if not np.isfinite(score):
            # Mirrors weighted_cross_entropy's all-ignored-pixels handling in
            # training/losses.py: a clean degenerate value, not a NaN
            # propagating into (and crashing) the study.
            return -1.0
        return score

    sampler_name = str(sampler).strip().lower()
    optuna_sampler: BaseSampler
    if sampler_name == "tpe":
        optuna_sampler = TPESampler(seed=seed)
    elif sampler_name == "random":
        optuna_sampler = RandomSampler(seed=seed)
    elif sampler_name == "cmaes":
        optuna_sampler = CmaEsSampler(seed=seed)
    else:
        raise ValueError(f"unknown sampler {sampler!r} (expected 'tpe', 'random', or 'cmaes')")

    study = optuna.create_study(direction="maximize", sampler=optuna_sampler, storage=storage)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s)

    if study.best_trial is None or not np.isfinite(study.best_value):
        logger.warning("[threshold] search_thresholds found no finite-scoring trial; returning untrained model")
        return ThresholdModel(num_classes=num_classes)

    best_params = study.best_trial.params
    best_rules = [
        ThresholdRule(
            class_id=class_id,
            channel_index=int(best_params[f"class_{class_id}_channel"]),
            threshold=float(best_params[f"class_{class_id}_threshold"]),
            direction=str(best_params[f"class_{class_id}_direction"]),
        )
        for class_id in range(num_classes)
    ]
    logger.info(
        "[threshold] search_thresholds best trial #%d: macro mIoU=%.4f",
        study.best_trial.number,
        study.best_value,
    )
    model = ThresholdModel(num_classes=num_classes, rules=best_rules, default_class=0)
    from landscape_change_detection_pipeline.training.optuna_utils import attach_hpo_summary

    attach_hpo_summary(model, study)
    return model


def build_threshold(
    num_classes: int,
    n_trials: int = 200,
    sampler: str = "tpe",
    timeout_s: Optional[int] = None,
    study_storage: Optional[str] = None,
    train_records: Optional[Sequence] = None,
    feature_names: Optional[Sequence[str]] = None,
    seed: int = 0,
    **_ignored,
) -> ThresholdModel:
    """Registry entry point for ``model.type: threshold``
    (see :mod:`landscape_change_detection_pipeline.models.registry`).

    When ``train_records`` is ``None`` (the registry may construct a model
    before any training data is available, e.g. to validate a config before
    a training run starts), this returns an **untrained** model --
    ``ThresholdModel(num_classes=num_classes)`` with no rules and
    ``default_class=0`` -- rather than raising or running a search with
    nothing to search over. The Optuna search in :func:`search_thresholds`
    only runs once ``train_records`` is actually supplied by the caller
    (e.g. the training loop, once the training cache and split exist).

    ``**_ignored`` absorbs any registry-supplied keyword arguments this
    model type does not use (mirrors the other ``build_*`` model
    constructors' calling convention), so the registry can call every model
    type with the same superset of config fields.
    """
    if train_records is None:
        return ThresholdModel(num_classes=num_classes)

    return search_thresholds(
        train_records=train_records,
        feature_names=feature_names or [],
        num_classes=num_classes,
        n_trials=n_trials,
        sampler=sampler,
        timeout_s=timeout_s,
        storage=study_storage,
        seed=seed,
    )
