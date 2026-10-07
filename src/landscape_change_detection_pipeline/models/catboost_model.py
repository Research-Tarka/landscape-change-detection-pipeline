"""Pixel-level CatBoost classifier for 14-class land-cover classification.

This module is new, written to the same
pixel-as-row shape as
:mod:`landscape_change_detection_pipeline.models.random_forest`: every valid (non-ignore)
pixel across every training scene becomes one row, its feature-channel
values the columns, its label the target. Standard tabular-ML framing for
per-pixel spectral features, and the same framing both tree-based baselines
in this project share.

GPU by default
---------------
Unlike Random Forest (no viable GPU path on Windows, hence CPU-only),
CatBoost's GPU support ships in the standard PyPI wheel -- ``task_type="GPU"``
needs no special build, just a CUDA-capable GPU and driver at fit time (see
``requirements-torch.txt`` for the one-line install note). This project's
actual training runs have a GPU available, so :class:`CatBoostModel` defaults
to ``task_type="GPU"`` and deliberately does **not** catch a GPU-unavailable
error and fall back to CPU: a silent fallback would hide a real
misconfiguration (wrong driver, no CUDA, wrong catboost build) behind a
successful-looking CPU run that silently ignores the config's intent. If GPU
training is not possible in a given environment, the caller must pass
``task_type="CPU"`` explicitly (as this module's own unit tests do, since the
sandbox they run in may have no GPU).
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
from catboost import CatBoostClassifier

from landscape_change_detection_pipeline.features.training_cache import load_scene_cache
from landscape_change_detection_pipeline.training.losses import IGNORE_INDEX, compute_class_weights

__all__ = ["CatBoostModel", "build_catboost", "search_catboost"]

logger = logging.getLogger(__name__)


def _scene_rows(record) -> tuple[np.ndarray, np.ndarray]:
    """Load one scene's cache and flatten it to ``(N, C)`` features / ``(N,)``
    labels, dropping every pixel whose label equals :data:`IGNORE_INDEX`."""
    cache = load_scene_cache(record.cache_dir, label_remap=record.label_remap)
    features = cache["features"].astype(np.float32)  # (C, H, W)
    labels = cache["labels"].astype(np.int64)  # (H, W)

    c = features.shape[0]
    flat_x = features.reshape(c, -1).T  # (H*W, C)
    flat_y = labels.reshape(-1)  # (H*W,)

    valid = flat_y != IGNORE_INDEX
    return flat_x[valid], flat_y[valid]


def _stack_records(records: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate :func:`_scene_rows` across every record, in the order
    given, into one ``(N, C)`` / ``(N,)`` pair."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for record in records:
        x, y = _scene_rows(record)
        if x.shape[0] == 0:
            continue
        xs.append(x)
        ys.append(y)
    if not xs:
        raise ValueError("No valid (non-ignore-index) pixels found across the given records.")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


class CatBoostModel:
    """Pixel-level multiclass CatBoost classifier over per-scene feature
    caches (see :mod:`landscape_change_detection_pipeline.features.training_cache`).

    Wraps ``catboost.CatBoostClassifier`` with ``loss_function="MultiClass"``
    and a fixed ``classes_count=num_classes``, so the model's output space
    always spans every configured class even when a particular training run
    never sees pixels of one of them (see :meth:`predict_proba`).

    ``task_type="GPU"`` is the default (see module docstring): if no
    CUDA-capable GPU is available at :meth:`fit` time, CatBoost itself raises
    -- this is intentional and is not caught here. Pass ``task_type="CPU"``
    explicitly to train without a GPU.
    """

    def __init__(
        self,
        num_classes: int,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        depth: int = 8,
        task_type: str = "GPU",
        random_state: int = 0,
        early_stopping_rounds: int = 50,
        class_weighting: bool = True,
    ) -> None:
        self.num_classes = num_classes
        self.early_stopping_rounds = early_stopping_rounds
        self.class_weighting = class_weighting
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            task_type=task_type,
            random_state=random_state,
            loss_function="MultiClass",
            classes_count=num_classes,
            verbose=100,
        )

    def fit(
        self,
        train_records: Sequence,
        eval_records: Optional[Sequence] = None,
        callbacks: Optional[list] = None,
        init_model: Optional[CatBoostClassifier] = None,
    ) -> "CatBoostModel":
        """Fit on every valid pixel across ``train_records``.

        ``early_stopping_rounds`` is only meaningful when ``eval_records`` is
        given -- CatBoost's early stopping monitors an eval set's metric, and
        is a no-op without one (training simply runs the full ``iterations``
        instead).

        ``callbacks``, when given, is passed straight through to CatBoost's
        own ``fit(callbacks=...)`` -- the injection point
        :func:`search_catboost` uses for its Optuna pruning callback on CPU
        (a callback that itself needs ``eval_set`` to have something to read
        each round, so it is only meaningful together with ``eval_records``).
        CatBoost rejects ``callbacks`` outright on GPU, and also rejects
        ``init_model`` (continuing an already-fitted booster) on GPU -- see
        :class:`_CatBoostPruningCallback`'s docstring for both, confirmed
        live. There is therefore no GPU-compatible way to prune a trial
        partway through at all (not just no ``callbacks=`` alternative).

        ``init_model``, when given, is passed through to CatBoost's own
        ``fit(init_model=...)`` -- continues boosting an already-fitted
        model instead of training from scratch. Not currently used by
        anything in this module (CPU pruning uses ``callbacks`` instead),
        exposed here only because CatBoost's own ``fit()`` supports it.
        """
        x_train, y_train = _stack_records(train_records)

        if self.class_weighting:
            counts = np.bincount(y_train, minlength=self.num_classes)
            weights = compute_class_weights({i: int(c) for i, c in enumerate(counts)}, self.num_classes)
            self.model.set_params(class_weights=weights)

        fit_kwargs = {}
        if eval_records is not None:
            x_val, y_val = _stack_records(eval_records)
            fit_kwargs["eval_set"] = (x_val, y_val)
            fit_kwargs["early_stopping_rounds"] = self.early_stopping_rounds
        if callbacks:
            fit_kwargs["callbacks"] = callbacks
        if init_model is not None:
            fit_kwargs["init_model"] = init_model

        self.model.fit(x_train, y_train, **fit_kwargs)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """``(C, H, W)`` features -> ``(H, W)`` predicted class ids.

        CatBoost's ``.predict()`` for ``MultiClass`` returns an ``(N, 1)``
        array (not ``(N,)``), so the result is flattened before reshaping
        back to the image grid.
        """
        c, h, w = features.shape
        flat_x = features.reshape(c, -1).T.astype(np.float32)  # (H*W, C)
        pred = self.model.predict(flat_x)
        pred = np.asarray(pred).reshape(-1).astype(np.int64)
        return pred.reshape(h, w)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """``(C, H, W)`` features -> ``(num_classes, H, W)`` class
        probabilities.

        CatBoost's own ``classes_`` (populated after fit) only lists classes
        that were fit with ``classes_count``, which should already be every
        class 0..num_classes-1 since the constructor always passes
        ``classes_count=num_classes``. Columns are nonetheless remapped
        explicitly by ``classes_`` rather than assumed positional, so a
        class absent from the training pixels still gets a
        ``(H, W)`` slice of zeros at its correct index instead of the
        output silently being narrower than ``num_classes``.
        """
        c, h, w = features.shape
        flat_x = features.reshape(c, -1).T.astype(np.float32)  # (H*W, C)
        # verbose=False: without it CatBoost logs internal diagnostics (e.g.
        # "Object info sizes: ...") to stdout on every call -- harmless but
        # drowns out this script's own per-scene progress output, especially
        # with many threads/scenes calling predict_proba concurrently.
        proba = self.model.predict_proba(flat_x, verbose=False)  # (H*W, K) K<=num_classes

        full = np.zeros((flat_x.shape[0], self.num_classes), dtype=np.float32)
        model_classes = np.asarray(self.model.classes_).reshape(-1).astype(np.int64)
        for col, class_id in enumerate(model_classes):
            if 0 <= class_id < self.num_classes:
                full[:, class_id] = proba[:, col]

        return full.T.reshape(self.num_classes, h, w)


class _CatBoostPruningCallback:
    """CatBoost ``fit(callbacks=[...])`` hook that reports the eval-set
    ``MultiClass`` loss to an Optuna trial after every boosting round, and
    signals a stop the moment the trial's pruner says to.

    **CPU only.** CatBoost rejects any user-defined ``callbacks=`` outright
    on GPU (``CatBoostError: User defined callbacks are not supported for
    GPU``, confirmed live). The natural GPU-compatible workaround --
    fitting in slices and resuming each one via ``init_model=`` -- is also
    a dead end: CatBoost raises ``CatBoostError: Training continuation for
    GPU is not yet supported`` (confirmed live too) the moment ``fit()`` is
    called with ``init_model`` under ``task_type="GPU"``. There is
    therefore no way to prune a CatBoost GPU trial partway through its
    ``iterations`` -- :func:`search_catboost` logs a warning and runs every
    trial to completion unpruned whenever ``task_type="GPU"``, regardless
    of ``pruner``.

    Written against CatBoost's own callback protocol (an object with
    ``after_iteration(info) -> bool``, ``True`` meaning "keep training") --
    not ``optuna_integration.CatBoostPruningCallback``, which is a separate
    package this project does not otherwise depend on (see
    ``requirements-torch.txt``, only plain ``optuna`` is listed).

    ``should_prune`` is left as a plain ``bool`` flag rather than raising
    ``optuna.TrialPruned`` directly: CatBoost's Cython training loop calls
    this callback through its own C++ layer and wraps *any* exception
    raised inside ``after_iteration`` (confirmed live: raising
    ``optuna.TrialPruned`` here surfaces at the ``.fit()`` call site as a
    ``catboost.CatBoostError``, not the original ``TrialPruned``, which
    Optuna's ``study.optimize`` then treats as a genuine failed trial
    instead of a pruned one). Returning ``False`` stops training cleanly
    instead, and :func:`search_catboost`'s ``objective`` raises
    ``optuna.TrialPruned`` itself afterward, once back in plain Python.
    """

    def __init__(self, trial: "optuna.trial.Trial", metric: str = "MultiClass") -> None:
        self._trial = trial
        self._metric = metric
        self.pruned = False

    def after_iteration(self, info) -> bool:
        values = info.metrics.get("validation", {}).get(self._metric)
        if not values:
            return True  # no eval_set metric recorded yet -- nothing to report
        self._trial.report(float(values[-1]), step=info.iteration)
        if self._trial.should_prune():
            self.pruned = True
            return False  # stop training; objective() raises TrialPruned itself
        return True


def search_catboost(
    train_records: Sequence,
    val_records: Sequence,
    num_classes: int,
    n_trials: int = 30,
    sampler: str = "tpe",
    pruner: str = "none",
    timeout_s: Optional[int] = None,
    storage: Optional[str] = None,
    task_type: str = "GPU",
    random_state: int = 0,
    seed: int = 0,
    class_weighting: bool = True,
    search_iterations: int = 500,
    search_early_stopping_rounds: int = 30,
    final_iterations: int = 2000,
    final_early_stopping_rounds: int = 50,
) -> CatBoostModel:
    """Tune (``learning_rate``, ``depth``, ``l2_leaf_reg``) with Optuna,
    each trial fit on ``train_records`` and scored by macro mIoU (pooled
    confusion matrix, mirrors :mod:`.threshold`'s own search) on
    ``val_records``. The winning trial is refit once more on
    ``train_records`` -- at the real, full ``final_iterations`` budget
    (with early stopping against ``val_records``) -- and returned.

    Every trial during the search itself is capped at ``search_iterations``
    (with ``search_early_stopping_rounds``), deliberately lower than
    ``final_iterations``: the point of the search is to compare many
    candidate (learning_rate, depth, l2_leaf_reg) combinations cheaply, not
    to fully train each one. That ``search_iterations`` budget is also
    Hyperband's resource axis when ``pruner="hyperband"``: each trial
    reports its eval-set loss and can be pruned before its
    ``search_iterations`` rounds (or its own early stopping) finish -- via
    a per-round ``callbacks=`` hook (:class:`_CatBoostPruningCallback`).
    This only works with ``task_type="CPU"``: CatBoost supports neither
    user-defined callbacks nor resumable/incremental fitting on GPU (both
    confirmed live -- see :class:`_CatBoostPruningCallback`'s docstring), so
    there is no way to prune a trial partway through on GPU at all; with
    ``task_type="GPU"`` every trial always runs to completion regardless of
    ``pruner``, and a warning is logged once up front when that silently
    downgrades a requested ``pruner``.
    """
    import optuna

    from landscape_change_detection_pipeline.models.threshold import _macro_miou, _pooled_confusion
    from landscape_change_detection_pipeline.training.optuna_utils import build_pruner, build_sampler

    if not val_records:
        raise ValueError("search_catboost needs val_records to score trials against")

    pruning_supported = str(task_type).strip().upper() != "GPU"
    if pruner != "none" and not pruning_supported:
        logger.warning(
            "[catboost] pruner=%r requested but task_type=%r supports neither pruning "
            "callbacks nor resumable fitting on GPU (both are hard CatBoost limitations, "
            "confirmed live) -- every trial will run to completion unpruned. Use "
            "task_type='CPU' to actually prune trials.",
            pruner, task_type,
        )

    val_scenes = []
    for record in val_records:
        cache = load_scene_cache(record.cache_dir, label_remap=record.label_remap)
        val_scenes.append((cache["features"].astype(np.float32), cache["labels"].astype(np.int64)))

    def objective(trial: "optuna.trial.Trial") -> float:
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        depth = trial.suggest_int("depth", 4, 10)
        l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True)

        model = CatBoostModel(
            num_classes=num_classes,
            iterations=search_iterations,
            learning_rate=learning_rate,
            depth=depth,
            task_type=task_type,
            random_state=random_state,
            early_stopping_rounds=search_early_stopping_rounds,
            class_weighting=class_weighting,
        )
        model.model.set_params(l2_leaf_reg=l2_leaf_reg)
        pruning_cb = _CatBoostPruningCallback(trial) if (pruner != "none" and pruning_supported) else None
        model.fit(train_records, eval_records=val_records, callbacks=[pruning_cb] if pruning_cb else None)
        if pruning_cb is not None and pruning_cb.pruned:
            raise optuna.TrialPruned()

        confusion = _pooled_confusion(val_scenes, model, num_classes)
        score = _macro_miou(confusion)
        return score if np.isfinite(score) else -1.0

    study = optuna.create_study(
        direction="maximize",
        sampler=build_sampler(sampler, seed=seed),
        pruner=build_pruner(pruner, max_resource=search_iterations),
        storage=storage,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s)

    best_params = study.best_trial.params
    logger.info("[catboost] search_catboost best trial #%d: macro mIoU=%.4f, params=%s",
                study.best_trial.number, study.best_value, best_params)

    model = CatBoostModel(
        num_classes=num_classes,
        iterations=final_iterations,
        learning_rate=best_params["learning_rate"],
        depth=best_params["depth"],
        task_type=task_type,
        random_state=random_state,
        early_stopping_rounds=final_early_stopping_rounds,
        class_weighting=class_weighting,
    )
    model.model.set_params(l2_leaf_reg=best_params["l2_leaf_reg"])
    model.fit(train_records, eval_records=val_records)
    from landscape_change_detection_pipeline.training.optuna_utils import attach_hpo_summary

    attach_hpo_summary(model, study)
    return model


def build_catboost(
    num_classes: int,
    iterations: int = 1000,
    learning_rate: float = 0.05,
    depth: int = 8,
    task_type: str = "GPU",
    random_state: int = 0,
    early_stopping_rounds: int = 50,
    class_weighting: bool = True,
    train_records: Optional[Sequence] = None,
    val_records: Optional[Sequence] = None,
    **_ignored,
) -> CatBoostModel:
    """Construct (and, if data is given, fit) a :class:`CatBoostModel`.

    Called by :mod:`landscape_change_detection_pipeline.models.registry`'s ``model.type
    == "catboost"`` branch, with ``**_ignored`` absorbing any config fields
    not relevant here so a registry-wide kwarg does not break this
    constructor.

    If ``train_records`` is ``None``, returns an unfitted model without
    crashing -- the config-driven registry builds a model instance
    unconditionally, whether or not this particular run trains one from
    scratch. Calling ``.predict``/``.predict_proba`` on an unfitted model
    raises CatBoost's own "model is not fitted" error, which is expected and
    is not worked around here.
    """
    model = CatBoostModel(
        num_classes=num_classes,
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        task_type=task_type,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
        class_weighting=class_weighting,
    )
    if train_records is not None:
        model.fit(train_records, eval_records=val_records)
    return model
