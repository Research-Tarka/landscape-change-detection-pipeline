"""Pixel-level LightGBM classifier for 14-class land-cover classification.

Same pixel-as-row shape as :mod:`.random_forest`/:mod:`.catboost_model`:
every valid (non-ignore) pixel across every training scene becomes one row,
its feature-channel values the columns, its label the target.

Why LightGBM over ``RandomForestClassifier`` for this project's RAM budget
-----------------------------------------------------------------------------
:mod:`.random_forest`'s own docstring already documents its memory
tradeoff: sklearn's ``RandomForestClassifier`` has no incremental/partial
fit, so the entire ``(N_valid_pixels, C)`` design matrix must be resident in
memory for the whole ``.fit()`` call, *and* ``n_estimators`` (300 by
default) trees' worth of bootstrap-sample/node-split bookkeeping must
coexist during ensemble construction (more so with ``n_jobs>1``, where
several trees build concurrently). Confirmed live: this saturates 32 GB of
RAM on this project's real training corpus.

LightGBM's ``lgb.Dataset`` sidesteps this differently: it histogram-bins
every feature (up to 255 bins, ``max_bin``) into its own compact internal
representation as part of construction, and with ``free_raw_data=True``
(LightGBM's own default) it does not need to keep the original dense
float32 array alive afterward -- only the binned copy. This module still
assembles one dense ``(N, C)`` float32 array first (the same two-pass
pre-allocation pattern :mod:`.random_forest` already uses, to avoid holding
two copies via list-append + concatenate), but then explicitly ``del``s it
and calls ``gc.collect()`` immediately after ``Dataset`` construction --
so the dense corpus and LightGBM's binned copy are never both resident for
longer than the construction call itself. This is simpler than a
generator/chunked ``Dataset`` build and captures most of the benefit: the
binned representation is materially smaller than float32, and there is no
per-tree bootstrap-sample bookkeeping to begin with (gradient boosting
builds one tree at a time, sequentially, unlike a random forest's
independent trees).

Uses the native ``lgb.train``/``lgb.Dataset`` API, not the sklearn-style
``LGBMClassifier`` wrapper -- the native API gives direct control over the
``Dataset``-then-free memory pattern above and over ``valid_sets``-based
early stopping, matching this project's own "don't hide the memory-relevant
mechanics behind a wrapper" spirit for this module specifically.

GPU: unverified on this project's machine, default CPU
--------------------------------------------------------
Confirmed live, 2026-09-25, on this project's own environment (Windows,
RTX 5070 Laptop GPU, ``pip install lightgbm`` standard PyPI wheel,
version 4.7.0): both ``device="gpu"`` and ``device="cuda"`` fail outright
at train time with ``LightGBMError: GPU/CUDA Tree Learner was not enabled
in this build`` -- the standard PyPI wheel is CPU-only; GPU/CUDA support
needs a from-source build (``cmake -DUSE_GPU=1`` or ``-DUSE_CUDA=1``,
historically also needing an OpenCL ICD + Boost toolchain for the GPU
path), which this project does not currently do. Unlike
:mod:`.catboost_model` (which defaults to GPU because that was already
confirmed working in this project's environment), this module therefore
defaults to ``device="cpu"``. If a from-source GPU/CUDA build is set up
later, ``device`` can be switched to ``"gpu"``/``"cuda"`` via config -- but
do not assume it works without re-confirming live first, per this
project's own live-verification convention (see e.g.
``docs/decisions/topographic_correction.md``'s worked example).
"""

from __future__ import annotations

import gc
import logging
from typing import Optional, Sequence

import lightgbm as lgb
import numpy as np

from landscape_change_detection_pipeline.features.training_cache import load_scene_cache
from landscape_change_detection_pipeline.training.losses import IGNORE_INDEX, compute_class_weights

__all__ = ["LightGBMModel", "build_lightgbm", "search_lightgbm"]

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


class LightGBMModel:
    """Pixel-level multiclass LightGBM classifier over per-scene feature
    caches (see :mod:`landscape_change_detection_pipeline.features.training_cache`).

    Wraps native ``lgb.Dataset``/``lgb.train`` with
    ``objective="multiclass"`` and a fixed ``num_class=num_classes``, so the
    model's output space always spans every configured class even when a
    particular training run never sees pixels of one of them (see
    :meth:`predict_proba`). See the module docstring for why this uses the
    native API (not ``LGBMClassifier``) and why ``device`` defaults to CPU.
    """

    def __init__(
        self,
        num_classes: int,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        n_estimators: int = 1000,
        max_depth: int = -1,
        min_data_in_leaf: int = 20,
        device: str = "cpu",
        random_state: int = 0,
        early_stopping_rounds: int = 50,
        class_weighting: bool = True,
    ) -> None:
        self.num_classes = num_classes
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.class_weighting = class_weighting
        self.params = {
            "objective": "multiclass",
            "num_class": num_classes,
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_data_in_leaf": min_data_in_leaf,
            "device": device,
            "seed": random_state,
            "verbose": -1,
        }
        self.booster: Optional[lgb.Booster] = None

    def fit(
        self,
        train_records: Sequence,
        eval_records: Optional[Sequence] = None,
        extra_callbacks: Optional[list] = None,
    ) -> "LightGBMModel":
        """Fit on every valid pixel across ``train_records``.

        ``early_stopping_rounds`` is only meaningful when ``eval_records`` is
        given -- without one, training simply runs the full
        ``n_estimators`` rounds.

        ``extra_callbacks``, when given, is appended to the callback list
        passed to ``lgb.train`` alongside early stopping -- the injection
        point :func:`search_lightgbm` uses for its Optuna pruning callback.

        See the module docstring for the memory-management pattern here:
        the dense ``(x_train, y_train)`` arrays are dropped (``del`` +
        ``gc.collect()``) as soon as ``lgb.Dataset`` has consumed them,
        before ``lgb.train`` itself runs.
        """
        x_train, y_train = _stack_records(train_records)
        sample_weight = None
        if self.class_weighting:
            counts = np.bincount(y_train, minlength=self.num_classes)
            class_weight = np.asarray(
                compute_class_weights({i: int(c) for i, c in enumerate(counts)}, self.num_classes), dtype=np.float32
            )
            sample_weight = class_weight[y_train]
        train_set = lgb.Dataset(x_train, label=y_train, weight=sample_weight, free_raw_data=True)
        del x_train, y_train, sample_weight
        gc.collect()

        valid_sets = None
        callbacks = []
        if eval_records is not None:
            x_val, y_val = _stack_records(eval_records)
            valid_set = lgb.Dataset(x_val, label=y_val, reference=train_set, free_raw_data=True)
            del x_val, y_val
            gc.collect()
            valid_sets = [valid_set]
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        self.booster = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            callbacks=callbacks or None,
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """``(C, H, W)`` features -> ``(H, W)`` predicted class ids."""
        proba = self.predict_proba(features)  # (num_classes, H, W)
        return np.argmax(proba, axis=0).astype(np.int64)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """``(C, H, W)`` features -> ``(num_classes, H, W)`` class
        probabilities.

        LightGBM's multiclass ``Booster.predict`` always returns a full
        ``(N, num_class)`` array spanning every configured class (unlike
        sklearn's ``RandomForestClassifier``/CatBoost's own
        ``predict_proba``, neither of which do -- see those modules'
        docstrings), since ``num_class`` is fixed at construction time
        rather than inferred from the classes seen during fit. No
        column-scatter defense is therefore needed here, but the output is
        still reshaped to this project's ``(num_classes, H, W)`` convention.
        """
        if self.booster is None:
            raise RuntimeError("LightGBMModel.predict_proba called before fit().")
        c, h, w = features.shape
        flat_x = features.reshape(c, -1).T.astype(np.float32)  # (H*W, C)
        proba = self.booster.predict(flat_x)  # (H*W, num_classes)
        return np.moveaxis(proba.reshape(h, w, self.num_classes), -1, 0).astype(np.float32)


def _lightgbm_pruning_callback(trial: "optuna.trial.Trial"):
    """LightGBM ``callbacks=[...]`` hook (native ``CallbackEnv`` protocol,
    not ``optuna_integration.LightGBMPruningCallback`` -- see
    :class:`landscape_change_detection_pipeline.models.catboost_model._CatBoostPruningCallback`'s
    docstring for why this project builds its own instead of depending on
    the separate ``optuna_integration`` package).

    Reports the first ``valid_sets`` entry's first metric after every
    boosting round and raises ``optuna.TrialPruned`` once the trial's
    pruner says to stop.
    """
    import optuna

    def _callback(env) -> None:
        if not env.evaluation_result_list:
            return
        _data_name, _eval_name, value, _is_higher_better = env.evaluation_result_list[0]
        trial.report(float(value), step=env.iteration)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return _callback


def search_lightgbm(
    train_records: Sequence,
    val_records: Sequence,
    num_classes: int,
    n_trials: int = 30,
    sampler: str = "tpe",
    pruner: str = "none",
    timeout_s: Optional[int] = None,
    storage: Optional[str] = None,
    device: str = "cpu",
    random_state: int = 0,
    seed: int = 0,
    class_weighting: bool = True,
    search_n_estimators: int = 500,
    search_early_stopping_rounds: int = 30,
    final_n_estimators: int = 2000,
    final_early_stopping_rounds: int = 50,
) -> LightGBMModel:
    """Tune (``learning_rate``, ``num_leaves``, ``min_data_in_leaf``) with
    Optuna, each trial fit on ``train_records`` and scored by macro mIoU
    (pooled confusion matrix, mirrors :mod:`.threshold`'s own search) on
    ``val_records``. The winning trial is refit once more -- at the real,
    full ``final_n_estimators`` budget -- and returned.

    Every trial during the search itself is capped at
    ``search_n_estimators`` (with ``search_early_stopping_rounds``),
    deliberately lower than ``final_n_estimators``: the point of the search
    is to compare many candidate (learning_rate, num_leaves,
    min_data_in_leaf) combinations cheaply, not to fully train each one.
    That ``search_n_estimators`` budget is also Hyperband's resource axis
    when ``pruner="hyperband"``: each trial reports its eval-set metric
    every round (see :func:`_lightgbm_pruning_callback`) so a trial doing
    poorly early can be cut well before its ``search_n_estimators`` rounds
    (or its own early stopping) finish.
    """
    import optuna

    from landscape_change_detection_pipeline.models.threshold import _macro_miou, _pooled_confusion
    from landscape_change_detection_pipeline.training.optuna_utils import build_pruner, build_sampler

    if not val_records:
        raise ValueError("search_lightgbm needs val_records to score trials against")

    val_scenes = []
    for record in val_records:
        cache = load_scene_cache(record.cache_dir, label_remap=record.label_remap)
        val_scenes.append((cache["features"].astype(np.float32), cache["labels"].astype(np.int64)))

    def objective(trial: "optuna.trial.Trial") -> float:
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        num_leaves = trial.suggest_int("num_leaves", 15, 255, log=True)
        min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 5, 100, log=True)

        model = LightGBMModel(
            num_classes=num_classes,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=search_n_estimators,
            max_depth=-1,
            min_data_in_leaf=min_data_in_leaf,
            device=device,
            random_state=random_state,
            early_stopping_rounds=search_early_stopping_rounds,
            class_weighting=class_weighting,
        )
        extra_callbacks = [_lightgbm_pruning_callback(trial)] if pruner != "none" else None
        model.fit(train_records, eval_records=val_records, extra_callbacks=extra_callbacks)

        confusion = _pooled_confusion(val_scenes, model, num_classes)
        score = _macro_miou(confusion)
        return score if np.isfinite(score) else -1.0

    study = optuna.create_study(
        direction="maximize",
        sampler=build_sampler(sampler, seed=seed),
        pruner=build_pruner(pruner, max_resource=search_n_estimators),
        storage=storage,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s)

    best_params = study.best_trial.params
    logger.info("[lightgbm] search_lightgbm best trial #%d: macro mIoU=%.4f, params=%s",
                study.best_trial.number, study.best_value, best_params)

    model = LightGBMModel(
        num_classes=num_classes,
        num_leaves=best_params["num_leaves"],
        learning_rate=best_params["learning_rate"],
        n_estimators=final_n_estimators,
        max_depth=-1,
        min_data_in_leaf=best_params["min_data_in_leaf"],
        device=device,
        random_state=random_state,
        early_stopping_rounds=final_early_stopping_rounds,
        class_weighting=class_weighting,
    )
    model.fit(train_records, eval_records=val_records)
    from landscape_change_detection_pipeline.training.optuna_utils import attach_hpo_summary

    attach_hpo_summary(model, study)
    return model


def build_lightgbm(
    num_classes: int,
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    n_estimators: int = 1000,
    max_depth: int = -1,
    min_data_in_leaf: int = 20,
    device: str = "cpu",
    random_state: int = 0,
    early_stopping_rounds: int = 50,
    class_weighting: bool = True,
    train_records: Optional[Sequence] = None,
    val_records: Optional[Sequence] = None,
    **_ignored,
) -> LightGBMModel:
    """Construct (and, if data is given, fit) a :class:`LightGBMModel`.

    Called by :mod:`landscape_change_detection_pipeline.models.registry`'s
    ``model.type == "lightgbm"`` branch, with ``**_ignored`` absorbing any
    config fields not relevant here so a registry-wide kwarg does not break
    this constructor.

    If ``train_records`` is ``None``, returns an unfitted model without
    crashing -- the config-driven registry builds a model instance
    unconditionally, whether or not this particular run trains one from
    scratch. Calling ``.predict``/``.predict_proba`` on an unfitted model
    raises ``RuntimeError`` (see :meth:`LightGBMModel.predict_proba`).
    """
    model = LightGBMModel(
        num_classes=num_classes,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_data_in_leaf=min_data_in_leaf,
        device=device,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
        class_weighting=class_weighting,
    )
    if train_records is not None:
        model.fit(train_records, eval_records=val_records)
    return model
