"""Shared Optuna sampler/pruner construction, reused by every Optuna search
in this project: the torch-model HPO (:mod:`.hpo`) and the tree-based
searches (:func:`landscape_change_detection_pipeline.models.catboost_model.search_catboost`,
:func:`landscape_change_detection_pipeline.models.lightgbm_model.search_lightgbm`).

Factored out from :mod:`.hpo` (which used to define these inline, while
``catboost_model.py``/``lightgbm_model.py`` each duplicated their own
sampler if/elif) so the three call sites build the same sampler/pruner from
the same config strings, instead of three copies drifting apart.
"""

from __future__ import annotations

from typing import Any, Optional

__all__ = ["build_sampler", "build_pruner"]


def build_sampler(name: str, seed: Optional[int] = None) -> Any:
    """Build an Optuna sampler. TPE is the default: it models the density of
    good and bad configurations separately, which suits a small budget over
    a handful of continuous hyperparameters better than random search.

    ``seed`` is accepted (and ignored by the torch HPO call site, which
    leaves sampler seeding to Optuna's own default) so the tree-based
    searches, which have always seeded their sampler for reproducibility,
    keep doing so through this shared function.
    """
    import optuna

    key = str(name or "tpe").strip().lower()
    if key == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    if key == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if key != "tpe":
        raise ValueError(f"unknown sampler {name!r} (expected 'tpe', 'random', or 'cmaes')")
    return optuna.samplers.TPESampler(seed=seed)


def build_pruner(name: str, max_resource: Optional[int] = None) -> Any:
    """Build an Optuna pruner. ``none`` is the reference setting: with only
    a handful of trials, pruning risks discarding a configuration that
    starts slow and finishes well.

    ``max_resource`` is Hyperband's resource axis upper bound (epochs for
    the torch HPO, boosting rounds for catboost/lightgbm) -- ``"auto"`` when
    not given, letting Optuna infer it from observed trials instead.
    """
    import optuna

    key = str(name or "none").strip().lower()
    if key == "median":
        return optuna.pruners.MedianPruner()
    if key == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=max(1, int(max_resource)) if max_resource else "auto"
        )
    return optuna.pruners.NopPruner()


def study_summary(study) -> dict:
    """JSON-safe summary of a finished Optuna study (every trial's params,
    score and state, plus the best one), for the run report."""
    return {
        "best_trial": study.best_trial.number,
        "best_value": float(study.best_value),
        "best_params": dict(study.best_trial.params),
        "trials": [
            {"trial_id": t.number, "params": dict(t.params), "score": t.value, "state": t.state.name}
            for t in study.trials
        ],
    }


def attach_hpo_summary(model, study) -> None:
    """Attach ``study_summary(study)`` to ``model`` as ``hpo_summary`` (best-effort)."""
    try:
        model.hpo_summary = study_summary(study)
    except Exception:  # noqa: BLE001 -- a report detail must never break training
        pass
