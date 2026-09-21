"""Hyperparameter optimisation with Optuna.

``run_trials`` is the entry point: with ``hpo.trials <= 0`` (or an empty
search space) it trains once at ``training``'s/``model``'s own configured
hyperparameters, appropriate once those values
are already tuned. With ``trials > 0`` it runs that many short trials
(``trial_epochs`` each, not the full ``training.epochs``) over
``hpo.search_space``, then retrains once at full length with the winning
hyperparameters.

Two aspects of the protocol are deliberate:

**Trials are short, the final run is long.** Ranking hyperparameters does
not need convergence -- it needs enough signal to tell good from bad. The
winner is then retrained at full length.

**Every trial shares the same seed** (``training.seed``, untouched by the
search). Searching across seeds would confound hyperparameter quality with
initialization luck and amount to picking the best seed, not the best
hyperparameters -- initialization variance is measured separately by
:mod:`.bootstrap`.

Pruned and failed trials are recorded rather than discarded, so a sweep
where everything got pruned still yields a best-effort answer instead of an
error.

Where a sampled hyperparameter actually lives
----------------------------------------------
``lr``/``weight_decay`` are ``TrainingConfig`` fields, but ``dropout_p`` is
not -- it lives on ``ModelConfig.<active model type>.dropout_p``
(``model.unet.dropout_p``, ``model.deeplabv3plus.dropout_p``, ...). A pydantic
``model_copy(update={...})`` silently accepts and stores an unknown field
name without validating or ever reading it back anywhere, so routing
``dropout_p`` onto ``TrainingConfig`` (as a naive single-object update
would) would make the search sample real Optuna trial values that are
never actually applied to the model -- the sweep would silently measure
nothing. :func:`_apply_overrides` routes each named hyperparameter to the
one config object that actually has that field, and raises immediately on
a name neither config recognizes, rather than letting it disappear.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from landscape_change_detection_pipeline.config import HpoConfig, ModelConfig, TrainingConfig
from landscape_change_detection_pipeline.training.dataset import SceneRecord
from landscape_change_detection_pipeline.training.metrics import canonical_main_iou_metric
from landscape_change_detection_pipeline.training.train import TrainingResult, train_model

__all__ = ["TrialRecord", "run_trials", "build_sampler", "build_pruner", "sanitise_overrides"]

#: Every hyperparameter name this module's default search_space (or a
#: config-overridden one) can legally target, and which config object +
#: field actually holds it. Extend this table, not the search space alone,
#: before naming a new hyperparameter in config -- an unlisted name fails
#: loudly in _apply_overrides rather than being silently dropped.
_TRAINING_FIELDS: tuple[str, ...] = ("lr", "weight_decay", "patch_size", "batch_size", "grad_clip_norm")
#: Model sub-config fields that live under ``model.<type>.<field>`` for
#: whichever type is currently active -- resolved against model_cfg.type at
#: override time, since e.g. "dropout_p" means a different object depending
#: on whether model.type is unet/deeplabv3plus/segformer.
_MODEL_SUBSECTION_FIELDS: tuple[str, ...] = ("dropout_p", "base_ch")

#: Integer-valued hyperparameters, floored after Optuna sampling.
_INTEGER_FLOORS: dict[str, int] = {"patch_size": 8, "batch_size": 1, "base_ch": 4}


@dataclass
class TrialRecord:
    """One trial's outcome, including the pruned and failed cases."""

    trial_id: int
    params: dict[str, Any]
    score: float
    pruned: bool = False
    failed: bool = False
    error: Optional[str] = None
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return not self.failed and math.isfinite(self.score) and self.score > -1e11


def sanitise_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """Coerce sampled hyperparameters into legal values.

    Integer-typed fields are rounded and floored; the ``stride <= patch_size``
    invariant (when both happen to be present) is restored last, so it holds
    regardless of sampling order.
    """
    out: dict[str, Any] = dict(overrides or {})
    for name, floor in _INTEGER_FLOORS.items():
        if name in out:
            try:
                out[name] = max(floor, int(round(float(out[name]))))
            except (TypeError, ValueError):
                del out[name]
    if "patch_size" in out and "stride" in out:
        out["stride"] = min(out["stride"], out["patch_size"])
    return out


def _suggest_with_optuna(trial: Any, space: dict[str, Any]) -> dict[str, Any]:
    """Ask an Optuna trial for one value per configured hyperparameter.

    Sorted iteration keeps parameter registration order stable, which
    matters for resuming a study from persistent storage.
    """
    suggested: dict[str, Any] = {}
    for name in sorted(space):
        entry = space[name]
        kind = entry.type.strip().lower()
        if not entry.bounds or len(entry.bounds) != 2:
            raise ValueError(f"hpo.search_space[{name!r}] needs exactly 2 bounds [low, high]")
        low, high = entry.bounds
        suggested[name] = trial.suggest_float(name, float(low), float(high), log=(kind == "log_uniform"))
    return sanitise_overrides(suggested)


def _apply_overrides(
    overrides: dict[str, Any], training_cfg: TrainingConfig, model_cfg: ModelConfig
) -> tuple[TrainingConfig, ModelConfig]:
    """Route each named hyperparameter to the config object that actually
    has that field. Raises on a name neither config recognizes -- see the
    module docstring for why silent misrouting is the failure mode this
    guards against."""
    training_updates: dict[str, Any] = {}
    model_subsection_updates: dict[str, Any] = {}

    for name, value in overrides.items():
        if name in _TRAINING_FIELDS:
            training_updates[name] = value
        elif name in _MODEL_SUBSECTION_FIELDS:
            model_subsection_updates[name] = value
        else:
            raise ValueError(
                f"hpo.search_space names {name!r}, which is not a recognised hyperparameter "
                f"(known: {_TRAINING_FIELDS + _MODEL_SUBSECTION_FIELDS}). Add it to "
                f"training.hpo._TRAINING_FIELDS or _MODEL_SUBSECTION_FIELDS, pointing at the "
                f"config object that actually holds it, before using it in a search space."
            )

    new_training_cfg = training_cfg.model_copy(update=training_updates) if training_updates else training_cfg

    new_model_cfg = model_cfg
    if model_subsection_updates:
        subsection = getattr(model_cfg, model_cfg.type)
        new_subsection = subsection.model_copy(update=model_subsection_updates)
        new_model_cfg = model_cfg.model_copy(update={model_cfg.type: new_subsection})

    return new_training_cfg, new_model_cfg


def build_sampler(name: str) -> Any:
    """Build an Optuna sampler. TPE is the default: it models the density of
    good and bad configurations separately, which suits a small budget over
    a handful of continuous hyperparameters better than random search."""
    import optuna

    key = str(name or "tpe").strip().lower()
    if key == "cmaes":
        return optuna.samplers.CmaEsSampler()
    if key == "random":
        return optuna.samplers.RandomSampler()
    return optuna.samplers.TPESampler()


def build_pruner(name: str, max_resource: Optional[int] = None) -> Any:
    """Build an Optuna pruner. ``none`` is the reference setting: with only
    a handful of trials over 2-3 hyperparameters, pruning risks discarding a
    configuration that starts slow and finishes well."""
    import optuna

    key = str(name or "none").strip().lower()
    if key == "median":
        return optuna.pruners.MedianPruner()
    if key == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=max(1, int(max_resource)) if max_resource else "auto"
        )
    return optuna.pruners.NopPruner()


def run_trials(
    hpo_cfg: HpoConfig,
    training_cfg: TrainingConfig,
    model_cfg: ModelConfig,
    train_records: list[SceneRecord],
    val_records: list[SceneRecord],
    mean,
    std,
    class_counts,
    num_classes: int,
    class_names: tuple[str, ...],
    feature_names: tuple[str, ...],
    num_sensors: int = 1,
    device: Optional[str] = None,
) -> tuple[TrainingResult, TrainingConfig, ModelConfig, list[TrialRecord]]:
    """Run the search (if any) and return the final model.

    Returns ``(result, winning_training_cfg, winning_model_cfg,
    trial_records)``. With ``hpo_cfg.trials <= 0`` or an empty search space,
    ``trial_records`` is empty and both winning configs are the ones passed
    in, unchanged.
    """
    if hpo_cfg.trials <= 0 or not hpo_cfg.search_space:
        result = train_model(
            training_cfg, model_cfg, train_records, val_records, mean, std, class_counts,
            num_classes, class_names, feature_names, num_sensors=num_sensors, device=device,
        )
        return result, training_cfg, model_cfg, []

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    metric_name = canonical_main_iou_metric(training_cfg.main_iou_metric)
    records: list[TrialRecord] = []

    trial_epochs = hpo_cfg.trial_epochs if hpo_cfg.trial_epochs > 0 else training_cfg.epochs

    def objective(trial: Any) -> float:
        overrides = _suggest_with_optuna(trial, hpo_cfg.search_space)
        trial_training_cfg, trial_model_cfg = _apply_overrides(overrides, training_cfg, model_cfg)
        trial_training_cfg = trial_training_cfg.model_copy(
            update={"epochs": trial_epochs, "patience": min(training_cfg.patience, trial_epochs)}
        )
        print(f"[hpo] trial {trial.number + 1}/{hpo_cfg.trials} {overrides}")

        try:
            result = train_model(
                trial_training_cfg, trial_model_cfg, train_records, val_records, mean, std, class_counts,
                num_classes, class_names, feature_names, num_sensors=num_sensors, device=device,
                progress=False,
            )
        except Exception as exc:  # noqa: BLE001 - one bad trial must not end the sweep
            records.append(
                TrialRecord(trial_id=trial.number + 1, params=overrides, score=-1e12, failed=True,
                            error=f"{type(exc).__name__}: {exc}")
            )
            print(f"[hpo] trial {trial.number + 1} failed: {exc}")
            return -1e12

        score = float(result.best_state.get("val_metric", float("nan")))
        if not math.isfinite(score):
            score = -1e12
        records.append(
            TrialRecord(trial_id=trial.number + 1, params=overrides, score=score, metrics=dict(result.best_state))
        )
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=build_sampler(hpo_cfg.sampler),
        pruner=build_pruner(hpo_cfg.pruner, max_resource=trial_epochs),
        storage=hpo_cfg.storage,
        load_if_exists=hpo_cfg.storage is not None,
    )
    study.optimize(objective, n_trials=hpo_cfg.trials)

    usable = [r for r in records if r.usable]
    if not usable:
        raise RuntimeError(
            "no Optuna trial produced a usable score; check hpo.search_space, "
            "the data, and available GPU memory"
        )
    best = max(usable, key=lambda r: r.score)
    best_params = sanitise_overrides(best.params)

    print(f"[hpo] best {metric_name} at {best_params}; retraining at full length")
    final_training_cfg, final_model_cfg = _apply_overrides(best_params, training_cfg, model_cfg)
    final_result = train_model(
        final_training_cfg, final_model_cfg, train_records, val_records, mean, std, class_counts,
        num_classes, class_names, feature_names, num_sensors=num_sensors, device=device,
    )
    return final_result, final_training_cfg, final_model_cfg, records
