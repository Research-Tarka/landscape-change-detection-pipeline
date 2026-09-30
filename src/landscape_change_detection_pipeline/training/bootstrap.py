"""Multi-seed bootstrap: quantifying initialization variance.

Trains the same configuration ``bootstrap.n_seeds`` times, varying only the
random seed, and reports the spread. This separates two things a single run
conflates: how good the configuration is, and how much of a given run's
score was luck in weight initialization and mini-batch ordering.

The protocol's key property is that **the split is identical across
seeds** -- only ``training.seed`` varies, never ``split.split_seed``. If the
split changed too, the resulting spread would mix initialization variance
with split variance and could not be reported as either.

Reporting the mean and population standard deviation across seeds, rather
than the best seed, is what makes the number honest -- picking the best of
five and reporting it is a silent one-in-five selection on the validation
set.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch

from landscape_change_detection_pipeline.config import BootstrapConfig, HpoConfig, ModelConfig, TrainingConfig
from landscape_change_detection_pipeline.training.dataset import SceneRecord
from landscape_change_detection_pipeline.training.hpo import run_trials
from landscape_change_detection_pipeline.training.train import TrainingResult

__all__ = ["run_bootstrap", "build_ensemble", "aggregate_rows", "AGGREGATE_METRICS"]

#: Every numeric field a per-seed summary can carry, aggregated across
#: seeds. Listed explicitly rather than discovered, so a field appearing in
#: only some seeds' summaries cannot silently change what the aggregate
#: covers.
AGGREGATE_METRICS: tuple[str, ...] = (
    "best_epoch",
    "stopped_epoch",
    "val_metric",
    "train_loss",
    "val_loss",
)


def build_ensemble(state_dicts: list[dict[str, Any]], out_dir: Path) -> Optional[Path]:
    """Write an ensemble checkpoint holding every seed's weights side by
    side. Averaging weights across independently initialized networks is
    meaningless -- their hidden units are not in correspondence. The
    ensemble is applied at inference by averaging softmax outputs instead,
    which is well defined."""
    if not state_dicts:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ensemble.pt"
    torch.save({"ensemble_state_dicts": state_dicts, "n_models": len(state_dicts), "ensemble": True}, path)
    return path


def _seed_summary(seed: int, result: TrainingResult) -> dict[str, Any]:
    row: dict[str, Any] = {"seed": seed, "best_epoch": result.best_epoch, "stopped_epoch": result.stopped_epoch}
    for key in ("val_metric", "train_loss", "val_loss", "metric_name"):
        row[key] = result.best_state.get(key)
    return row


def _flatten_eval(eval_by_split: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    """``{split: metrics}`` -> flat numeric row (``val_kappa``, ``test_iou_open_water``...)
    so the existing mean/std aggregation covers every split and every class."""
    flat: dict[str, float] = {}
    for split, metrics in eval_by_split.items():
        if not metrics:
            continue
        macro = metrics.get("macro", {})
        for key in ("kappa", "mcc", "miou", "miou_weighted", "miou_inv_freq"):
            if key in macro:
                flat[f"{split}_{key}"] = macro[key]
        for entry in metrics.get("per_class", []):
            for key in ("iou", "precision", "recall"):
                flat[f"{split}_{key}_{entry['class_name']}"] = entry[key]
    return flat


def run_bootstrap(
    bootstrap_cfg: BootstrapConfig,
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
    run_root: str | Path,
    num_sensors: int = 1,
    device: Optional[str] = None,
    num_spectral_channels: Optional[int] = None,
    pseudo_label_records: Optional[list[SceneRecord]] = None,
    evaluate_fn: Optional[Callable[[Any], dict[str, Any]]] = None,
) -> Path:
    """Train ``bootstrap.n_seeds`` models (each optionally HPO-tuned) and
    summarise the spread. Returns the run directory holding
    ``seed_*/checkpoint.pt``, ``ensemble/``, and the summary files.

    ``evaluate_fn(model)``, when given, returns ``{split: metrics}`` (train /
    val / test, each a :func:`compute_confusion_metrics` dict plus its
    confusion matrix); it is run per seed, written to
    ``seed_*/seed_report.json`` together with that seed's HPO trials and
    training history, and its kappa/MCC/mIoU/per-class values are aggregated
    (mean +/- std) across seeds."""
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    n_seeds = max(1, bootstrap_cfg.n_seeds)
    print(f"[bootstrap] {n_seeds} seed(s) from base {bootstrap_cfg.seed_base} -> {run_root}")

    summaries: list[dict[str, Any]] = []
    state_dicts: list[dict[str, Any]] = []

    for offset in range(n_seeds):
        seed = bootstrap_cfg.seed_base + offset
        seed_dir = run_root / f"seed_{offset}"
        print(f"[bootstrap] seed {seed}")

        # Only `seed` changes; split.split_seed (baked into train_records/
        # val_records before this call) is untouched, so every seed trains
        # and validates on exactly the same scenes.
        seed_cfg = training_cfg.model_copy(update={"seed": seed})

        try:
            result, _winning_training_cfg, winning_model_cfg, trials = run_trials(
                hpo_cfg, seed_cfg, model_cfg, train_records, val_records, mean, std, class_counts,
                num_classes, class_names, feature_names, num_sensors=num_sensors, device=device,
                num_spectral_channels=num_spectral_channels, pseudo_label_records=pseudo_label_records,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue the sweep
            print(f"[bootstrap] seed {seed} failed: {exc}")
            summaries.append({"seed": seed, "error": str(exc)})
            continue

        from landscape_change_detection_pipeline.training.train import checkpoint_metadata, save_checkpoint

        meta = checkpoint_metadata(
            winning_model_cfg, in_channels=len(feature_names), num_classes=num_classes,
            class_names=class_names, feature_names=feature_names, mean=mean, std=std, num_sensors=num_sensors,
        )
        seed_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(seed_dir / "checkpoint.pt", result.model, meta)

        row = _seed_summary(seed, result)
        eval_by_split: dict[str, Any] = {}
        if evaluate_fn is not None:
            try:
                eval_by_split = evaluate_fn(result.model)
                row.update(_flatten_eval(eval_by_split))
            except Exception as exc:  # noqa: BLE001 - a failed evaluation must not lose the trained seed
                print(f"[bootstrap] seed {seed} evaluation failed: {exc}")
                row["eval_error"] = str(exc)
        summaries.append(row)
        (seed_dir / "seed_report.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "summary": row,
                    "eval": eval_by_split,
                    "hpo_trials": [
                        {"trial_id": t.trial_id, "params": t.params, "score": t.score,
                         "pruned": t.pruned, "failed": t.failed, "error": t.error}
                        for t in trials
                    ],
                    "winning_training": _winning_training_cfg.model_dump(mode="json"),
                    "winning_model": winning_model_cfg.model_dump(mode="json"),
                    "history": result.history,
                    "class_counts": [int(c) for c in class_counts],
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        if result.best_state.get("model") is not None:
            state_dicts.append(result.best_state["model"])

    ensemble_path = build_ensemble(state_dicts, run_root / "ensemble") if bootstrap_cfg.create_ensemble else None

    extra_keys = sorted({k for row in summaries for k in row if k not in AGGREGATE_METRICS})
    aggregate = _aggregate(summaries, (*AGGREGATE_METRICS, *extra_keys))
    (run_root / "bootstrap_summary.json").write_text(
        json.dumps(
            {
                "n_seeds": n_seeds,
                "seed_base": bootstrap_cfg.seed_base,
                "ensemble_path": str(ensemble_path) if ensemble_path else None,
                "per_seed": summaries,
                "aggregate": aggregate,
                "created_at": datetime.now().isoformat(),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    for metric, stats in sorted(aggregate.items()):
        print(f"[bootstrap] {metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

    return run_root


def _aggregate(
    summaries: list[dict[str, Any]], metrics: Sequence[str] = AGGREGATE_METRICS
) -> dict[str, dict[str, float]]:
    """Mean, std, min and max of each metric across the seeds that
    succeeded. Population std (ddof=0), matching how seed-to-seed spread is
    conventionally reported for a fixed, small number of replications."""
    out: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = [
            float(row[metric])
            for row in summaries
            if isinstance(row.get(metric), (int, float)) and not isinstance(row.get(metric), bool)
            and row[metric] == row[metric]
        ]
        if values:
            array = np.asarray(values, dtype=np.float64)
            out[metric] = {
                "mean": float(array.mean()),
                "std": float(array.std()),
                "min": float(array.min()),
                "max": float(array.max()),
                "n": len(values),
            }
    return out


def aggregate_rows(aggregate: Mapping[str, Mapping[str, float]]) -> list[dict[str, Any]]:
    """The aggregate as one row per metric, for a CSV alongside the per-seed rows."""
    return [
        {
            "metric": metric,
            "n_seeds": int(stats.get("n", 0)),
            "mean": stats.get("mean"),
            "std": stats.get("std"),
            "min": stats.get("min"),
            "max": stats.get("max"),
        }
        for metric, stats in sorted(aggregate.items())
    ]
