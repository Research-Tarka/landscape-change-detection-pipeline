#!/usr/bin/env python3
"""Stage 5 -- train/val/test split, then train a land-cover model.

Purpose
-------
Discover every scene exported by Stage 4, build the tile-level train/val/test
split, and train the model selected by ``config.model.type``.
For the torch model types (``unet``, ``deeplabv3plus``, ``segformer``) this runs
the full mixed-precision training loop and writes a
self-describing checkpoint; the non-torch types (``threshold``,
``random_forest``, ``catboost``) are fit directly via their own
``build_<name>(...).fit(...)`` (no epoch loop, so the training loop does not
apply to them).

Usage
-----
    python scripts/05_train_model.py --config configs/config.yaml
    python scripts/05_train_model.py --epochs 3 --patience 2   # quick smoke test
"""

from __future__ import annotations

import argparse
import os
import sys

# Must be set before torch initialises CUDA: avoids allocator fragmentation (VRAM creep).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landscape_change_detection_pipeline.classes.class_config import (  # noqa: E402
    label_remap_table,
    load_class_config,
)
from landscape_change_detection_pipeline.config import (  # noqa: E402
    all_resolved_feature_names,
    load_config,
    resolved_feature_names,
)
from landscape_change_detection_pipeline.training.dataset import (  # noqa: E402
    SceneRecord,
    collect_class_pixel_counts_by_scene,
    compute_mean_std,
    discover_scene_records,
    split_scenes,
    with_label_remap,
)
from landscape_change_detection_pipeline.training.losses import IGNORE_INDEX  # noqa: E402
from landscape_change_detection_pipeline.training.metrics import compute_confusion_metrics  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument("--classes", default=None, help="Path to classes.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument("--patience", type=int, default=None, help="Override training.patience")
    parser.add_argument("--device", default=None, help="cuda | cpu (default: auto-detect)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv if argv is not None else sys.argv[1:]))
    config = load_config(args.config, args.env_file)
    class_config = load_class_config(args.classes)
    num_classes = len(class_config.classes)
    class_names = tuple(c.name for c in class_config.classes)

    training_cfg = config.training.model_copy(
        update={
            k: v
            for k, v in {"epochs": args.epochs, "patience": args.patience}.items()
            if v is not None
        }
    )

    label_remap = label_remap_table(class_names, training_cfg.class_merge)
    if label_remap is not None:
        merged_ids = {class_names.index(src) for src in training_cfg.class_merge}
        for where, rare in (
            ("training.scene_oversampling.rare_classes", training_cfg.scene_oversampling.rare_classes),
            ("training.pseudo_label.rare_classes", training_cfg.pseudo_label.rare_classes),
        ):
            stale = sorted(merged_ids & set(rare))
            if stale:
                print(
                    f"[train] WARNING: {where} lists merged class id(s) {stale} "
                    f"({[class_names[i] for i in stale]}) -- they no longer exist after class_merge; remove them"
                )
        print(f"[train] class_merge: {dict(training_cfg.class_merge)}")

    records = discover_scene_records(config.features.train_root)
    print(f"[train] found {len(records)} cached scenes under {config.features.train_root}")
    if not records:
        print("[train] no training data -- run scripts/04_export_training_cache.py first")
        return 1
    records = with_label_remap(records, label_remap)

    class_pixel_counts = collect_class_pixel_counts_by_scene(records, num_classes)
    split_result = split_scenes(
        records,
        class_pixel_counts,
        num_classes,
        ratios=tuple(config.split.ratios),
        split_seed=config.split.split_seed,
        split_by=config.split.split_by,
    )
    train_records = split_result.scenes_by_partition["train"]
    val_records = split_result.scenes_by_partition["val"]
    test_records = split_result.scenes_by_partition["test"]
    group_label = "tiles" if config.split.split_by == "tile" else "scenes (split_by='scene')"
    print(
        f"[train] split: {len(train_records)} train / {len(val_records)} val / "
        f"{len(test_records)} test scenes ({len(split_result.tile_partition)} {group_label})"
    )

    mean, std = compute_mean_std(train_records)
    from landscape_change_detection_pipeline.training.dataset import compute_class_counts

    class_counts = compute_class_counts(train_records, num_classes)
    feature_names = all_resolved_feature_names(config.features)
    index_names, _dem_layer_names = resolved_feature_names(config.features)
    num_spectral_channels = len(index_names)

    model_type = config.model.type
    splits = {"train": train_records, "val": val_records, "test": test_records}
    if model_type in ("threshold", "random_forest", "catboost", "lightgbm"):
        return _fit_non_torch_model(
            config, model_type, train_records, val_records, num_classes, class_config, class_names,
            splits=splits,
        )

    from landscape_change_detection_pipeline.training.train import checkpoint_metadata, save_checkpoint

    pseudo_label_records = None
    if training_cfg.pseudo_label.enabled:
        pseudo_label_records = with_label_remap(
            discover_scene_records(training_cfg.pseudo_label.pseudo_label_root), label_remap
        )
        print(
            f"[train] pseudo-label: found {len(pseudo_label_records)} cached scenes under "
            f"{training_cfg.pseudo_label.pseudo_label_root} "
            f"(run scripts/05b_generate_pseudo_labels.py first if this is 0)"
        )

    if config.bootstrap.enabled:
        from landscape_change_detection_pipeline.training.bootstrap import run_bootstrap

        run_root = Path(training_cfg.checkpoint_dir) / f"bootstrap_{model_type}"
        run_bootstrap(
            config.bootstrap, config.hpo, training_cfg, config.model, train_records, val_records,
            mean, std, class_counts, num_classes, class_names, feature_names, run_root,
            num_sensors=(config.model.unet.num_sensors if model_type == "unet" else 1), device=args.device,
            num_spectral_channels=num_spectral_channels, pseudo_label_records=pseudo_label_records,
            evaluate_fn=lambda model: evaluate_all_splits(
                model, model_type, splits, num_classes, class_names, mean=mean, std=std,
                patch_size=config.inference.patch_size, stride=config.inference.stride,
                batch_size=config.inference.batch_size,
            ),
        )
        write_run_report(
            run_root, f"bootstrap_{model_type}", config, None,
            training={"class_names": list(class_names), "class_counts": [int(c) for c in class_counts],
                      "n_scenes": {k: len(v) for k, v in splits.items()}},
            note="per-seed metrics/HPO in seed_*/seed_report.json; mean/std across seeds in bootstrap_summary.json",
        )
        print(f"[train] bootstrap run written to {run_root}")
        return 0

    from landscape_change_detection_pipeline.training.hpo import make_cv_folds, run_trials

    cv_folds = None
    if config.hpo.trials > 0 and config.hpo.cv_folds >= 2:
        cv_folds = make_cv_folds(
            train_records + val_records, config.hpo.cv_folds, class_pixel_counts, num_classes,
            seed=config.split.split_seed, split_by=config.split.split_by,
        )
        print(
            f"[hpo] {len(cv_folds)}-fold class-balanced CV (split_by={config.split.split_by!r}) over train+val "
            f"(fold val sizes: {[len(v) for _, v in cv_folds]})"
        )

    result, winning_training_cfg, winning_model_cfg, trial_records = run_trials(
        config.hpo, training_cfg, config.model, train_records, val_records, mean, std, class_counts,
        num_classes, class_names, feature_names, num_sensors=(config.model.unet.num_sensors if model_type == "unet" else 1), device=args.device,
        num_spectral_channels=num_spectral_channels, pseudo_label_records=pseudo_label_records,
        cv_folds=cv_folds,
    )
    if trial_records:
        print(
            f"[train] HPO: {len(trial_records)} trials, best config "
            f"lr={winning_training_cfg.lr} weight_decay={winning_training_cfg.weight_decay}"
        )

    print(
        f"[train] best epoch {result.best_epoch} (stopped at {result.stopped_epoch}): "
        f"{result.best_state.get('metric_name')}={result.best_state.get('val_metric')}"
    )

    meta = checkpoint_metadata(
        winning_model_cfg,
        in_channels=len(feature_names),
        num_classes=num_classes,
        class_names=class_names,
        feature_names=feature_names,
        mean=mean,
        std=std,
        num_sensors=(config.model.unet.num_sensors if model_type == "unet" else 1),
    )
    checkpoint_path = Path(training_cfg.checkpoint_dir) / f"{model_type}_best.pt"
    out = save_checkpoint(checkpoint_path, result.model, meta)
    print(f"[train] wrote checkpoint to {out}")

    eval_metrics = evaluate_all_splits(
        result.model, model_type, splits, num_classes, class_names, mean=mean, std=std,
        patch_size=config.inference.patch_size, stride=config.inference.stride,
        batch_size=config.inference.batch_size,
    )
    write_run_report(
        training_cfg.checkpoint_dir, model_type, config, eval_metrics,
        hpo={
            "trials": [
                {"trial_id": t.trial_id, "params": t.params, "score": t.score,
                 "pruned": t.pruned, "failed": t.failed, "error": t.error}
                for t in trial_records
            ],
            "winning_training": winning_training_cfg.model_dump(mode="json"),
            "winning_model": winning_model_cfg.model_dump(mode="json"),
        },
        training={
            "best_epoch": result.best_epoch,
            "stopped_epoch": result.stopped_epoch,
            "best_metric_name": result.best_state.get("metric_name"),
            "best_val_metric": result.best_state.get("val_metric"),
            "history": result.history,
            "class_counts": [int(c) for c in class_counts],
            "class_names": list(class_names),
            "n_scenes": {"train": len(train_records), "val": len(val_records), "test": len(test_records)},
            "pseudo_label_scenes": len(pseudo_label_records or []),
        },
        checkpoint=str(out),
    )
    return 0


def _fit_non_torch_model(
    config, model_type: str, train_records, val_records, num_classes, class_config, class_names,
    splits=None,
) -> int:
    """Fit a non-epoch-based model type.

    Each of the four non-torch model types has its own real ``fit``
    contract (none of them take a pre-flattened ``(X, y)`` pair):
    ``RandomForestModel.fit``, ``CatBoostModel.fit``, and
    ``LightGBMModel.fit`` all load and flatten ``SceneRecord``s themselves
    (see their own modules' memory-tradeoff docstrings for why), and
    ``build_threshold`` runs its Optuna search internally when given
    ``train_records`` rather than exposing a separate ``.fit()``. Dispatch
    to each one's real entry point instead of assuming a shared
    sklearn-style ``.fit(X, y)`` across all four.
    """
    n_train_pixels = sum(
        int((load_scene_cache_labels(r.cache_dir, r.label_remap) != IGNORE_INDEX).sum()) for r in train_records
    )
    print(f"[train] fitting {model_type} on {len(train_records)} scenes (~{n_train_pixels} valid pixels)")

    if model_type == "threshold":
        from landscape_change_detection_pipeline.models.threshold import build_threshold

        cfg = config.model.threshold
        model = build_threshold(
            num_classes=num_classes,
            n_trials=cfg.n_trials,
            sampler=cfg.sampler,
            timeout_s=cfg.timeout_s,
            study_storage=cfg.study_storage,
            train_records=train_records,
        )
    elif model_type == "random_forest":
        from landscape_change_detection_pipeline.models.registry import build_model

        model = build_model(config.model, in_channels=0, num_classes=num_classes)
        model.fit(train_records)
    elif model_type == "catboost":
        cfg = config.model.catboost
        if cfg.n_trials > 0 and val_records:
            from landscape_change_detection_pipeline.models.catboost_model import search_catboost

            model = search_catboost(
                train_records, val_records, num_classes,
                n_trials=cfg.n_trials, sampler=cfg.sampler, pruner=cfg.pruner, timeout_s=cfg.timeout_s,
                storage=cfg.study_storage, task_type=cfg.task_type, random_state=cfg.random_state,
                class_weighting=cfg.class_weighting,
                search_iterations=cfg.search_iterations,
                search_early_stopping_rounds=cfg.search_early_stopping_rounds,
                final_iterations=cfg.iterations,
                final_early_stopping_rounds=cfg.early_stopping_rounds,
            )
        else:
            from landscape_change_detection_pipeline.models.registry import build_model

            model = build_model(config.model, in_channels=0, num_classes=num_classes)
            model.fit(train_records, eval_records=val_records or None)
    elif model_type == "lightgbm":
        cfg = config.model.lightgbm
        if cfg.n_trials > 0 and val_records:
            from landscape_change_detection_pipeline.models.lightgbm_model import search_lightgbm

            model = search_lightgbm(
                train_records, val_records, num_classes,
                n_trials=cfg.n_trials, sampler=cfg.sampler, pruner=cfg.pruner, timeout_s=cfg.timeout_s,
                storage=cfg.study_storage, device=cfg.device, random_state=cfg.random_state,
                class_weighting=cfg.class_weighting,
                search_n_estimators=cfg.search_n_estimators,
                search_early_stopping_rounds=cfg.search_early_stopping_rounds,
                final_n_estimators=cfg.n_estimators,
                final_early_stopping_rounds=cfg.early_stopping_rounds,
            )
        else:
            from landscape_change_detection_pipeline.models.registry import build_model

            model = build_model(config.model, in_channels=0, num_classes=num_classes)
            model.fit(train_records, eval_records=val_records or None)
    else:  # pragma: no cover - config.model.type's own validator already restricts this
        raise ValueError(f"Unsupported non-torch model.type={model_type!r}")

    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    out_path = checkpoint_dir / f"{model_type}_best.joblib"
    import joblib

    joblib.dump(model, out_path)
    print(f"[train] wrote {model_type} model to {out_path}")

    eval_metrics = evaluate_all_splits(
        model, model_type, splits or {"train": train_records, "val": val_records}, num_classes, class_names,
    )
    write_run_report(
        checkpoint_dir, model_type, config, eval_metrics,
        hpo=getattr(model, "hpo_summary", None),
        training={"n_scenes": {"train": len(train_records), "val": len(val_records)}, "class_names": list(class_names)},
        checkpoint=str(out_path),
    )
    return 0


def evaluate_val_records(
    model,
    model_type: str,
    val_records: list[SceneRecord],
    num_classes: int,
    class_names: tuple[str, ...],
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    patch_size: int = 256,
    stride: int = 128,
    batch_size: int = 32,
) -> dict:
    """Pooled confusion matrix + Kappa/MCC/per-class IoU on ``val_records``,
    for any of the seven supported model types (see ``config.py::MODEL_TYPES``).

    Dispatches on ``model_type`` only to pick the right prediction path
    (:func:`predict_scene_sklearn` for the four non-torch types,
    :func:`predict_scene` for the three torch types) -- everything after that
    (confusion accumulation, metrics) is identical for every model type, so
    this is the one place results across models are directly comparable.
    """
    from landscape_change_detection_pipeline.features.scene_context import record_context
    from landscape_change_detection_pipeline.features.training_cache import load_scene_cache
    from landscape_change_detection_pipeline.inference.engine import predict_scene, predict_scene_sklearn

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    non_torch = model_type in ("threshold", "random_forest", "catboost", "lightgbm")

    for record in val_records:
        cache = load_scene_cache(record.cache_dir, label_remap=record.label_remap)
        features = cache["features"].astype(np.float32)
        labels = cache["labels"]

        if non_torch:
            pred = predict_scene_sklearn(model, features, num_classes, nodata=IGNORE_INDEX)
        else:
            pred = predict_scene(
                model, features, num_classes, mean, std,
                patch_size=patch_size, stride=stride, batch_size=batch_size, nodata=IGNORE_INDEX,
                context=record_context(record) if getattr(model, "uses_scene_context", False) else None,
            )

        if record.label_remap is not None:
            # Fold predictions of a merged-away class onto its target (see train.evaluate).
            pred = np.asarray(record.label_remap, dtype=np.uint8)[pred]

        mask = (labels != IGNORE_INDEX) & (pred != IGNORE_INDEX)
        true_flat = labels[mask].astype(np.int64)
        pred_flat = pred[mask].astype(np.int64)
        flat_index = true_flat * num_classes + pred_flat
        counts = np.bincount(flat_index, minlength=num_classes * num_classes)
        confusion += counts.reshape(num_classes, num_classes)

    metrics = compute_confusion_metrics(confusion, num_classes, class_names=class_names)
    metrics["confusion_matrix"] = {"labels": list(class_names), "rows_true_cols_pred": confusion.tolist()}
    return metrics


def evaluate_all_splits(model, model_type: str, splits: dict, num_classes: int, class_names, **kwargs) -> dict:
    """``{split: metrics}`` for every non-empty split (train / val / test), each
    printed and carrying its confusion matrix. Train is evaluated on the same
    full-scene inference as val/test, so it is slower than val alone."""
    out: dict = {}
    for split, records in splits.items():
        if not records:
            print(f"[eval:{model_type}] {split}: skipped -- no scenes in this split")
            continue
        print(f"[eval:{model_type}] evaluating {split} ({len(records)} scenes)...")
        metrics = evaluate_val_records(model, model_type, records, num_classes, class_names, **kwargs)
        print(f"[eval:{model_type}] === {split} ===")
        print_evaluation(model_type, metrics)
        out[split] = metrics
    return out


def print_evaluation(model_type: str, metrics: dict) -> None:
    macro = metrics["macro"]
    print(f"[eval:{model_type}] kappa={macro['kappa']} mcc={macro['mcc']} "
          f"miou_macro={macro['miou']} miou_weighted={macro['miou_weighted']}")
    print(f"[eval:{model_type}] per-class metrics:")
    for entry in metrics["per_class"]:
        print(
            f"  {entry['class']:>2} {entry['class_name']:<28} iou={entry['iou']} "
            f"precision={entry['precision']} recall={entry['recall']} support={entry['support']}"
        )


def _compact_config(config) -> dict:
    """Only what identifies a run: the active model's own settings, the input
    variables and the training/HPO knobs -- not every unrelated config section
    (GEE, tiles, download, change detection, ...)."""
    dump = config.model_dump(mode="json")
    model_type = config.model.type
    features = dict(dump["features"])
    features["feature_names"] = list(all_resolved_feature_names(config.features))
    model = {"type": model_type, **(dump["model"].get(model_type) or {})}
    return {
        "model": model,
        "features": {k: features[k] for k in (
            "feature_names", "index_names", "dem_layer_names", "include_doy_features", "include_latlon_features",
        )},
        "training": dump["training"],
        "split": {k: dump["split"][k] for k in ("ratios", "split_by", "split_seed")},
    }


def _compact_metrics(metrics: dict) -> dict:
    """Per-split summary: macro scores, per-class IoU/precision/recall/support
    and the confusion matrix -- no specificity/f-beta duplicates."""
    return {
        "confusion_matrix": metrics.get("confusion_matrix"),
        "macro": {k: metrics["macro"][k] for k in ("miou", "miou_weighted", "miou_inv_freq", "kappa", "mcc")},
        "per_class": [
            {k: entry[k] for k in ("class_name", "iou", "precision", "recall", "support")}
            for entry in metrics["per_class"]
        ],
    }


def write_run_report(out_dir, model_type: str, config, eval_metrics: dict | None, **extra) -> Path:
    """One compact JSON per run, ``<out_dir>/<model_type>_run_report.json``:
    the model/feature/training config, every HPO trial + the winning params
    the per-epoch history and per-split macro, per-class metrics and confusion
    matrix."""
    import json
    from datetime import datetime

    hpo = extra.pop("hpo", None)
    training = extra.pop("training", None)

    report = {
        "model_type": model_type,
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "config": _compact_config(config),
        "eval": {split: _compact_metrics(m) for split, m in eval_metrics.items()} if eval_metrics else None,
        "hpo": hpo,
        "training": training,
        **extra,
    }
    out_path = Path(out_dir) / f"{model_type}_run_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[eval:{model_type}] run report saved to {out_path}")
    return out_path


def load_scene_cache_labels(cache_dir, label_remap=None) -> np.ndarray:
    """Just the ``labels`` array from one scene's cache -- for the pixel-count
    log line above, without loading the (much larger) feature stack too."""
    from landscape_change_detection_pipeline.features.training_cache import load_scene_cache

    return load_scene_cache(cache_dir, label_remap=label_remap)["labels"]


if __name__ == "__main__":
    raise SystemExit(main())
