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
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landscape_change_detection_pipeline.classes.class_config import load_class_config  # noqa: E402
from landscape_change_detection_pipeline.config import load_config, resolved_feature_names  # noqa: E402
from landscape_change_detection_pipeline.training.dataset import (  # noqa: E402
    collect_class_pixel_counts_by_scene,
    compute_mean_std,
    discover_scene_records,
    split_scenes,
)
from landscape_change_detection_pipeline.training.losses import IGNORE_INDEX  # noqa: E402


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

    records = discover_scene_records(config.features.train_root)
    print(f"[train] found {len(records)} cached scenes under {config.features.train_root}")
    if not records:
        print("[train] no training data -- run scripts/04_export_training_cache.py first")
        return 1

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
    index_names, dem_layer_names = resolved_feature_names(config.features)
    feature_names = tuple(index_names) + tuple(dem_layer_names)

    model_type = config.model.type
    if model_type in ("threshold", "random_forest", "catboost", "lightgbm"):
        return _fit_non_torch_model(config, model_type, train_records, val_records, num_classes, class_config)

    from landscape_change_detection_pipeline.training.train import checkpoint_metadata, save_checkpoint

    if config.bootstrap.enabled:
        from landscape_change_detection_pipeline.training.bootstrap import run_bootstrap

        run_root = Path(training_cfg.checkpoint_dir) / f"bootstrap_{model_type}"
        run_bootstrap(
            config.bootstrap, config.hpo, training_cfg, config.model, train_records, val_records,
            mean, std, class_counts, num_classes, class_names, feature_names, run_root,
            num_sensors=1, device=args.device,
        )
        print(f"[train] bootstrap run written to {run_root}")
        return 0

    from landscape_change_detection_pipeline.training.hpo import run_trials

    result, winning_training_cfg, winning_model_cfg, trial_records = run_trials(
        config.hpo, training_cfg, config.model, train_records, val_records, mean, std, class_counts,
        num_classes, class_names, feature_names, num_sensors=1, device=args.device,
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
        num_sensors=1,
    )
    checkpoint_path = Path(training_cfg.checkpoint_dir) / f"{model_type}_best.pt"
    out = save_checkpoint(checkpoint_path, result.model, meta)
    print(f"[train] wrote checkpoint to {out}")
    return 0


def _fit_non_torch_model(config, model_type: str, train_records, val_records, num_classes, class_config) -> int:
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
        int((load_scene_cache_labels(r.cache_dir) != IGNORE_INDEX).sum()) for r in train_records
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
        from landscape_change_detection_pipeline.models.registry import build_model

        model = build_model(config.model, in_channels=0, num_classes=num_classes)
        model.fit(train_records, eval_records=val_records or None)
    elif model_type == "lightgbm":
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
    return 0


def load_scene_cache_labels(cache_dir) -> np.ndarray:
    """Just the ``labels`` array from one scene's cache -- for the pixel-count
    log line above, without loading the (much larger) feature stack too."""
    from landscape_change_detection_pipeline.features.training_cache import load_scene_cache

    return load_scene_cache(cache_dir)["labels"]


if __name__ == "__main__":
    raise SystemExit(main())
