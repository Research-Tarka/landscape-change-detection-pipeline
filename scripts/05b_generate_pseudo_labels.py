#!/usr/bin/env python3
"""Stage 5b -- semi-supervised pseudo-labeling over the unlabeled scene pool.

Purpose
-------
Score every scene under ``config.dem.tile_dir`` (``data/tiles``) that has no
MaskForge annotation, using an already-trained checkpoint
(``scripts/05_train_model.py``'s output), keep only the pixels the model is
confident about (``training.pseudo_label.confidence_threshold``), and write
the result into ``training.pseudo_label.pseudo_label_root`` in the same
per-scene ``.npz`` cache schema real annotated scenes use -- so a later
``scripts/05_train_model.py`` run with ``training.pseudo_label.enabled: true``
picks it up as a second, reduced-weight training set (see
:class:`landscape_change_detection_pipeline.config.PseudoLabelConfig`).

This step is deliberately separate from training: ``data/tiles`` is far
larger than what fits in RAM (see
:mod:`landscape_change_detection_pipeline.training.pseudo_label`'s module
docstring), so it streams one unlabeled scene at a time rather than loading
a corpus the way :class:`~landscape_change_detection_pipeline.training.train.PatchDataset`
does for the (much smaller) annotated cache.

Usage
-----
    python scripts/05b_generate_pseudo_labels.py --config configs/config.yaml \\
        --checkpoint models/checkpoints/unet_best.pt
"""

from __future__ import annotations

import os

# Windows/conda: torch (libiomp5md) and another lib (libomp) both ship an
# OpenMP runtime; must be set before either is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landscape_change_detection_pipeline.config import load_config  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to a trained checkpoint (default: <training.checkpoint_dir>/<model.type>_best.pt)",
    )
    parser.add_argument("--device", default=None, help="cuda | cpu (default: auto-detect)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(list(argv if argv is not None else sys.argv[1:]))
    config = load_config(args.config, args.env_file)
    pseudo_cfg = config.training.pseudo_label

    if not pseudo_cfg.enabled:
        print("[pseudo-label] training.pseudo_label.enabled is false -- nothing to do")
        return 0

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (
        Path(config.training.checkpoint_dir) / f"{config.model.type}_best.pt"
    )
    if not checkpoint_path.is_file():
        print(f"[pseudo-label] no checkpoint at {checkpoint_path} -- run scripts/05_train_model.py first")
        return 1

    from landscape_change_detection_pipeline.training.train import (
        load_checkpoint,
        rebuild_model_from_checkpoint,
        resolve_device,
    )
    from landscape_change_detection_pipeline.training.pseudo_label import generate_pseudo_labels
    from landscape_change_detection_pipeline.classes.class_config import label_remap_table, load_class_config

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    model = rebuild_model_from_checkpoint(checkpoint).to(device)
    model.eval()

    meta = checkpoint["metadata"]
    import numpy as np

    mean = np.asarray(meta["mean"], dtype=np.float32)
    std = np.asarray(meta["std"], dtype=np.float32)
    num_classes = int(meta["num_classes"])
    feature_names = list(meta["feature_names"])

    from landscape_change_detection_pipeline.config import resolved_feature_names

    index_names, dem_layer_names = resolved_feature_names(config.features)
    expected = [
        *index_names, *dem_layer_names,
        *(("doy_sin", "doy_cos") if config.features.include_doy_features else ()),
        *(("lat_norm", "lon_norm") if config.features.include_latlon_features else ()),
    ]
    if list(feature_names) != expected:
        print(
            "[pseudo-label] WARNING: checkpoint feature_names do not match the current "
            "config.features -- the checkpoint was trained on a different feature stack; "
            "results will be meaningless unless config.features matches what training used."
        )

    checkpoint_id = f"{checkpoint_path.resolve()}::{checkpoint_path.stat().st_mtime_ns}"

    written = generate_pseudo_labels(
        tile_dir=config.dem.tile_dir,
        mask_root=config.features.mask_root,
        pseudo_label_root=pseudo_cfg.pseudo_label_root,
        model=model,
        checkpoint_id=checkpoint_id,
        mean=mean,
        std=std,
        num_classes=num_classes,
        index_names=tuple(index_names),
        dem_layer_names=tuple(dem_layer_names),
        include_doy_features=config.features.include_doy_features,
        include_latlon_features=config.features.include_latlon_features,
        confidence_threshold=pseudo_cfg.confidence_threshold,
        max_scenes=pseudo_cfg.max_scenes,
        sensors=pseudo_cfg.sensors,
        patch_size=config.inference.patch_size,
        stride=pseudo_cfg.inference_stride or config.inference.patch_size,
        inference_batch_size=pseudo_cfg.inference_batch_size,
        use_amp=pseudo_cfg.use_amp,
        rare_classes=tuple(pseudo_cfg.rare_classes),
        rare_confidence_threshold=pseudo_cfg.rare_confidence_threshold,
        min_rare_pixels=pseudo_cfg.min_rare_pixels,
        prefetch_workers=pseudo_cfg.prefetch_workers,
        label_remap=label_remap_table(
            [c.name for c in load_class_config().classes], config.training.class_merge
        ),
    )
    print(f"[pseudo-label] wrote {len(written)} pseudo-labeled scene caches to {pseudo_cfg.pseudo_label_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
