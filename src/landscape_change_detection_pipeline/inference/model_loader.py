"""Load a trained checkpoint for inference, rebuilding its exact architecture
from its own self-describing metadata.

A thin wrapper around :mod:`landscape_change_detection_pipeline.training.train`'s
checkpoint functions (:func:`~.train.load_checkpoint`,
:func:`~.train.rebuild_model_from_checkpoint`): the caller never has to know
or guess ``model.type``, channel count, class count, or architecture
hyperparameters -- they are embedded in the checkpoint at save time (see
``training/train.py::checkpoint_metadata``) and read back here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class LoadedModel:
    """A model plus everything inference needs to reproduce training's
    preprocessing: normalization stats and the exact feature order the
    checkpoint was trained on.

    ``is_torch`` distinguishes the two inference paths
    :mod:`.engine` supports: the sliding-window, softmax-normalized path
    (``predict_scene``) for the torch models (unet/deeplabv3plus/segformer),
    versus the direct per-pixel ``predict_proba`` path (``predict_scene_sklearn``)
    for threshold/random_forest/catboost/lightgbm -- those four have no
    notion of a receptive field larger than one pixel, so sliding-window
    patching and mean/std normalization (``mean``/``std`` are meaningless for
    them and left as zeros/ones) would add cost without changing the
    result.
    """

    def __init__(
        self,
        model,
        mean: np.ndarray,
        std: np.ndarray,
        feature_names: tuple[str, ...],
        class_names: tuple[str, ...],
        num_classes: int,
        is_torch: bool = True,
    ) -> None:
        self.model = model
        self.mean = mean
        self.std = std
        self.feature_names = feature_names
        self.class_names = class_names
        self.num_classes = num_classes
        self.is_torch = is_torch


def _load_torch_checkpoint(checkpoint_path: str | Path, device: Optional[str]) -> LoadedModel:
    from landscape_change_detection_pipeline.training.train import load_checkpoint, rebuild_model_from_checkpoint, resolve_device

    resolved_device = resolve_device(device)
    checkpoint = load_checkpoint(checkpoint_path, device=resolved_device)
    model = rebuild_model_from_checkpoint(checkpoint)
    model = model.to(resolved_device)
    model.eval()

    meta = checkpoint["metadata"]
    return LoadedModel(
        model=model,
        mean=np.asarray(meta["mean"], dtype=np.float32),
        std=np.asarray(meta["std"], dtype=np.float32),
        feature_names=tuple(meta["feature_names"]),
        class_names=tuple(meta["class_names"]),
        num_classes=int(meta["num_classes"]),
        is_torch=True,
    )


def _load_joblib_checkpoint(checkpoint_path: str | Path) -> LoadedModel:
    """Load a threshold/random_forest/catboost/lightgbm model saved by
    :func:`scripts.05_train_model._fit_non_torch_model` (a plain
    ``joblib.dump`` of the model object -- no self-describing metadata
    sidecar exists for these, unlike the torch checkpoints' embedded
    ``metadata`` dict, so feature names/class names are not recoverable
    from the file alone and must come from the caller's own config."""
    import joblib

    model = joblib.load(checkpoint_path)
    num_classes = int(getattr(model, "num_classes"))
    return LoadedModel(
        model=model,
        mean=np.zeros(0, dtype=np.float32),
        std=np.ones(0, dtype=np.float32),
        feature_names=(),
        class_names=(),
        num_classes=num_classes,
        is_torch=False,
    )


def load_model_for_inference(checkpoint_path: str | Path, device: Optional[str] = None) -> LoadedModel:
    """Load a checkpoint for inference, dispatching on file extension:
    ``.pt`` -> a torch checkpoint written by
    :func:`landscape_change_detection_pipeline.training.train.save_checkpoint`
    (rebuilds its exact architecture from embedded metadata); ``.joblib`` ->
    a plain non-torch model (threshold/random_forest/catboost/lightgbm)
    written by ``scripts/05_train_model.py``'s ``_fit_non_torch_model``.
    """
    path = Path(checkpoint_path)
    if path.suffix == ".joblib":
        return _load_joblib_checkpoint(path)
    return _load_torch_checkpoint(path, device)
