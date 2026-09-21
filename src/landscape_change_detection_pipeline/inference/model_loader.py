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
    checkpoint was trained on."""

    def __init__(
        self,
        model: nn.Module,
        mean: np.ndarray,
        std: np.ndarray,
        feature_names: tuple[str, ...],
        class_names: tuple[str, ...],
        num_classes: int,
    ) -> None:
        self.model = model
        self.mean = mean
        self.std = std
        self.feature_names = feature_names
        self.class_names = class_names
        self.num_classes = num_classes


def load_model_for_inference(checkpoint_path: str | Path, device: Optional[str] = None) -> LoadedModel:
    """Load a checkpoint written by
    :func:`landscape_change_detection_pipeline.training.train.save_checkpoint` and rebuild
    its exact architecture, ready for ``model.eval()`` inference."""
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
    )
