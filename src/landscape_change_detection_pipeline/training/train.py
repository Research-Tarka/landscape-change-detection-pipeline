"""The training loop.

``train_model`` runs one training job end to end for the pixel/patch-based
model types (``unet``, ``deeplabv3plus``, ``segformer`` -- see
:mod:`landscape_change_detection_pipeline.models.registry`): build the model, iterate
epochs of mixed-precision optimisation, evaluate on validation each epoch,
early-stop on the configured mIoU metric, and return the best model with its
metrics and a self-describing checkpoint.

Key documented gotchas this loop handles:

**Best-epoch snapshots are true detached CPU copies.** ``state_dict()``
returns live references to the model's own tensors; copying explicitly
(:func:`_snapshot`) is what makes "best" actually mean best rather than
"whatever the last epoch produced" (continued training would otherwise
silently overwrite an aliased snapshot in place).

**Losses run in float32.** Under autocast, ``logits.float()`` before the
softmax the loss/penalty terms read -- a 14-class softmax loses precision in
float16 exactly where those terms read it.

**Gradient clipping runs after ``scaler.unscale_``.** Clipping before
unscaling would apply the threshold to gradients still multiplied by the AMP
loss scale.

**Early stopping compares a value already rounded to three decimals**
(:func:`landscape_change_detection_pipeline.training.metrics.compute_confusion_metrics`),
preserved deliberately -- it makes patience tighter than it looks.

**Geometric augmentation (flips, 90-degree rotations) defaults to on**: a
rotation-invariance argument specific to normalized spectral ratios does not
apply strongly enough here to justify leaving it off.
"""

from __future__ import annotations

import copy
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from landscape_change_detection_pipeline.config import (
    ModelConfig,
    RadiometricAugmentationConfig,
    TrainingConfig,
)
from landscape_change_detection_pipeline.models.registry import build_model
from landscape_change_detection_pipeline.training.dataset import SceneRecord
from landscape_change_detection_pipeline.training.losses import (
    ConfusionPenalty,
    IGNORE_INDEX,
    compute_class_weights,
    deep_supervision_loss,
    soft_dice_loss,
    total_penalty,
    weighted_cross_entropy,
)
from landscape_change_detection_pipeline.training.metrics import (
    compute_confusion_metrics,
    confusion_from_predictions,
    select_main_iou_metric,
)

__all__ = [
    "TrainingResult",
    "PatchDataset",
    "seed_everything",
    "resolve_device",
    "checkpoint_metadata",
    "save_checkpoint",
    "load_checkpoint",
    "train_model",
]

#: Model types this loop can train. threshold/random_forest/catboost fit
#: differently (not epoch/gradient-based) and are out of scope here.
TORCH_MODEL_TYPES: tuple[str, ...] = ("unet", "deeplabv3plus", "segformer")


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG that can affect a run, before anything consumes one.

    Covers
    Python's ``random``, NumPy, torch (CPU and CUDA), and ``PYTHONHASHSEED``
    for spawned workers. ``deterministic=True`` pins cuDNN to deterministic
    algorithms (opt-in: it costs throughput).
    """
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def resolve_device(requested: Optional[str] = None) -> torch.device:
    """Resolve the training device, falling back to CPU with a warning.

    Silently running on CPU when CUDA was asked for turns a short run into a
    much longer one, so the fallback is announced rather than silent.
    """
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            print("[train] CUDA requested but unavailable; falling back to CPU")
            return torch.device("cpu")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """Detached CPU copy of the model's parameters and buffers.

    ``copy=True`` is the whole point: without it this aliases the live
    tensors and the snapshot is silently destroyed by the next optimizer
    step. See the module docstring.
    """
    return {key: value.detach().to("cpu", copy=True) for key, value in model.state_dict().items()}


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the original module behind a ``torch.compile`` wrapper.

    Compiled models prefix every ``state_dict`` key with ``_orig_mod.``,
    which makes their checkpoints unloadable into an uncompiled model.
    """
    return getattr(model, "_orig_mod", model)


# -- dataset ------------------------------------------------------------


class PatchDataset(Dataset):
    """Random fixed-size patches drawn from a set of annotated scenes.

    Each scene's cached ``(C, H, W)`` feature stack and ``(H, W)`` label map
    (see :mod:`landscape_change_detection_pipeline.features.training_cache`) are loaded
    once into memory, and every ``__getitem__`` draws one random
    ``patch_size``-square window, over scenes then uniformly over
    position within the chosen scene. Scene draw is uniform by default, or
    weighted per :func:`scene_sampling_weights` when ``scene_weights`` is
    given (see :class:`landscape_change_detection_pipeline.config.SceneOversamplingConfig`).
    Scenes smaller than ``patch_size`` in either dimension are reflect-padded
    up to it before any patch is drawn (mirrors
    :mod:`landscape_change_detection_pipeline.inference.engine`'s undersized-scene
    handling, so training and inference treat small scenes the same way).

    Normalisation (``mean``/``std``, train-only per
    :func:`landscape_change_detection_pipeline.training.dataset.compute_mean_std`) is
    applied here rather than at cache-build time, so the same cached
    ``.npz`` can be reused across runs with different splits/statistics.

    ``epoch_length`` decouples "one epoch" from "one pass over scenes" --
    with few annotated scenes and many possible patches per scene, a fixed
    number of patches per epoch (default: scene count) gives a stable,
    comparable epoch length as the corpus grows.

    ``__getitem__`` returns a ``(features, labels, loss_weight)`` triple, not
    a pair -- ``loss_weight`` is a scalar tensor scaling this patch's
    contribution to the loss (1.0 for every real-annotation scene; see
    :func:`train_model`'s use of it, and ``PseudoLabelConfig`` for the one
    caller that sets it below 1.0 for pseudo-labeled scenes).
    """

    def __init__(
        self,
        records: list[SceneRecord],
        mean: np.ndarray,
        std: np.ndarray,
        patch_size: int,
        augment_flips: bool = True,
        augment_rotate90: bool = True,
        radiometric_augmentation: Optional[RadiometricAugmentationConfig] = None,
        num_spectral_channels: Optional[int] = None,
        epoch_length: Optional[int] = None,
        seed: int = 0,
        scene_weights: Optional[list[float]] = None,
        scene_loss_weight: float = 1.0,
    ) -> None:
        if not records:
            raise ValueError("PatchDataset requires at least one scene record")

        self.patch_size = int(patch_size)
        self.mean = mean.astype(np.float32).reshape(-1, 1, 1)
        self.std = np.maximum(std.astype(np.float32), 1e-6).reshape(-1, 1, 1)
        self.augment_flips = augment_flips
        self.augment_rotate90 = augment_rotate90
        self.radiometric_augmentation = radiometric_augmentation
        self.num_spectral_channels = num_spectral_channels
        self.scene_loss_weight = float(scene_loss_weight)
        self._rng = random.Random(seed)
        self._noise_rng = np.random.default_rng(seed)

        # Kept at the cache's own dtype (float16 features, uint8/int labels)
        # rather than upcast to float32/int64 here -- every DataLoader
        # worker on Windows (no fork; each worker is a fresh process that
        # unpickles its own copy of this Dataset) ends up holding one full
        # copy of every scene it was given, so the upcast dtype is what
        # actually sets peak RAM times num_workers. Patches are cast to
        # float32/int64 individually in __getitem__ instead, where the
        # per-item cost is negligible next to the RAM this saves.
        # Scenes are memory-mapped on demand (see _LazyScenes): the Dataset
        # pickled to each Windows DataLoader worker only holds the records,
        # and the OS page cache is shared by all workers, so RAM no longer
        # scales with corpus size or num_workers.
        self.scenes = _LazyScenes(sorted(records, key=lambda r: r.sort_key), self.patch_size)
        self._num_scenes = len(self.scenes)

        if scene_weights is not None:
            if len(scene_weights) != self._num_scenes:
                raise ValueError(
                    f"scene_weights has {len(scene_weights)} entries but {self._num_scenes} "
                    f"records were loaded -- must be one weight per record, same order."
                )
            self._scene_indices = list(range(self._num_scenes))
            self._scene_cum_weights = list(np.cumsum(np.asarray(scene_weights, dtype=np.float64)))
        else:
            self._scene_indices = None
            self._scene_cum_weights = None

        self.epoch_length = int(epoch_length) if epoch_length else self._num_scenes

    def __len__(self) -> int:
        return self.epoch_length

    def _draw_scene_index(self) -> int:
        if self._scene_cum_weights is None:
            return self._rng.randrange(self._num_scenes)
        total = self._scene_cum_weights[-1]
        target = self._rng.random() * total
        # bisect manually: self._num_scenes is small (annotated-scene corpus
        # size), so a linear scan over cumulative weights is not worth
        # importing bisect for.
        for index, cum in enumerate(self._scene_cum_weights):
            if target <= cum:
                return index
        return self._num_scenes - 1

    def _apply_radiometric_augmentation(self, patch_features: np.ndarray) -> np.ndarray:
        cfg = self.radiometric_augmentation
        if cfg is None or not cfg.enabled:
            return patch_features

        n_spectral = self.num_spectral_channels
        if n_spectral is None or n_spectral <= 0:
            return patch_features

        spectral = patch_features[:n_spectral]
        rest = patch_features[n_spectral:]

        # One contrast/brightness draw per patch, shared across its spectral
        # channels -- a real sensor's brightness/contrast shift affects every
        # band together, not each independently.
        if cfg.contrast_std > 0.0:
            contrast = 1.0 + self._rng.gauss(0.0, cfg.contrast_std)
            channel_mean = spectral.mean(axis=(1, 2), keepdims=True)
            spectral = (spectral - channel_mean) * contrast + channel_mean
        if cfg.brightness_std > 0.0:
            spectral = spectral + self._rng.gauss(0.0, cfg.brightness_std)
        if cfg.noise_std > 0.0:
            # A persistent generator instead of a fresh
            # np.random.default_rng(...) per patch -- PCG64 construction has
            # real fixed overhead, paid here once per patch per epoch across
            # every sample, which showed up as CPU time competing with data
            # loading. Changes the exact noise sequence for a given seed
            # (not bit-reproducible against caches from before this), but is
            # the same i.i.d. Gaussian distribution.
            noise = self._noise_rng.normal(0.0, cfg.noise_std, size=spectral.shape).astype(np.float32)
            spectral = spectral + noise

        return np.concatenate([spectral.astype(np.float32), rest], axis=0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scene_index = self._draw_scene_index()
        features, labels = self.scenes[scene_index]

        _, height, width = features.shape
        y0 = self._rng.randint(0, height - self.patch_size)
        x0 = self._rng.randint(0, width - self.patch_size)
        # Views only through flip/rotate -- copying once at the end (via
        # ascontiguousarray) instead of after every transform avoids
        # materializing up to three full patch copies per item, which
        # matters here since this runs once per patch per epoch across
        # every worker process.
        patch_features = features[:, y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        patch_labels = labels[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]

        if self.augment_flips:
            if self._rng.random() < 0.5:
                patch_features = patch_features[:, :, ::-1]
                patch_labels = patch_labels[:, ::-1]
            if self._rng.random() < 0.5:
                patch_features = patch_features[:, ::-1, :]
                patch_labels = patch_labels[::-1, :]
        if self.augment_rotate90:
            k = self._rng.randrange(4)
            if k:
                patch_features = np.rot90(patch_features, k, axes=(1, 2))
                patch_labels = np.rot90(patch_labels, k, axes=(0, 1))

        # Cast off the cache's storage dtype (float16/uint8) here, on the
        # small patch rather than the full scene -- see this class's
        # __init__ comment on why scenes are kept at their cache dtype.
        # float32 also avoids float16 precision loss in the mean/contrast
        # arithmetic below.
        patch_features = np.ascontiguousarray(patch_features, dtype=np.float32)
        patch_labels = np.ascontiguousarray(patch_labels, dtype=np.int64)

        # Radiometric jitter runs on raw (pre-normalization) feature values,
        # in the same units channel mean/std were computed over, so
        # contrast_std/brightness_std/noise_std read as fractions of the
        # corpus's own natural spread rather than needing a different scale
        # per channel.
        patch_features = self._apply_radiometric_augmentation(patch_features)

        normalized = (patch_features - self.mean) / self.std
        # A patch can still land on a NaN region (DEM reprojection halo, a
        # degenerate spectral-index ratio) even after corpus-level mean/std
        # excluded NaNs from the statistics themselves -- zero out any
        # remaining non-finite value here (post-normalization, so 0.0 lands
        # at each channel's own mean) rather than letting it reach the model
        # and poison the loss with a NaN gradient.
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        return (
            torch.from_numpy(normalized.astype(np.float32)),
            torch.from_numpy(patch_labels.astype(np.int64)),
            torch.tensor(self.scene_loss_weight, dtype=torch.float32),
        )


class GridPatchDataset(Dataset):
    """Every ``patch_size``-square tile of every scene, covering each scene
    fully instead of drawing one random window per scene per epoch.

    Used for validation: :class:`PatchDataset`'s one-random-patch-per-scene
    sampling means a class confined to a small part of a val scene can be
    absent from the window drawn at any given epoch, so the epoch's
    ``val_miou_macro`` silently skips it (see the ``compute_confusion_metrics``
    docstring -- a class with an empty union is excluded from the mean rather
    than scored, so a class that is never sampled looks the same as a class
    that does not exist in the corpus). Tiling deterministically over a grid
    with ``stride`` <= ``patch_size`` (default: non-overlapping) instead
    means every val pixel is scored exactly once (or, with overlap, at
    least once), so the reported score matches the full-scene evaluation
    used for the final checkpoint metrics.

    No augmentation is applied -- this mirrors :class:`PatchDataset` with
    ``augment_flips=augment_rotate90=False``, which is also how the val
    dataset is already built.
    """

    def __init__(
        self,
        records: list[SceneRecord],
        mean: np.ndarray,
        std: np.ndarray,
        patch_size: int,
        stride: Optional[int] = None,
    ) -> None:
        if not records:
            raise ValueError("GridPatchDataset requires at least one scene record")

        self.patch_size = int(patch_size)
        self.stride = int(stride) if stride else self.patch_size
        self.mean = mean.astype(np.float32).reshape(-1, 1, 1)
        self.std = np.maximum(std.astype(np.float32), 1e-6).reshape(-1, 1, 1)

        # Memory-mapped on demand, like PatchDataset (see _LazyScenes).
        self.scenes = _LazyScenes(sorted(records, key=lambda r: r.sort_key), self.patch_size)
        self._index: list[tuple[int, int, int]] = []  # (scene_index, y0, x0)
        for scene_index in range(len(self.scenes)):
            _, height, width = self.scenes[scene_index][0].shape
            ys = list(range(0, height - self.patch_size, self.stride)) + [height - self.patch_size]
            xs = list(range(0, width - self.patch_size, self.stride)) + [width - self.patch_size]
            for y0 in sorted(set(ys)):
                for x0 in sorted(set(xs)):
                    self._index.append((scene_index, y0, x0))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scene_index, y0, x0 = self._index[index]
        features, labels = self.scenes[scene_index]

        patch_features = features[:, y0 : y0 + self.patch_size, x0 : x0 + self.patch_size].astype(np.float32)
        patch_labels = labels[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size].astype(np.int64)

        normalized = (patch_features - self.mean) / self.std
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        return (
            torch.from_numpy(normalized.astype(np.float32)),
            torch.from_numpy(patch_labels.astype(np.int64)),
            torch.tensor(1.0, dtype=torch.float32),
        )


class _LazyScenes:
    """Sequence of ``(features, labels)`` per scene record, opened lazily as
    read-only memory maps (see ``load_scene_cache(mmap=True)``).

    Only the records are pickled to DataLoader workers (the map cache is
    dropped in ``__getstate__``); each process opens the files it touches,
    so reading a patch faults in just that patch's pages and the OS page
    cache is shared across workers. Scenes smaller than ``min_size`` are
    padded (an in-RAM copy, only for those rare small scenes).
    """

    _MAX_OPEN = 64

    def __init__(self, records: list[SceneRecord], min_size: int) -> None:
        self._records = records
        self._min_size = int(min_size)
        self._open: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self._records)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_open"] = {}
        return state

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        scene = self._open.get(index)
        if scene is None:
            from landscape_change_detection_pipeline.features.training_cache import load_scene_cache

            if len(self._open) >= self._MAX_OPEN:
                self._open.pop(next(iter(self._open)))
            cache = load_scene_cache(
                self._records[index].cache_dir, mmap=True, label_remap=self._records[index].label_remap,
            )
            scene = self._open[index] = _pad_to_min_size(cache["features"], cache["labels"], self._min_size)
        return scene


def _pad_to_min_size(features: np.ndarray, labels: np.ndarray, min_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Reflect-pad a scene up to at least ``min_size`` in each spatial dim."""
    _, height, width = features.shape
    pad_h = max(0, min_size - height)
    pad_w = max(0, min_size - width)
    if pad_h == 0 and pad_w == 0:
        return features, labels

    features = np.pad(features, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    labels = np.pad(labels, ((0, pad_h), (0, pad_w)), mode="reflect")
    return features, labels


# -- result / checkpoint --------------------------------------------------


@dataclass
class TrainingResult:
    """A finished training run: the model plus everything worth recording."""

    model: nn.Module
    best_state: dict[str, Any]
    history: dict[str, list[float]]
    class_counts: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    feature_names: tuple[str, ...]
    best_epoch: int
    stopped_epoch: int
    metrics: dict[str, Any] = field(default_factory=dict)


def checkpoint_metadata(
    model_cfg: ModelConfig,
    in_channels: int,
    num_classes: int,
    class_names: tuple[str, ...],
    feature_names: tuple[str, ...],
    mean: np.ndarray,
    std: np.ndarray,
    num_sensors: int = 1,
) -> dict[str, Any]:
    """Self-describing checkpoint metadata: everything the inference engine
    needs to rebuild the exact architecture and preprocessing a checkpoint
    was trained with, without the caller having to know or guess it.
    """
    unet_cfg = model_cfg.unet
    return {
        "model_type": model_cfg.type,
        "in_channels": int(in_channels),
        "num_classes": int(num_classes),
        "num_sensors": int(num_sensors),
        "class_names": list(class_names),
        "feature_names": list(feature_names),
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
        "unet": {
            "base_ch": unet_cfg.base_ch,
            "dropout_p": unet_cfg.dropout_p,
            "use_spatial_context": unet_cfg.use_spatial_context,
            "use_sensor_film": unet_cfg.use_sensor_film,
            "norm_type": unet_cfg.norm_type,
            "bottleneck_attention": unet_cfg.bottleneck_attention,
            "bottleneck_attention_heads": unet_cfg.bottleneck_attention_heads,
            "deep_supervision": unet_cfg.deep_supervision,
        }
        if model_cfg.type == "unet"
        else None,
        "deeplabv3plus": {
            "backbone": model_cfg.deeplabv3plus.backbone,
            "dropout_p": model_cfg.deeplabv3plus.dropout_p,
        }
        if model_cfg.type == "deeplabv3plus"
        else None,
        "segformer": {
            "variant": model_cfg.segformer.variant,
            "dropout_p": model_cfg.segformer.dropout_p,
        }
        if model_cfg.type == "segformer"
        else None,
    }


def save_checkpoint(path: str | Path, model: nn.Module, metadata: dict[str, Any]) -> Path:
    """Write a checkpoint carrying both the weights and self-describing
    metadata (see :func:`checkpoint_metadata`), so the inference engine
    never has to guess what it is loading."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": _unwrap(model).state_dict(), "metadata": metadata}, out)
    return out


def load_checkpoint(path: str | Path, device: Optional[torch.device] = None) -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint`."""
    return torch.load(Path(path), map_location=device or "cpu", weights_only=False)


def rebuild_model_from_checkpoint(checkpoint: dict[str, Any]) -> nn.Module:
    """Rebuild the exact architecture a checkpoint was trained with, from its
    embedded metadata, and load the saved weights."""
    from landscape_change_detection_pipeline.models.deeplabv3plus import build_deeplabv3plus
    from landscape_change_detection_pipeline.models.segformer import build_segformer
    from landscape_change_detection_pipeline.models.unet import build_unet

    meta = checkpoint["metadata"]
    model_type = meta["model_type"]

    if model_type == "unet":
        cfg = meta["unet"]
        model = build_unet(
            in_channels=meta["in_channels"],
            num_classes=meta["num_classes"],
            base_ch=cfg["base_ch"],
            dropout_p=cfg["dropout_p"],
            use_spatial_context=cfg["use_spatial_context"],
            num_sensors=meta["num_sensors"],
            use_sensor_film=cfg["use_sensor_film"],
            norm_type=cfg["norm_type"],
            bottleneck_attention=cfg["bottleneck_attention"],
            bottleneck_attention_heads=cfg["bottleneck_attention_heads"],
            deep_supervision=cfg["deep_supervision"],
        )
    elif model_type == "deeplabv3plus":
        cfg = meta["deeplabv3plus"]
        model = build_deeplabv3plus(
            in_channels=meta["in_channels"],
            num_classes=meta["num_classes"],
            backbone=cfg["backbone"],
            pretrained=False,
            dropout_p=cfg["dropout_p"],
        )
    elif model_type == "segformer":
        cfg = meta["segformer"]
        model = build_segformer(
            in_channels=meta["in_channels"],
            num_classes=meta["num_classes"],
            variant=cfg["variant"],
            pretrained=False,
            dropout_p=cfg["dropout_p"],
        )
    else:
        raise ValueError(f"Checkpoint model_type={model_type!r} is not a torch model this loader supports.")

    model.load_state_dict(checkpoint["state_dict"])
    return model


# -- training loop ---------------------------------------------------------


def _build_loaders(
    train_records: list[SceneRecord],
    val_records: list[SceneRecord],
    mean: np.ndarray,
    std: np.ndarray,
    config: TrainingConfig,
    device: torch.device,
    num_spectral_channels: Optional[int] = None,
    class_counts: Optional[np.ndarray] = None,
    pseudo_label_records: Optional[list[SceneRecord]] = None,
    full_val: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Build the train/val ``DataLoader``s.

    ``num_spectral_channels`` gates :class:`RadiometricAugmentationConfig`
    (see its docstring -- only the spectral-index channels are jittered).
    ``class_counts``, when given alongside
    ``config.scene_oversampling.enabled``, drives per-scene sampling weights
    (see :func:`landscape_change_detection_pipeline.training.dataset.scene_sampling_weights`).
    ``pseudo_label_records``, when non-empty and ``config.pseudo_label.enabled``,
    is mixed into the training set as a second, separately-weighted
    :class:`PatchDataset` via ``ConcatDataset`` -- see
    :class:`landscape_change_detection_pipeline.config.PseudoLabelConfig`.

    ``full_val``: ``True`` (default) scores every pixel of every val scene
    each epoch via :class:`GridPatchDataset`, so ``val_miou_macro`` matches
    the full-scene evaluation used for the final checkpoint metrics -- see
    that class's docstring for why a single random patch per scene can
    silently miss a class confined to part of a scene. HPO passes ``False``
    for its search trials, where only the *relative* ranking between
    hyperparameter configs matters and this runs once per epoch across
    potentially thousands of trial-epochs -- there, the cheap single-random-
    patch :class:`PatchDataset` is used instead, and only the final refit
    (the winning config, retrained at full length) gets the full grid.
    """
    from torch.utils.data import ConcatDataset

    scene_weights = None
    if config.scene_oversampling.enabled and class_counts is not None:
        from landscape_change_detection_pipeline.training.dataset import (
            collect_class_pixel_counts_by_scene,
            scene_sampling_weights,
        )

        num_classes = len(class_counts)
        counts_by_scene = collect_class_pixel_counts_by_scene(train_records, num_classes)
        scene_weights = scene_sampling_weights(
            train_records,
            counts_by_scene,
            class_counts,
            rare_class_threshold=config.scene_oversampling.rare_class_threshold,
            boost_factor=config.scene_oversampling.boost_factor,
            rare_classes=config.scene_oversampling.rare_classes,
            presence_min_pixels=config.scene_oversampling.presence_min_pixels,
        )

    train_dataset: Dataset = PatchDataset(
        train_records,
        mean,
        std,
        patch_size=config.patch_size,
        augment_flips=config.augment_flips,
        augment_rotate90=config.augment_rotate90,
        radiometric_augmentation=config.radiometric_augmentation,
        num_spectral_channels=num_spectral_channels,
        seed=config.seed,
        scene_weights=scene_weights,
    )

    if config.pseudo_label.enabled and pseudo_label_records:
        pseudo_dataset = PatchDataset(
            pseudo_label_records,
            mean,
            std,
            patch_size=config.patch_size,
            augment_flips=config.augment_flips,
            augment_rotate90=config.augment_rotate90,
            radiometric_augmentation=config.radiometric_augmentation,
            num_spectral_channels=num_spectral_channels,
            epoch_length=max(1, round(len(train_records) * config.pseudo_label.pseudo_label_weight)),
            seed=config.seed + 1,
            scene_loss_weight=config.pseudo_label.pseudo_label_loss_weight,
        )
        train_dataset = ConcatDataset([train_dataset, pseudo_dataset])

    val_dataset: Dataset
    if full_val:
        val_dataset = GridPatchDataset(
            val_records,
            mean,
            std,
            patch_size=config.patch_size,
        )
    else:
        val_dataset = PatchDataset(
            val_records,
            mean,
            std,
            patch_size=config.patch_size,
            augment_flips=False,
            augment_rotate90=False,
            seed=config.seed,
        )

    loader_kwargs: dict[str, Any] = {
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: Optional[torch.Tensor],
    use_amp: bool,
    num_classes: int,
    pred_remap: Optional[torch.Tensor] = None,
) -> tuple[float, np.ndarray]:
    """Validation pass: mean loss and the pooled confusion matrix.

    ``pred_remap`` (the class-merge lookup table as a tensor) folds
    predictions of a merged-away class onto its target, so a stray prediction
    of it cannot create a zero-IoU class that drags the macro mean down.

    ``inference_mode`` rather than ``no_grad`` additionally skips autograd's
    version-counter bookkeeping. The confusion matrix accumulates on-device
    and is copied to host once, not once per batch.
    """
    model.eval()
    total_loss = 0.0
    total_items = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    for x, y, _sample_weight in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits = model(x)
            loss = weighted_cross_entropy(logits.float(), y, class_weights)

        total_loss += float(loss.item()) * x.size(0)
        total_items += x.size(0)
        predicted = logits.argmax(dim=1)
        if pred_remap is not None:
            predicted = pred_remap[predicted]
        confusion += confusion_from_predictions(predicted, y, num_classes, IGNORE_INDEX)

    mean_loss = total_loss / max(total_items, 1)
    return mean_loss, confusion.cpu().numpy()


def train_model(
    config: TrainingConfig,
    model_cfg: ModelConfig,
    train_records: list[SceneRecord],
    val_records: list[SceneRecord],
    mean: np.ndarray,
    std: np.ndarray,
    class_counts: np.ndarray,
    num_classes: int,
    class_names: tuple[str, ...],
    feature_names: tuple[str, ...],
    num_sensors: int = 1,
    device: Optional[str] = None,
    progress: bool = True,
    trial: Optional[Any] = None,
    num_spectral_channels: Optional[int] = None,
    pseudo_label_records: Optional[list[SceneRecord]] = None,
) -> TrainingResult:
    """Train one torch model (unet/deeplabv3plus/segformer) end to end.

    AdamW + cosine annealing, mixed precision with float32 loss computation,
    gradient clipping after ``scaler.unscale_``, early stopping on
    ``config.main_iou_metric`` with a true detached-CPU best-epoch snapshot.
    See the module docstring for why each of these matters.

    ``trial``, when given (an ``optuna.trial.Trial`` from
    :func:`landscape_change_detection_pipeline.training.hpo.run_trials`), turns on
    Optuna's mid-training pruning: the main IoU metric is reported to it
    after every epoch (``trial.report(score, step=epoch)``), and
    ``optuna.TrialPruned`` is raised the moment ``trial.should_prune()``
    says so -- the only way ``hpo.HpoConfig.pruner="hyperband"`` (or
    ``"median"``) actually cuts a bad trial short, since without a report
    Optuna's pruner has nothing to judge and every trial silently runs to
    ``trial_epochs`` regardless of the configured pruner. ``None`` (the
    default, used by every non-HPO caller) reports nothing and never prunes.

    ``num_spectral_channels`` gates ``config.radiometric_augmentation`` (see
    :class:`landscape_change_detection_pipeline.config.RadiometricAugmentationConfig`).
    ``pseudo_label_records``, when given alongside ``config.pseudo_label.enabled``,
    mixes pseudo-labeled scenes into training at reduced sampling/loss weight
    (see :class:`landscape_change_detection_pipeline.config.PseudoLabelConfig` and
    ``scripts/05b_generate_pseudo_labels.py``).

    Validation coverage is tied to ``trial``: an HPO search trial (``trial``
    given) scores val with one random patch per scene per epoch -- cheap,
    and sufficient since only the *relative* ranking between hyperparameter
    configs matters there, over what can add up to thousands of trial-
    epochs. Every other caller -- ``trial=None``, in particular the final
    refit of the winning HPO config at full ``training.epochs`` -- scores
    every pixel of every val scene each epoch instead (see
    :class:`GridPatchDataset`'s docstring), so its reported
    ``val_miou_macro`` and early-stopping decisions match the full-scene
    evaluation used for the final checkpoint metrics.
    """
    if model_cfg.type not in TORCH_MODEL_TYPES:
        raise ValueError(
            f"train_model only supports {TORCH_MODEL_TYPES}, got model.type={model_cfg.type!r} "
            f"(threshold/random_forest/catboost fit via their own build_<name>().fit(), not this loop)"
        )

    resolved_device = resolve_device(device)
    seed_everything(config.seed, deterministic=config.deterministic)

    if not train_records:
        raise RuntimeError("the training split contains no scenes")
    if not val_records:
        raise RuntimeError("the validation split contains no scenes")

    in_channels = len(feature_names)
    train_loader, val_loader = _build_loaders(
        train_records,
        val_records,
        mean,
        std,
        config,
        resolved_device,
        num_spectral_channels=num_spectral_channels,
        class_counts=class_counts,
        pseudo_label_records=pseudo_label_records,
        full_val=trial is None,
    )

    model = build_model(model_cfg, in_channels=in_channels, num_classes=num_classes, num_sensors=num_sensors)
    model = model.to(resolved_device)

    if config.class_weighting:
        class_weight_values = compute_class_weights(
            {i: int(c) for i, c in enumerate(class_counts)}, num_classes,
            power=config.class_weight_power, max_weight=config.class_weight_max,
        )
    else:
        class_weight_values = [1.0] * num_classes
    class_weights = torch.tensor(class_weight_values, dtype=torch.float32, device=resolved_device)

    from landscape_change_detection_pipeline.classes.class_config import label_remap_table

    remap_table = label_remap_table(class_names, config.class_merge)
    pred_remap = (
        torch.tensor(remap_table, dtype=torch.int64, device=resolved_device) if remap_table is not None else None
    )
    penalties = [
        ConfusionPenalty(true_class=p.true_class, predicted_class=p.predicted_class, beta=p.beta)
        for p in config.confusion_penalties
    ]

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    use_amp = config.amp and resolved_device.type == "cuda"
    scaler = GradScaler(resolved_device.type, enabled=use_amp)

    supports_deep_supervision = model_cfg.type == "unet" and model_cfg.unet.deep_supervision

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_miou": [], "lr": []}
    best_score = -math.inf
    best_state: dict[str, Any] = {}
    best_epoch = 0
    epochs_without_improvement = 0
    last_epoch = 0

    for epoch in range(1, config.epochs + 1):
        started = time.time()
        model.train()
        running_loss = 0.0
        seen = 0

        for x, y, sample_weight in train_loader:
            x = x.to(resolved_device, non_blocking=True)
            y = y.to(resolved_device, non_blocking=True)
            sample_weight = sample_weight.to(resolved_device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=resolved_device.type, enabled=use_amp):
                output = model(x, return_aux=True) if supports_deep_supervision else model(x)
                logits, aux_logits = output if supports_deep_supervision else (output, [])

                logits_f32 = logits.float()
                loss = weighted_cross_entropy(
                    logits_f32, y, class_weights, IGNORE_INDEX, sample_weight=sample_weight,
                    focal_gamma=config.focal_gamma,
                ) + total_penalty(logits_f32, y, penalties, IGNORE_INDEX)
                if config.dice_weight > 0.0:
                    loss = loss + config.dice_weight * soft_dice_loss(logits_f32, y, IGNORE_INDEX)
                if aux_logits:
                    loss = loss + deep_supervision_loss(
                        [t.float() for t in aux_logits], y, class_weights, ignore_index=IGNORE_INDEX
                    )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * x.size(0)
            seen += x.size(0)

        train_loss = running_loss / max(seen, 1)
        val_loss, confusion = evaluate(
            model, val_loader, resolved_device, class_weights, use_amp, num_classes, pred_remap=pred_remap,
        )
        val_metrics = compute_confusion_metrics(confusion, num_classes, class_names=list(class_names))["macro"]

        scheduler.step()

        metric_name, score = select_main_iou_metric(
            config.main_iou_metric,
            miou_macro=val_metrics["miou"],
            miou_weighted=val_metrics["miou_weighted"],
            miou_inv_freq=val_metrics["miou_inv_freq"],
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(score)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if trial is not None:
            import optuna

            trial.report(score, step=epoch)
            if trial.should_prune():
                if progress:
                    print(f"[hpo] trial pruned at epoch {epoch} (val_{metric_name}={score:.4f})")
                raise optuna.TrialPruned()

        last_epoch = epoch
        if progress:
            print(
                f"[epoch {epoch}/{config.epochs}] train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_{metric_name}={score:.4f} "
                f"({time.time() - started:.1f}s)"
            )

        improved = math.isfinite(score) and score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {
                "model": _snapshot(_unwrap(model)),
                "epoch": epoch,
                "metric_name": metric_name,
                "val_metric": score,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "confusion": confusion.copy(),
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            if progress:
                print(f"[train] early stop at epoch {epoch} (patience {config.patience})")
            break

    if not best_state:
        best_state = {
            "model": _snapshot(_unwrap(model)),
            "epoch": last_epoch,
            "metric_name": config.main_iou_metric,
            "val_metric": history["val_miou"][-1] if history["val_miou"] else float("nan"),
            "train_loss": history["train_loss"][-1] if history["train_loss"] else float("nan"),
            "val_loss": history["val_loss"][-1] if history["val_loss"] else float("nan"),
            "confusion": None,
        }
        best_epoch = last_epoch

    # A real restore: the snapshot was a detached copy, so this is not a no-op.
    _unwrap(model).load_state_dict(best_state["model"])

    metrics = (
        compute_confusion_metrics(best_state["confusion"], num_classes, class_names=list(class_names))
        if best_state.get("confusion") is not None
        else {}
    )

    return TrainingResult(
        model=_unwrap(model),
        best_state=best_state,
        history=history,
        class_counts=class_counts,
        mean=mean,
        std=std,
        feature_names=feature_names,
        best_epoch=best_epoch,
        stopped_epoch=last_epoch,
        metrics=metrics,
    )
