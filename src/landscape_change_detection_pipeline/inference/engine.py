"""Sliding-window inference over full scenes.

Patches
are extracted as a single strided batch, pushed through the model in
mini-batches, and accumulated into the full-scene probability volume with
Hann-weighted overlap-averaging (avoids the visible seams a plain average
leaves at patch boundaries, since predictions near a patch edge see less
context). The final origin on each axis is snapped to ``size - patch`` so
the trailing strip is always covered even when ``size - patch`` is not a
multiple of ``stride``. Scenes smaller than one patch are reflect-padded up
to it first (the same padding :class:`landscape_change_detection_pipeline.training.train.PatchDataset`
uses, so training and inference treat undersized scenes identically).

``patch_size``/``stride`` are config-driven here (``config.inference``),
not hardcoded.

Ambiguity tie-break
--------------------
The fixed class-priority fallback (used only when the top-two softmax gap
falls below a calibrated ``ambiguity_threshold``) is ordered so that pixels
close between two land-cover classes err toward the classes most costly to
miss, and so that imaging artifacts (cloud, shadow) never win a real-cover
tie-break. ``ambiguity_threshold=0`` (the default -- no calibration
file yet exists for this project) is exactly plain argmax.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

#: Fixed class-priority fallback for the ambiguity tie-break, most-important
#: first. Water/wetland (open_water, ice_cover, wetland_marsh) lead: they are
#: land-cover features most consequential to miss and are the pair
#: most easily confused with shadow/dark-canopy classes. Disturbance classes
#: (cutblock_harvest, burned_disturbed, bare_ground,
#: built_up_infrastructure) follow: under- or over-mapping human/fire disturbance
#: directly skews downstream change-detection reads. Snow, rock/alpine, and alpine
#: tundra come next (unambiguous but high-elevation/seasonal context worth
#: resolving deterministically), then the vegetation classes in order of
#: successional/structural distinctiveness, and finally the two
#: imaging-artifact classes (cloud, shadow) -- see class_config.ClassDef.change_eligible
#: for why these two are excluded from land-cover comparisons entirely; here
#: they are ranked last so a real land-cover class always wins a close tie
#: against them.
DEFAULT_CLASS_PRIORITY_ORDER: tuple[int, ...] = (
    5,   # open_water
    9,   # ice_cover
    4,   # wetland_marsh
    6,   # cutblock_harvest
    7,   # burned_disturbed
    12,  # bare_ground
    13,  # built_up_infrastructure
    8,   # snow_cover
    10,  # rock_alpine_bare
    11,  # alpine_tundra
    3,   # cultivated_agriculture
    2,   # grassland_herbaceous
    1,   # shrub_early_regrowth
    0,   # forest
    14,  # cloud
    15,  # shadow
)

DEFAULT_PATCH_SIZE = 256
DEFAULT_STRIDE = 128
DEFAULT_BATCH_SIZE = 32

#: Per-scene inference status codes, written alongside the class map so a
#: resumed run can skip work already done and a caller can distinguish "ran,
#: produced nothing useful" from "not attempted".
STATUS_OK = "ok"
STATUS_SKIPPED_PRECHECK = "skip (failed precheck)"
STATUS_SKIPPED_EXISTS = "skip (already done)"


def patch_origins(size: int, patch: int, stride: int) -> list[int]:
    """Top-left offsets tiling one axis, always including the final edge patch."""
    if size <= patch:
        return [0]
    origins = list(range(0, size - patch + 1, stride))
    if origins[-1] != size - patch:
        origins.append(size - patch)
    return origins


def _hann_weights(patch: int) -> np.ndarray:
    """Separable raised-cosine weights tapering each patch towards its edges."""
    window = np.hanning(patch + 2)[1:-1]
    weights = np.outer(window, window).astype(np.float32)
    return np.maximum(weights, 1e-6)


def apply_priority_rule(
    probabilities: np.ndarray,
    ambiguity_threshold: float,
    priority_order: tuple[int, ...],
) -> np.ndarray:
    """Assign classes by argmax, except where the top-two probability gap is
    below ``ambiguity_threshold`` -- there, the highest-priority class among
    the pixel's top-two candidates wins instead of an arbitrary argmax
    tie-break."""
    num_classes = probabilities.shape[0]
    sorted_idx = np.argsort(-probabilities, axis=0)
    top1 = sorted_idx[0]
    top2 = sorted_idx[1] if num_classes > 1 else sorted_idx[0]
    top1_p = np.take_along_axis(probabilities, top1[None, :, :], axis=0)[0]
    top2_p = np.take_along_axis(probabilities, top2[None, :, :], axis=0)[0]
    gap = top1_p - top2_p

    rank = np.full(num_classes, num_classes, dtype=np.int32)
    for order_index, class_id in enumerate(priority_order):
        if 0 <= class_id < num_classes:
            rank[class_id] = order_index

    ambiguous = gap < ambiguity_threshold
    top1_rank = rank[top1]
    top2_rank = rank[top2]
    resolved = np.where(top1_rank <= top2_rank, top1, top2)
    return np.where(ambiguous, resolved, top1).astype(np.int64)


def probabilities_to_classes(
    probabilities: np.ndarray,
    ambiguity_threshold: float = 0.0,
    priority_order: tuple[int, ...] = DEFAULT_CLASS_PRIORITY_ORDER,
) -> np.ndarray:
    """Reduce a ``(num_classes, H, W)`` probability volume to a class-index map."""
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if ambiguity_threshold and ambiguity_threshold > 0:
        return apply_priority_rule(probabilities, float(ambiguity_threshold), priority_order)
    return np.argmax(probabilities, axis=0).astype(np.int64)


def scene_passes_precheck(features: np.ndarray, min_valid_ratio: float = 0.30) -> bool:
    """Whether a scene has enough finite pixels to be worth running inference on."""
    features = np.asarray(features, dtype=np.float32)
    if features.size == 0:
        return False
    finite = np.all(np.isfinite(features), axis=0)
    return float(finite.mean()) >= min_valid_ratio


def valid_ratio(class_map: np.ndarray, nodata: int) -> float:
    """Share of pixels carrying a real class (not the nodata sentinel)."""
    array = np.asarray(class_map)
    return float((array != nodata).mean()) if array.size else 0.0


def predict_scene(
    model,
    features: np.ndarray,
    num_classes: int,
    mean: np.ndarray,
    std: np.ndarray,
    patch_size: int = DEFAULT_PATCH_SIZE,
    stride: int = DEFAULT_STRIDE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    valid_mask: Optional[np.ndarray] = None,
    return_probabilities: bool = False,
    use_hann_weighting: bool = True,
    ambiguity_threshold: float = 0.0,
    priority_order: tuple[int, ...] = DEFAULT_CLASS_PRIORITY_ORDER,
    nodata: int = 255,
):
    """Run sliding-window inference over a full scene.

    ``features`` is the same ``(C, H, W)`` feature stack
    :mod:`landscape_change_detection_pipeline.features.training_cache` builds for
    training (spectral indices then DEM layers), normalized here with the
    checkpoint's own train-only ``mean``/``std`` -- the same normalization
    :class:`landscape_change_detection_pipeline.training.train.PatchDataset` applies, so
    inference sees exactly the distribution the model was trained on.

    Returns the ``(H, W)`` class-index map (``nodata`` where no finite
    feature/mask covers a pixel), or ``(class_map, probabilities)`` with
    ``return_probabilities``.
    """
    import torch

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError(f"Expected a (C, H, W) feature stack, got {features.shape}")

    n_channels, height, width = features.shape
    pad_h = max(0, patch_size - height)
    pad_w = max(0, patch_size - width)
    if pad_h or pad_w:
        features = np.pad(features, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    _, padded_h, padded_w = features.shape

    finite = np.all(np.isfinite(features), axis=0)
    mean_r = mean.astype(np.float32).reshape(-1, 1, 1)
    std_r = np.maximum(std.astype(np.float32), 1e-6).reshape(-1, 1, 1)
    normalized = np.nan_to_num((features - mean_r) / std_r, nan=0.0, posinf=0.0, neginf=0.0)

    rows = patch_origins(padded_h, patch_size, stride)
    cols = patch_origins(padded_w, patch_size, stride)
    positions = [(r, c) for r in rows for c in cols]

    accumulator = np.zeros((num_classes, padded_h, padded_w), dtype=np.float32)
    weight_sum = np.zeros((padded_h, padded_w), dtype=np.float32)
    weights = (
        _hann_weights(patch_size) if use_hann_weighting else np.ones((patch_size, patch_size), dtype=np.float32)
    )

    model.eval()
    model_device = next(model.parameters()).device

    with torch.inference_mode():
        for start in range(0, len(positions), batch_size):
            chunk = positions[start : start + batch_size]
            batch = np.stack(
                [normalized[:, r : r + patch_size, c : c + patch_size] for r, c in chunk],
                axis=0,
            )
            tensor = torch.from_numpy(batch).to(model_device)
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

            for (r, c), prob in zip(chunk, probabilities):
                accumulator[:, r : r + patch_size, c : c + patch_size] += prob * weights
                weight_sum[r : r + patch_size, c : c + patch_size] += weights

    np.divide(accumulator, np.maximum(weight_sum, 1e-6), out=accumulator, where=weight_sum > 0)

    accumulator = accumulator[:, :height, :width]
    finite = finite[:height, :width]

    class_map = probabilities_to_classes(accumulator, ambiguity_threshold, priority_order).astype(np.uint8)
    class_map_full = np.where(finite, class_map, nodata).astype(np.uint8)
    if valid_mask is not None:
        class_map_full = np.where(np.asarray(valid_mask, dtype=bool), class_map_full, nodata).astype(np.uint8)

    if return_probabilities:
        return class_map_full, accumulator
    return class_map_full


def scene_output_paths(output_root: str | Path, tile_id: str, sensor: str, scene_id: str) -> Path:
    """Where one scene's inference class map is written:
    ``<output_root>/<tile_id>/<sensor>/<scene_id>/class_map.npz``."""
    return Path(output_root) / tile_id / sensor / scene_id / "class_map.npz"


def write_class_map(
    path: str | Path,
    class_map: np.ndarray,
    transform,
    crs_wkt: str,
    nodata: int = 255,
) -> Path:
    """Write one scene's class map plus its georeferencing to a small ``.npz``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        class_map=class_map.astype(np.uint8),
        transform=np.array(list(transform)[:6], dtype=np.float64),
        crs_wkt=np.array(crs_wkt),
        nodata=np.array(nodata, dtype=np.uint8),
    )
    return out


def read_class_map(path: str | Path) -> dict:
    """Read a class map written by :func:`write_class_map`."""
    with np.load(Path(path), allow_pickle=False) as data:
        return {
            "class_map": np.array(data["class_map"]),
            "transform": tuple(float(v) for v in data["transform"]),
            "crs_wkt": str(data["crs_wkt"]),
            "nodata": int(data["nodata"]),
        }
