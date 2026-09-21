"""Loss terms for multi-class land-cover segmentation.

Domain-agnostic loss terms
(inverse-frequency weighted cross-entropy, directional confusion penalties,
deep-supervision auxiliary loss), with confusion penalties expressed as a
single primitive, ``directional_penalty``, rather than hardcoded,
domain-specific wrappers. This project's 14-class taxonomy has no fixed set of
confusion pairs worth naming in code, and is expected to grow, so
confusion penalties are a single config-driven list of
``(true_class, predicted_class, beta)`` triples (see :class:`ConfusionPenalty`
and :func:`total_penalty`) to be filled in once real confusion patterns are
observed in early training runs.

Class weighting follows `w_k = N_total / (num_classes * N_k)` over the
training corpus, so a class occupying a tenth of the pixels gets ten times
the weight.

`deep_supervision_loss` is a separate, optional term added only when the
model carries auxiliary decoder heads (see
:class:`landscape_change_detection_pipeline.models.unet.UNet`'s ``deep_supervision``). It
reuses the same weighted cross-entropy, so a rare class is weighted
identically at every depth, and it reduces the labels to each head's
resolution by majority vote -- see `downsample_labels` for why neither
averaging nor nearest-neighbour sampling is correct for categorical labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

__all__ = [
    "IGNORE_INDEX",
    "ConfusionPenalty",
    "directional_penalty",
    "weighted_cross_entropy",
    "compute_class_weights",
    "total_penalty",
    "downsample_labels",
    "deep_supervision_loss",
    "DEEP_SUPERVISION_WEIGHTS",
]

#: Label value marking a no-data / unannotated pixel, excluded from every loss.
IGNORE_INDEX: int = 255

#: Loss weight per auxiliary head, coarsest level first. The heads sit at 1/4
#: and 1/2 of the input resolution; the coarser a head is, the less its
#: prediction constrains the output that is actually scored, so it carries
#: the smaller weight.
#:
#: Chosen to be small in total. The two terms sum to 0.5 against the primary
#: loss's 1.0, so the objective's magnitude rises by at most half and an
#: already-tuned learning rate stays in range.
DEEP_SUPERVISION_WEIGHTS: tuple[float, ...] = (0.2, 0.3)


@dataclass(frozen=True)
class ConfusionPenalty:
    """One directional confusion penalty: penalise predicting
    ``predicted_class`` where the ground truth is ``true_class`` (or, with
    ``true_class=None``, wherever the truth is *not* ``predicted_class``).

    Config-driven replacement for the reference repo's nine hardcoded,
    glacier-specific penalty functions -- see the module docstring.
    """

    true_class: Optional[int]
    predicted_class: int
    beta: float = 1.0


def directional_penalty(
    logits: torch.Tensor,
    targets: torch.Tensor,
    true_class: Optional[int],
    predicted_class: int,
    beta: float = 1.0,
    ignore_index: int = IGNORE_INDEX,
    probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Penalise probability mass on `predicted_class` where truth is `true_class`.

    This is the single primitive behind every confusion penalty:

        beta * mean over {pixels with target == true_class} of P(predicted_class)

    Passing `true_class=None` selects every valid pixel whose class is *not*
    `predicted_class`, which is what an "don't over-predict class X" term
    wants.

    Args:
        logits: (N, C, H, W) raw scores.
        targets: (N, H, W) class indices, `ignore_index` for no-data.
        true_class: the ground-truth class to restrict to, or None for
            "any class other than `predicted_class`".
        predicted_class: the class whose probability is penalised.
        beta: penalty weight. Zero returns zero without computing a softmax.
        probs: precomputed `softmax(logits, dim=1)`, to share one softmax
            across several penalties in the same step.

    Returns:
        Scalar tensor. Zero when the weight is zero or no pixel qualifies.
    """
    if beta <= 0.0:
        return logits.new_zeros(())

    valid = targets != ignore_index
    mask = valid & (targets != predicted_class) if true_class is None else valid & (targets == true_class)
    if not torch.any(mask):
        return logits.new_zeros(())

    if probs is None:
        probs = torch.softmax(logits, dim=1)
    return beta * probs[:, predicted_class][mask].mean()


def compute_class_weights(
    class_counts: Mapping[int, int],
    num_classes: int,
) -> list[float]:
    """Inverse-frequency class weights, `w_k = N_total / (num_classes * N_k)`.

    A class with an exactly average share gets weight 1. Absent classes are
    floored at a count of 1 rather than producing an infinite weight.
    """
    total = sum(int(class_counts.get(k, 0)) for k in range(num_classes))
    if total <= 0:
        return [1.0] * num_classes
    return [
        total / max(1, num_classes * int(class_counts.get(k, 0)))
        for k in range(num_classes)
    ]


def weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Class-weighted cross-entropy, averaged over valid pixels only.

    `reduction="mean"` with a `weight` normalises by the sum of weights
    rather than the pixel count, which makes the loss magnitude depend on
    the class mix of each batch. Reducing manually over the valid mask keeps
    the scale comparable from batch to batch, and returns a clean zero for
    an all-no-data batch instead of a NaN.
    """
    valid = targets != ignore_index
    if not torch.any(valid):
        return logits.new_zeros(())

    per_pixel = F.cross_entropy(
        logits,
        targets,
        weight=weight,
        ignore_index=ignore_index,
        reduction="none",
    )
    return per_pixel[valid].mean()


def total_penalty(
    logits: torch.Tensor,
    targets: torch.Tensor,
    penalties: Sequence[ConfusionPenalty],
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Sum of every configured confusion penalty.

    Returns zero without touching the logits when `penalties` is empty or
    every weight is zero. When any are active, one softmax is computed and
    shared across all of them rather than one softmax per term.
    """
    active = [p for p in penalties if p.beta > 0.0]
    if not active:
        return logits.new_zeros(())

    probs = torch.softmax(logits, dim=1)
    total = logits.new_zeros(())
    for penalty in active:
        total = total + directional_penalty(
            logits, targets, penalty.true_class, penalty.predicted_class,
            penalty.beta, ignore_index, probs,
        )
    return total


# -- deep supervision ---------------------------------------------------------


def downsample_labels(
    targets: torch.Tensor,
    size: tuple[int, int],
    num_classes: int,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Reduce a label map to `size` by majority vote within each cell.

    Class indices are categorical, so the two obvious reductions are both
    wrong. Averaging them is meaningless -- the mean of two class indices is
    generally a third class neither pixel belonged to. Nearest-neighbour
    sampling is at least type-correct but throws away every pixel it does
    not land on, which at 1/4 resolution is fifteen of every sixteen, and
    makes the auxiliary target depend on an arbitrary alignment choice
    rather than on the region.

    Majority vote uses all of them: each output cell takes the class holding
    the most pixels in the region that maps to it. Implemented as adaptive
    average pooling over a one-hot encoding, which is exactly a per-class
    count within each cell, followed by an argmax.

    `ignore_index` is handled by counting ignored pixels as their own
    additional channel and competing on equal terms. A cell that is mostly
    no-data becomes `ignore_index` and is skipped by the loss; a cell with a
    real majority class keeps it even if some of its pixels were ignored.
    Neither direction leaks: an ignored pixel can never be promoted into a
    real class it did not hold, and a real class is never suppressed by a
    minority of no-data.

    Ties go to the lowest index among the tied classes, with `ignore_index`
    ranked last so a cell that is exactly half annotated stays annotated.
    """
    if targets.shape[-2:] == torch.Size(size):
        return targets

    valid = targets != ignore_index
    safe = torch.where(valid, targets, torch.zeros_like(targets))

    # (B, num_classes + 1, H, W): one channel per class, plus one for no-data.
    one_hot = F.one_hot(safe.long(), num_classes).permute(0, 3, 1, 2).float()
    one_hot = one_hot * valid.unsqueeze(1).float()
    counts = torch.cat([one_hot, (~valid).unsqueeze(1).float()], dim=1)

    # Average pooling over a one-hot map is the per-class share within each
    # cell, which ranks identically to the count and needs no cell-size term.
    pooled = F.adaptive_avg_pool2d(counts, size)
    winner = pooled.argmax(dim=1)

    return torch.where(
        winner == num_classes,
        torch.full_like(winner, ignore_index),
        winner,
    ).to(targets.dtype)


def deep_supervision_loss(
    aux_logits: "list[torch.Tensor]",
    targets: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    weights: "tuple[float, ...]" = DEEP_SUPERVISION_WEIGHTS,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Weighted cross-entropy over the auxiliary decoder heads.

    Each head is scored against the ground truth reduced to that head's own
    resolution, rather than the head's logits being upsampled to full
    resolution. Upsampling would score the head on detail it never had the
    resolution to represent, which is the opposite of what deep supervision
    is for: the point is to ask each level for a correct prediction *at its
    own scale*, so a coarse level learns coarse structure instead of being
    penalised for missing fine boundaries.

    Uses the same class weighting as the primary loss, so a rare class is
    not weighted one way at the output and another two levels up.

    Args:
        aux_logits: one `(N, C, h, w)` tensor per auxiliary head, in the
            order `weights` describes.
        weights: per-head loss weight. Extra heads beyond its length are
            skipped rather than defaulting to a weight, which would put an
            unstated number into the objective.
    """
    if not aux_logits:
        return targets.new_zeros((), dtype=torch.float32)

    total = aux_logits[0].new_zeros(())
    for logits, head_weight in zip(aux_logits, weights):
        if head_weight <= 0.0:
            continue
        reduced = downsample_labels(
            targets,
            (int(logits.shape[-2]), int(logits.shape[-1])),
            num_classes=int(logits.shape[1]),
            ignore_index=ignore_index,
        )
        total = total + head_weight * weighted_cross_entropy(
            logits, reduced, weight, ignore_index
        )
    return total
