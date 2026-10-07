"""SegFormer (HuggingFace MiT encoder + all-MLP decoder) adapted for
multispectral land-cover input.

Wraps ``transformers.SegformerForSemanticSegmentation``, adapted to this
project's variable spectral-index channel count and 14-class taxonomy (see
``configs/classes.yaml``):

1. **Pretrained weights are used here**, since a shrunk, from-scratch MiT
   encoder is only needed when the input tiles are small enough to collapse
   the standard 4-stage/x32-downsampling schedule to a degenerate final
   stage (e.g. 48x48 tiles). This project's tiles are
   expected to be large enough that the standard, full-size MiT schedule
   stays meaningful throughout, so no stage/stride surgery is needed here,
   and a full pretrained checkpoint (``"nvidia/segformer-{variant}-finetuned-ade-512-512"``)
   can be loaded as a warm start.
2. **No architecture shrinking.** Every hierarchy stage, stride, hidden
   size, attention-head count and sequence-reduction ratio is left at the
   selected variant's standard MiT schedule -- there is no project-specific
   reason to deviate from it here.

`transformers.SegformerForSemanticSegmentation` bakes the input channel
count and output class count directly into its config at construction time
(``num_channels`` sizes the patch-embedding conv, ``num_labels`` sizes the
decode head's final classifier conv -- see ``SegformerConfig``). Adapting a
pretrained checkpoint to this project's channel/class counts is therefore
one ``from_pretrained(..., num_channels=in_channels, num_labels=num_classes,
ignore_mismatched_sizes=True)`` call: HuggingFace loads every weight whose
shape still matches the checkpoint (essentially all of the MiT encoder and
the decode head's internal MLP/fusion layers) and randomly reinitialises
only the two shape-mismatched layers -- the patch-embedding conv and the
classifier conv -- exactly the same "swap only the shape-mismatched layers,
keep the rest pretrained" treatment as ``models/deeplabv3plus.py``, just
expressed through HuggingFace's own config-driven mismatch handling instead
of manual layer surgery.

Design choices:

- **``transformers`` is imported lazily inside ``build_segformer``**, not at
  module top level, for the same reason as ``torchvision`` in
  ``models/deeplabv3plus.py``: importing this module must not require
  ``transformers`` to be installed unless this model type is actually built.
  See ``requirements-torch.txt``'s header.
- **The public interface returns a plain ``(B, num_classes, H, W)`` tensor.**
  SegFormer's all-MLP decode head produces logits at 1/4 of the input
  resolution (the first encoder stage's stride), not the reference's own
  shrunk-encoder 1/4-of-48px figure -- the stride-4 first patch embedding is
  unchanged here, so the ratio is the same regardless of tile size. A thin
  ``SegformerWrapper`` module bilinearly upsamples ``.logits`` back to the
  input's exact spatial size, matching ``models/unet.py`` and
  ``models/deeplabv3plus.py``'s shared ``model(x) -> logits`` interface so
  ``models/registry.py`` can dispatch to any of the three uniformly.
- **Only ``mit-b0`` and ``mit-b2`` are wired up as selectable variants** (out
  of the standard b0-b5 family), matching the two checkpoint sizes this
  project's config (``SegformerConfig.variant`` in
  :mod:`landscape_change_detection_pipeline.config`) is expected to choose between; adding
  b1/b3/b4/b5 later is a one-line addition to ``_VARIANT_CHECKPOINTS``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SegformerWrapper", "build_segformer", "SUPPORTED_VARIANTS"]

#: Maps this project's `variant` names to HuggingFace checkpoint ids. Each is
#: the ADE20K-finetuned SegFormer release for that MiT encoder size.
_VARIANT_CHECKPOINTS: dict[str, str] = {
    "mit-b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "mit-b2": "nvidia/segformer-b2-finetuned-ade-512-512",
}

#: Variant names accepted by `build_segformer`.
SUPPORTED_VARIANTS: tuple[str, ...] = tuple(_VARIANT_CHECKPOINTS)


class SegformerWrapper(nn.Module):
    """Upsamples SegFormer's coarse `.logits` back to input resolution.

    `SegformerForSemanticSegmentation` returns logits at 1/4 of the input's
    spatial size (the first patch embedding's stride); this wrapper
    bilinearly interpolates them back to the exact input `(H, W)`, matching
    the plain-tensor interface `models/unet.py` and `models/deeplabv3plus.py`
    both use.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_h, target_w = x.shape[-2], x.shape[-1]
        logits = self.model(pixel_values=x).logits
        if logits.shape[-2:] == (target_h, target_w):
            return logits
        return F.interpolate(
            logits, size=(target_h, target_w), mode="bilinear", align_corners=False
        )


def build_segformer(
    in_channels: int,
    num_classes: int,
    variant: str = "mit-b0",
    pretrained: bool = True,
    dropout_p: float = 0.1,
    **_ignored,
) -> nn.Module:
    """Construct a SegFormer segmentation model for `in_channels`-band input.

    With `pretrained=True`, loads the ADE20K-finetuned checkpoint for
    `variant` and adapts it in place via
    `SegformerForSemanticSegmentation.from_pretrained(...,
    ignore_mismatched_sizes=True)`: every weight whose shape still matches
    the checkpoint (the MiT encoder's attention/MLP blocks, the decode
    head's fusion layers) is loaded from the pretrained checkpoint, and only
    the two shape-mismatched layers -- the patch-embedding conv (sized for 3
    RGB channels) and the classifier conv (sized for ADE20K's 150 classes)
    -- are randomly reinitialised at `in_channels`/`num_classes`.

    With `pretrained=False`, the model is built from a bare `SegformerConfig`
    with no checkpoint download, entirely offline -- the config-driven
    ``num_channels``/``num_labels`` fields size those same two layers
    directly at construction time (see module docstring), so this path needs
    no separate layer-swap logic.

    Dispatched to by :func:`landscape_change_detection_pipeline.models.registry.build_model`
    for ``model.type == "segformer"``.

    Args:
        in_channels: input feature channel count (spectral indices, not raw
            RGB -- this project has no fixed channel count).
        num_classes: number of land-cover classes (see
            ``configs/classes.yaml``).
        variant: MiT encoder size; one of `SUPPORTED_VARIANTS`
            (``"mit-b0"``, ``"mit-b2"``).
        pretrained: load the ADE20K-pretrained checkpoint before adapting the
            patch embedding and classifier. Set ``False`` for fast,
            network-free construction (e.g. in tests or offline CI).
        dropout_p: dropout probability applied in the decode head and
            throughout the encoder (`classifier_dropout_prob`,
            `hidden_dropout_prob`, `attention_probs_dropout_prob`).

    Returns:
        An ``nn.Module`` whose ``forward(x)`` returns plain
        ``(B, num_classes, H, W)`` logits (see :class:`SegformerWrapper`).

    Raises:
        ValueError: if `variant` is not one of `SUPPORTED_VARIANTS`.
    """
    if variant not in _VARIANT_CHECKPOINTS:
        raise ValueError(
            f"unknown variant {variant!r}; expected one of {SUPPORTED_VARIANTS!r}"
        )

    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    if pretrained:
        checkpoint = _VARIANT_CHECKPOINTS[variant]
        model = SegformerForSemanticSegmentation.from_pretrained(
            checkpoint,
            num_channels=in_channels,
            num_labels=num_classes,
            classifier_dropout_prob=dropout_p,
            hidden_dropout_prob=dropout_p,
            attention_probs_dropout_prob=dropout_p,
            ignore_mismatched_sizes=True,
        )
    else:
        # Bare config, no checkpoint download: sizes the patch embedding and
        # classifier directly from num_channels/num_labels, matching the
        # selected variant's standard MiT schedule (SegformerConfig's
        # defaults already are the mit-b0 schedule; larger variants differ
        # only in hidden_sizes/depths, which pretrained=False has no
        # checkpoint to match anyway, so the architecture-agnostic default
        # schedule is used for every variant in this offline path).
        config = SegformerConfig(
            num_channels=in_channels,
            num_labels=num_classes,
            classifier_dropout_prob=dropout_p,
            hidden_dropout_prob=dropout_p,
            attention_probs_dropout_prob=dropout_p,
        )
        model = SegformerForSemanticSegmentation(config)

    return SegformerWrapper(model)
