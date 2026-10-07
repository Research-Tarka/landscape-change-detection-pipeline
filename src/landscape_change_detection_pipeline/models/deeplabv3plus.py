"""DeepLabv3+ (ResNet encoder, ASPP + low-level-feature skip decoder) adapted
for multispectral land-cover input.

Built on ``segmentation_models_pytorch`` (``smp.DeepLabV3Plus``) rather than
torchvision's ``deeplabv3_resnet50/101``: torchvision only ships plain
DeepLabv3 (ASPP head, then a single bilinear upsample straight back to input
resolution, no low-level-feature fusion), not DeepLabv3+ (Chen et al. 2018),
which adds a decoder module that concatenates a shallow, high-resolution
backbone feature map (via a skip connection from early in the encoder) with
the upsampled ASPP output before a final refinement -- this is what recovers
sharp object boundaries that plain v3's single coarse-to-fine upsample
blurs, and is the actual point of "the +". ``smp`` implements this decoder
directly (``smp.decoders.deeplabv3.decoder.DeepLabV3PlusDecoder``), so no
manual layer surgery is needed the way ``models/segformer.py``'s HuggingFace
wrapper avoids it for SegFormer.

Design choices:

- **``in_channels``/``classes`` are constructor arguments**, not a
  post-construction layer swap. Unlike torchvision's pretrained models
  (built for fixed 3-channel RGB input, requiring the "build pretrained,
  then replace the first conv" pattern used in this project's earlier
  torchvision-based version and still used in ``models/segformer.py`` for
  the same reason), ``smp`` builds the encoder's stem at the requested
  channel count directly. ``encoder_weights="imagenet"`` still loads
  ImageNet-pretrained weights into every layer whose shape is unaffected by
  the channel count (all of ``layer1``-``layer4``); only the stem conv is
  randomly initialized, same effective result as the manual-swap pattern,
  achieved without the manual swap.
- **``segmentation_models_pytorch`` is imported lazily inside
  ``build_deeplabv3plus``**, not at module top level, for the same reason as
  ``transformers`` in ``models/segformer.py``: importing this module must
  not require the dependency installed unless this model type is actually
  built. See ``requirements-torch.txt``.
- **The public interface returns a plain ``(B, num_classes, H, W)``
  tensor.** ``smp`` models already return a bare tensor (not a dict the way
  torchvision's segmentation models do), so no unwrapping wrapper is needed
  here, unlike the previous torchvision-based version's ``DeepLabV3Wrapper``.
- **Resolution: unlike the previous torchvision-based version, ``smp``
  requires the input's height/width to be exactly divisible by the
  encoder's ``output_stride`` (16 by default) and raises ``RuntimeError``
  otherwise** -- it does not pad internally the way torchvision's
  ``_SimpleSegmentationModel`` did. :class:`_DivisiblePadWrapper` restores
  that same "any input size just works" contract this project's other
  model types (``unet``, ``segformer``) share: reflect-pads up to the next
  multiple of ``output_stride`` before the forward pass, then crops the
  output logits back to the original size. Reflect padding (not zero
  padding) mirrors the padding convention already used elsewhere for
  undersized inputs (see
  :func:`landscape_change_detection_pipeline.training.train._pad_to_min_size` and
  :func:`landscape_change_detection_pipeline.inference.engine.predict_scene`).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["build_deeplabv3plus", "SUPPORTED_BACKBONES"]

#: Backbone names accepted by `build_deeplabv3plus`, mapped to smp encoder names.
SUPPORTED_BACKBONES: tuple[str, ...] = ("resnet50", "resnet101")


class _DivisiblePadWrapper(nn.Module):
    """Reflect-pads input up to a multiple of ``divisor``, crops the output
    logits back to the original size. See module docstring."""

    def __init__(self, model: nn.Module, divisor: int) -> None:
        super().__init__()
        self.model = model
        self.divisor = int(divisor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2], x.shape[-1]
        pad_h = (-height) % self.divisor
        pad_w = (-width) % self.divisor
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        logits = self.model(x)
        return logits[..., :height, :width]


def build_deeplabv3plus(
    in_channels: int,
    num_classes: int,
    backbone: str = "resnet50",
    pretrained: bool = True,
    dropout_p: float = 0.1,
    **_ignored,
) -> nn.Module:
    """Construct a DeepLabv3+ segmentation model for `in_channels`-band input.

    Built from ``segmentation_models_pytorch.DeepLabV3Plus`` with a ResNet
    encoder, ImageNet-pretrained by default. Unlike plain DeepLabv3 (see
    module docstring), the decoder fuses a shallow, high-resolution encoder
    feature map with the upsampled ASPP output via a skip connection before
    the final classifier -- this is the actual "+" in DeepLabv3+, and the
    reason to prefer it over the plain version for boundary-sensitive
    land-cover classes (e.g. water/wetland edges).

    Dispatched to by :func:`landscape_change_detection_pipeline.models.registry.build_model`
    for ``model.type == "deeplabv3plus"``.

    Args:
        in_channels: input feature channel count (spectral indices, not raw
            RGB -- this project has no fixed channel count).
        num_classes: number of land-cover classes (see
            ``configs/classes.yaml``).
        backbone: ``"resnet50"`` or ``"resnet101"``.
        pretrained: load ImageNet-pretrained encoder weights (all layers
            except the input stem, whose channel count is project-specific).
            Set ``False`` for fast, network-free construction (e.g. in tests
            or offline CI).
        dropout_p: dropout applied in the ASPP module
            (``decoder_aspp_dropout``).

    Returns:
        An ``nn.Module`` whose ``forward(x)`` returns plain
        ``(B, num_classes, H, W)`` logits.

    Raises:
        ValueError: if `backbone` is not one of `SUPPORTED_BACKBONES`.
    """
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"unknown backbone {backbone!r}; expected one of {SUPPORTED_BACKBONES!r}"
        )

    import segmentation_models_pytorch as smp

    model = smp.DeepLabV3Plus(
        encoder_name=backbone,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=num_classes,
        decoder_aspp_dropout=dropout_p,
    )
    return _DivisiblePadWrapper(model, divisor=model.encoder.output_stride)
