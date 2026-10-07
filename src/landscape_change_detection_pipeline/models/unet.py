"""U-Net (no attention gates) for multi-class land-cover segmentation.

The additive attention-gate mechanism is removed entirely rather than
merely disabled: an
``use_attention_gates=False`` ablation branch, where each skip connection
passes the raw encoder feature straight into the decoder's concatenation
(exactly as if ``AttentionGate.forward`` had returned ``x`` unchanged), is
the only branch this project ever needs. Rather than carry the flag and a
permanently-unused ``AttentionGate`` class forward as dead code, the
class is omitted and every skip connection is unconditionally the raw
encoder feature.

Everything else follows the standard design: ``DoubleConv``, per-sensor ``BatchNorm``/
``GroupNorm`` banks (useful here too -- Landsat 5/7/8/9 and Sentinel-2 have
different radiometric characteristics and different native resolutions),
optional ``BottleneckAttention`` self-attention over the bottleneck grid,
optional ``SceneFiLM`` conditioning (sensor / season / geography, each
switched independently) of the encoder and decoder stages, and optional ``deep_supervision`` auxiliary decoder-level logit heads.

``num_classes`` is a required constructor/``build_unet`` argument (no
default baked in): this project's class list is expected to grow (see
``configs/classes.yaml``).

Sharing one model across sensors at 10-30 m/pixel is a resolution
trade-off; if a sensor's scores diverge because of it, the documented
fallback is to fall back to per-sensor metrics first.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from landscape_change_detection_pipeline.features.scene_context import (
    DOY_COLS,
    LATLON_COLS,
    N_SENSORS,
    SENSOR_COL,
)

__all__ = [
    "SceneFiLM",
    "DoubleConv",
    "BottleneckAttention",
    "UNet",
    "build_unet",
    "make_norm",
    "GROUP_NORM_CHANNELS_PER_GROUP",
]

#: Channels per group when `norm_type="group"`. GroupNorm is parameterised by
#: a group *count*, but a fixed count divides the four channel widths here
#: (base_ch/2x/4x/8x) badly at some `base_ch` values. Fixing the group
#: *size* instead and deriving the count keeps the statistic computed over
#: the same number of channels at every depth regardless of `base_ch`.
GROUP_NORM_CHANNELS_PER_GROUP: int = 16


def make_norm(channels: int, norm_type: str = "batch") -> nn.Module:
    """One normalisation layer for `channels` channels.

    `"batch"` is the default. `"group"` substitutes GroupNorm, which
    normalises over channel groups within each sample and so does not depend
    on batch composition at all -- no running mean or variance is kept, and
    eval-time behaviour is identical to train-time behaviour.

    A channel count not divisible by `GROUP_NORM_CHANNELS_PER_GROUP` falls
    back to the largest divisor that is no larger than it, so an unusual
    `base_ch` still builds rather than raising.
    """
    kind = str(norm_type).strip().lower()
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        size = min(GROUP_NORM_CHANNELS_PER_GROUP, channels)
        while size > 1 and channels % size != 0:
            size -= 1
        return nn.GroupNorm(max(1, channels // size), channels)
    raise ValueError(f"unknown norm_type {norm_type!r} (expected 'batch' or 'group')")


def _route_by_sensor(
    x: torch.Tensor,
    sensor_ids: Optional[torch.Tensor],
    banks: nn.ModuleList,
) -> torch.Tensor:
    """Apply each batch item's own normalisation layer from `banks`.

    With a single bank, or with no sensor ids supplied, this is just
    `banks[0](x)` and costs nothing. Otherwise each sensor's slice of the
    batch goes through its own layer.

    Only sensors actually present in the batch are visited, so a batch drawn
    from one sensor costs one BatchNorm call rather than `len(banks)`.
    """
    if len(banks) == 1 or sensor_ids is None:
        return banks[0](x)

    out = torch.empty_like(x)
    for sensor in torch.unique(sensor_ids).tolist():
        index = int(sensor)
        if not 0 <= index < len(banks):
            index = 0
        mask = sensor_ids == sensor
        out[mask] = banks[index](x[mask])
    return out


class SceneFiLM(nn.Module):
    """FiLM conditioning on scene-level context (sensor, season, geography).

    Each of the three inputs is switched on independently (``use_sensor``,
    ``use_doy``, ``use_latlon``) and read from the fixed context layout of
    :mod:`landscape_change_detection_pipeline.features.scene_context`. A small MLP turns
    the selected inputs into one embedding; one linear head per modulated stage
    then produces that stage's per-channel ``(gamma, beta)`` and the stage's
    feature map becomes ``x * (1 + gamma) + beta``.

    The heads start at zero, so an untrained FiLM is the identity and the
    network begins exactly as the unconditioned U-Net would.
    """

    def __init__(
        self,
        channels: Sequence[int],
        use_sensor: bool,
        use_doy: bool,
        use_latlon: bool,
        hidden: int = 64,
        sensor_embed_dim: int = 8,
    ) -> None:
        super().__init__()
        if not (use_sensor or use_doy or use_latlon):
            raise ValueError("SceneFiLM needs at least one of use_sensor / use_doy / use_latlon")
        self.use_sensor, self.use_doy, self.use_latlon = bool(use_sensor), bool(use_doy), bool(use_latlon)
        self.embedding = nn.Embedding(N_SENSORS, sensor_embed_dim) if self.use_sensor else None
        in_dim = (sensor_embed_dim if self.use_sensor else 0) + (2 if self.use_doy else 0) + (2 if self.use_latlon else 0)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.ReLU(inplace=True)
        )
        self.heads = nn.ModuleList(nn.Linear(hidden, 2 * int(c)) for c in channels)
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def embed(self, context: torch.Tensor) -> torch.Tensor:
        """``(B, CONTEXT_DIM)`` scene context -> ``(B, hidden)`` embedding."""
        context = context.float()
        parts = []
        if self.embedding is not None:
            ids = context[:, SENSOR_COL].round().long().clamp(0, N_SENSORS - 1)
            parts.append(self.embedding(ids))
        if self.use_doy:
            parts.append(context[:, DOY_COLS])
        if self.use_latlon:
            parts.append(context[:, LATLON_COLS])
        return self.trunk(torch.cat(parts, dim=1))

    def modulate(self, level: int, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.heads[level](embedding).chunk(2, dim=1)
        return x * (1 + gamma[:, :, None, None].to(x.dtype)) + beta[:, :, None, None].to(x.dtype)


def _check_norm_and_sensors(norm_type: str, num_sensors: int) -> None:
    """Reject the one combination that cannot mean anything.

    The per-sensor bank exists so each sensor accumulates its own running
    mean and variance over its own radiometric distribution. GroupNorm keeps
    no running statistics -- it normalises within each sample at both train
    and eval time -- so a bank of GroupNorms would differ only in their
    affine parameters, which is a per-sensor rescaling wearing a
    normalisation's name.
    """
    if str(norm_type).strip().lower() == "group" and int(num_sensors) > 1:
        raise ValueError(
            "norm_type='group' with num_sensors>1 is unsupported: GroupNorm "
            "keeps no running statistics, so there is nothing for a per-sensor "
            "bank to accumulate. Use norm_type='batch' for per-sensor "
            "normalisation, or num_sensors=1 with GroupNorm."
        )


class DoubleConv(nn.Module):
    """Conv-Norm-ReLU twice, with optional per-sensor normalisation banks.

    `num_sensors=1` gives an ordinary double convolution block.
    `norm_type="group"` is incompatible with `num_sensors > 1`; see
    `_check_norm_and_sensors`.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dropout_p: float = 0.0,
        num_sensors: int = 1,
        norm_type: str = "batch",
    ) -> None:
        super().__init__()
        _check_norm_and_sensors(norm_type, num_sensors)
        self.num_sensors = int(num_sensors)
        self.norm_type = str(norm_type).strip().lower()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.ModuleList(
            make_norm(out_ch, norm_type) for _ in range(num_sensors)
        )
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.ModuleList(
            make_norm(out_ch, norm_type) for _ in range(num_sensors)
        )
        self.dropout = nn.Dropout2d(p=float(dropout_p)) if dropout_p > 0 else None

    def forward(
        self, x: torch.Tensor, sensor_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = F.relu(_route_by_sensor(self.conv1(x), sensor_ids, self.bn1), inplace=True)
        x = F.relu(_route_by_sensor(self.conv2(x), sensor_ids, self.bn2), inplace=True)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


def _sincos_2d(height: int, width: int, channels: int, device, dtype) -> torch.Tensor:
    """``(H*W, C)`` 2-D sine/cosine position code: the first half of the channels
    encodes the row, the second half the column. Computed for any grid size, so
    it works at any patch size (train and inference may differ)."""
    quarter = channels // 4
    freq = 1.0 / (10000.0 ** (torch.arange(quarter, device=device, dtype=torch.float32) / max(quarter, 1)))
    rows = torch.arange(height, device=device, dtype=torch.float32)[:, None] * freq[None]
    cols = torch.arange(width, device=device, dtype=torch.float32)[:, None] * freq[None]
    row_code = torch.cat([rows.sin(), rows.cos()], dim=1)  # (H, 2q)
    col_code = torch.cat([cols.sin(), cols.cos()], dim=1)  # (W, 2q)
    code = torch.cat(
        [row_code[:, None, :].expand(height, width, -1), col_code[None, :, :].expand(height, width, -1)], dim=-1
    )
    if code.shape[-1] < channels:  # channels not divisible by 4: pad with zeros
        code = torch.nn.functional.pad(code, (0, channels - code.shape[-1]))
    return code.reshape(height * width, channels).to(dtype)


class BottleneckAttention(nn.Module):
    """Multi-head self-attention over the bottleneck's spatial positions.

    The encoder's receptive field grows only by convolution and pooling, so
    at the bottleneck every position still summarises a bounded
    neighbourhood. Flattening the `H x W` bottleneck grid into a token
    sequence and running self-attention over it lets any position read any
    other directly.

    Cost: quadratic in the number of tokens. With a 256 px patch and three
    poolings the bottleneck is 32x32 = 1024 tokens (64 only for a 64 px patch).

    Two things make it safe to switch on (an earlier version had neither, and
    lowered the metrics):

    - **Position code.** Plain self-attention is blind to where a token is: it
      only sees a bag of features. A fixed 2-D sine/cosine code is added to the
      queries and keys, so "who is near me" can be learned. Values stay
      position-free.
    - **Zero-initialised gate (ReZero).** The block's output is
      ``tokens + gate * attended`` with ``gate`` starting at 0, so at the start
      of training the layer is exactly the identity and cannot inject noise
      into the decoder; the network opens the gate only if attention helps.

    No feed-forward sublayer: this is a single global-mixing layer, not a full
    transformer block (fewer parameters, less overfitting on a small corpus).
    """

    def __init__(self, channels: int, num_heads: int = 8, dropout_p: float = 0.0) -> None:
        super().__init__()
        heads = max(1, int(num_heads))
        while heads > 1 and channels % heads != 0:
            heads -= 1
        self.num_heads = heads
        self.channels = int(channels)

        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=float(dropout_p),
            batch_first=True,
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(B, C, H, W)` in, the same shape out."""
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        normed = self.norm(tokens)
        qk = normed + _sincos_2d(height, width, channels, normed.device, normed.dtype)[None]
        attended, _ = self.attention(qk, qk, normed, need_weights=False)
        tokens = tokens + self.gate.to(tokens.dtype) * attended
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class UNet(nn.Module):
    """U-Net with `depth` levels (default 4), no attention gates.

    Encoder channels progress `base_ch -> 2x -> 4x -> 8x ...` (one doubling per
    level), each level followed by 2x2 max-pooling. The input's height and
    width must be divisible by `2 ** (depth - 1)`. The decoder mirrors it with transposed
    convolutions; each skip connection concatenates the raw encoder feature
    map at that level directly (no attention gating). A 1x1 convolution
    produces the per-pixel class logits.

    Three options extend this, all off by default:

    `norm_type="group"` replaces every BatchNorm with a GroupNorm.

    `bottleneck_attention` inserts self-attention over the bottleneck grid
    between the last encoder stage and the first upsampling.

    `deep_supervision` attaches a 1x1 logit head to each intermediate decoder
    level. The heads are training machinery: the forward pass returns the
    same final logits with or without them unless a caller passes
    `return_aux`.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_ch: int = 48,
        dropout_p: float = 0.0,
        num_sensors: int = 1,
        film_sensor: bool = False,
        film_doy: bool = False,
        film_latlon: bool = False,
        norm_type: str = "batch",
        bottleneck_attention: bool = False,
        bottleneck_attention_heads: int = 8,
        deep_supervision: bool = False,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if int(depth) < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
        self.depth = int(depth)
        _check_norm_and_sensors(norm_type, num_sensors)
        self.num_sensors = int(num_sensors)
        self.norm_type = str(norm_type).strip().lower()
        self.deep_supervision = bool(deep_supervision)

        block = lambda i, o: DoubleConv(
            i, o, dropout_p=dropout_p, num_sensors=num_sensors, norm_type=norm_type
        )

        # Level i (1-based) is `enc{i}` / `dec{i}` / `up{i}`; for depth 4 these are the
        # same attribute names a checkpoint from the fixed 4-level U-Net uses.
        widths = [base_ch * 2**i for i in range(self.depth)]
        in_chs = [in_channels] + widths[:-1]
        for i in range(self.depth):
            setattr(self, f"enc{i + 1}", block(in_chs[i], widths[i]))

        self.pool = nn.MaxPool2d(2)

        self.bottleneck_attention = (
            BottleneckAttention(
                widths[-1], num_heads=bottleneck_attention_heads, dropout_p=dropout_p
            )
            if bottleneck_attention
            else None
        )

        for level in range(self.depth - 1, 0, -1):
            setattr(self, f"up{level}", nn.ConvTranspose2d(widths[level], widths[level - 1], 2, stride=2))
            setattr(self, f"dec{level}", block(widths[level - 1] * 2, widths[level - 1]))

        # Stage widths in modulation order: enc1..enc{depth}, then dec{depth-1}..dec1.
        decoder_widths = [widths[level - 1] for level in range(self.depth - 1, 0, -1)]
        self.film = (
            SceneFiLM(
                widths + decoder_widths,
                use_sensor=film_sensor, use_doy=film_doy, use_latlon=film_latlon,
            )
            if (film_sensor or film_doy or film_latlon)
            else None
        )

        self._num_sensors = int(num_sensors)
        self.out_conv = nn.Conv2d(base_ch, num_classes, 1)

        # Auxiliary heads, coarsest first: every decoder level except the last
        # (dec1 is the final level and already has `out_conv`, so giving it a
        # second head would only duplicate the primary loss). Depth 4: dec3 at
        # 1/4 resolution and dec2 at 1/2.
        self.aux_heads = (
            nn.ModuleList(
                [nn.Conv2d(widths[level - 1], num_classes, 1) for level in range(self.depth - 1, 1, -1)]
            )
            if deep_supervision
            else None
        )

    @property
    def uses_scene_context(self) -> bool:
        """True when FiLM is on: ``forward`` then needs ``context`` (see
        :mod:`landscape_change_detection_pipeline.features.scene_context`)."""
        return self.film is not None or self._num_sensors > 1

    def _film(self, level: int, x: torch.Tensor, embedding: Optional[torch.Tensor]) -> torch.Tensor:
        return x if embedding is None else self.film.modulate(level, x, embedding)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        sensor_ids: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        """
        Args:
            x: features, (B, C, H, W).
            context: (B, 5) scene context ``[sensor_index, doy_sin, doy_cos,
                lat_norm, lon_norm]`` (see ``features/scene_context.py``).
                Required when any ``film_*`` option is on, ignored otherwise.
            sensor_ids: (B,) int64 sensor indices for the per-sensor
                normalisation banks (``num_sensors > 1``).
            return_aux: also return the deep-supervision heads' logits. Only
                the training loop asks for these; every other caller gets the
                same single tensor whether deep supervision is on or off, so
                evaluation, export and inference need no branch for it.

        Returns:
            Logits, (B, num_classes, H, W). With `return_aux`, a
            `(logits, [aux_coarse, aux_mid])` pair -- the auxiliary list is
            empty when deep supervision is off, and each auxiliary tensor is
            at its own decoder level's resolution, not upsampled.
        """
        if sensor_ids is None and self._num_sensors > 1:
            if context is None:
                raise ValueError("num_sensors>1: pass `context` (or `sensor_ids`) to pick the per-sensor BatchNorm bank")
            sensor_ids = context[:, SENSOR_COL].round().long().clamp(0, self._num_sensors - 1)

        embedding = None
        if self.film is not None:
            if context is None:
                raise ValueError("this U-Net has FiLM enabled (film_sensor/film_doy/film_latlon): pass `context`")
            embedding = self.film.embed(context)

        skips = []
        h = x
        for i in range(self.depth):
            if i:
                h = self.pool(h)
            h = self._film(i, getattr(self, f"enc{i + 1}")(h, sensor_ids=sensor_ids), embedding)
            skips.append(h)

        d = skips[-1]
        if self.bottleneck_attention is not None:
            d = self.bottleneck_attention(d)

        decoded = []
        for level in range(self.depth - 1, 0, -1):
            d = getattr(self, f"up{level}")(d)
            d = getattr(self, f"dec{level}")(torch.cat([d, skips[level - 1]], dim=1), sensor_ids=sensor_ids)
            d = self._film(self.depth + (self.depth - 1 - level), d, embedding)
            decoded.append(d)

        logits = self.out_conv(d)
        if not return_aux:
            return logits

        aux = (
            [head(features) for head, features in zip(self.aux_heads, decoded[:-1])]
            if self.aux_heads is not None
            else []
        )
        return logits, aux


def build_unet(
    in_channels: int,
    num_classes: int,
    base_ch: int = 48,
    dropout_p: float = 0.0,
    num_sensors: int = 1,
    film_sensor: bool = False,
    film_doy: bool = False,
    film_latlon: bool = False,
    norm_type: str = "batch",
    bottleneck_attention: bool = False,
    bottleneck_attention_heads: int = 8,
    deep_supervision: bool = False,
    depth: int = 4,
) -> nn.Module:
    """Construct the U-Net segmentation model.

    `num_sensors=1` collapses the per-sensor normalisation banks to a single
    shared layer per block. Dispatched to by
    :func:`landscape_change_detection_pipeline.models.registry.build_model` for
    ``model.type == "unet"``.
    """
    return UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_ch=base_ch,
        dropout_p=dropout_p,
        num_sensors=num_sensors,
        film_sensor=film_sensor,
        film_doy=film_doy,
        film_latlon=film_latlon,
        norm_type=norm_type,
        bottleneck_attention=bottleneck_attention,
        bottleneck_attention_heads=bottleneck_attention_heads,
        deep_supervision=deep_supervision,
        depth=depth,
    )
