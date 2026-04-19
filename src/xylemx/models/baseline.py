"""Segmentation model registry for the challenge baseline."""

from __future__ import annotations

from functools import partial
import logging

import torch
import torch.nn.functional as F
from torch import nn

LOGGER = logging.getLogger(__name__)


def _create_timm_features_encoder(
    encoder_name: str,
    *,
    in_channels: int,
    pretrained: bool,
    stochastic_depth: float,
) -> nn.Module:
    """Create a timm encoder that exposes feature stages."""

    try:
        import timm
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("timm is required for timm-backed segmentation models") from exc

    create_kwargs = {
        "in_chans": in_channels,
        "features_only": True,
        "pretrained": pretrained,
    }
    if stochastic_depth > 0:
        create_kwargs["drop_path_rate"] = stochastic_depth
    try:
        return timm.create_model(encoder_name, **create_kwargs)
    except TypeError:
        if "drop_path_rate" not in create_kwargs:
            raise
        LOGGER.warning("Encoder %s does not accept drop_path_rate; retrying without stochastic depth", encoder_name)
        create_kwargs.pop("drop_path_rate")
        return timm.create_model(encoder_name, **create_kwargs)


class PyramidPoolingModule(nn.Module):
    """Pyramid pooling used by UPerNet-like heads."""

    def __init__(self, in_channels: int, out_channels: int, pool_sizes: tuple[int, ...] = (1, 2, 3, 6)) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=pool_size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for pool_size in pool_sizes
        )
        merged_channels = in_channels + out_channels * len(pool_sizes)
        self.bottleneck = ConvBlock(merged_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = x.shape[-2:]
        pooled = [x]
        for stage in self.stages:
            pooled_feature = stage(x)
            pooled.append(F.interpolate(pooled_feature, size=target_size, mode="bilinear", align_corners=False))
        return self.bottleneck(torch.cat(pooled, dim=1))


class ASPP(nn.Module):
    """Atrous spatial pyramid pooling used by DeepLabV3+."""

    def __init__(self, in_channels: int, out_channels: int, rates: tuple[int, ...] = (1, 6, 12, 18)) -> None:
        super().__init__()
        branches: list[nn.Module] = []
        for rate in rates:
            if rate == 1:
                kernel_size = 1
                padding = 0
            else:
                kernel_size = 3
                padding = rate
            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        merged_channels = out_channels * (len(rates) + 1)
        self.project = ConvBlock(merged_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = x.shape[-2:]
        outputs = [branch(x) for branch in self.branches]
        pooled = self.global_pool(x)
        outputs.append(F.interpolate(pooled, size=target_size, mode="bilinear", align_corners=False))
        return self.project(torch.cat(outputs, dim=1))


class ConvBlock(nn.Module):
    """A compact convolutional block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        use_cbam: bool = False,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.attention = CBAM(out_channels) if use_cbam else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(self.block(x))


class ChannelAttention(nn.Module):
    """Channel attention module from CBAM."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced_channels = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False),
        )
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_attention = self.mlp(self.avg_pool(x))
        max_attention = self.mlp(self.max_pool(x))
        return self.gate(avg_attention + max_attention)


class SpatialAttention(nn.Module):
    """Spatial attention module from CBAM."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        average_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.conv(torch.cat([average_map, max_map], dim=1))
        return self.gate(attention)


class CBAM(nn.Module):
    """Convolutional Block Attention Module with channel and spatial attention."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class DecoderBlock(nn.Module):
    """One upsampling stage in a UNet decoder."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        use_cbam: bool = False,
    ) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ConvBlock(out_channels + skip_channels, out_channels, dropout=dropout, use_cbam=use_cbam)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class SmallUNet(nn.Module):
    """A lightweight UNet suitable for modest GPUs."""

    def __init__(self, in_channels: int, base_channels: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder1 = ConvBlock(in_channels, base_channels, dropout=dropout)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2, dropout=dropout)
        self.encoder3 = ConvBlock(base_channels * 2, base_channels * 4, dropout=dropout)
        self.encoder4 = ConvBlock(base_channels * 4, base_channels * 8, dropout=dropout)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16, dropout=dropout)
        self.decoder4 = DecoderBlock(base_channels * 16, base_channels * 8, base_channels * 8, dropout=dropout)
        self.decoder3 = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4, dropout=dropout)
        self.decoder2 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2, dropout=dropout)
        self.decoder1 = DecoderBlock(base_channels * 2, base_channels, base_channels, dropout=dropout)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.pool(x1))
        x3 = self.encoder3(self.pool(x2))
        x4 = self.encoder4(self.pool(x3))
        bottleneck = self.bottleneck(self.pool(x4))

        y = self.decoder4(bottleneck, x4)
        y = self.decoder3(y, x3)
        y = self.decoder2(y, x2)
        y = self.decoder1(y, x1)
        return self.head(y)


class TimmUNet(nn.Module):
    """A generic UNet decoder on top of a timm encoder."""

    def __init__(
        self,
        encoder_name: str,
        *,
        in_channels: int,
        dropout: float = 0.0,
        pretrained: bool = False,
        stochastic_depth: float = 0.0,
        use_cbam: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = _create_timm_features_encoder(
            encoder_name,
            in_channels=in_channels,
            pretrained=pretrained,
            stochastic_depth=stochastic_depth,
        )

        channels = list(self.encoder.feature_info.channels())
        if len(channels) < 4:
            raise ValueError(f"Encoder {encoder_name} did not expose enough feature stages")

        self.bottleneck = ConvBlock(channels[-1], channels[-1], dropout=dropout, use_cbam=use_cbam)
        skip_channels = channels[:-1]
        decoder_blocks: list[DecoderBlock] = []
        current_in_channels = channels[-1]
        for block_index, skip_channels_value in enumerate(reversed(skip_channels)):
            is_last = block_index == len(skip_channels) - 1
            out_channels = max(skip_channels_value, 32) if is_last else skip_channels_value
            decoder_blocks.append(
                DecoderBlock(
                    current_in_channels,
                    skip_channels_value,
                    out_channels,
                    dropout=dropout,
                    use_cbam=use_cbam,
                )
            )
            current_in_channels = out_channels
        self.decoders = nn.ModuleList(decoder_blocks)
        self.head = nn.Sequential(
            ConvBlock(current_in_channels, current_in_channels, dropout=dropout, use_cbam=use_cbam),
            nn.Conv2d(current_in_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        y = self.bottleneck(features[-1])
        for decoder, skip in zip(self.decoders, reversed(features[:-1]), strict=True):
            y = decoder(y, skip)
        logits = self.head(y)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


class TimmFPN(nn.Module):
    """A feature pyramid segmentation head on top of a timm encoder."""

    def __init__(
        self,
        encoder_name: str,
        *,
        in_channels: int,
        dropout: float = 0.0,
        pretrained: bool = False,
        stochastic_depth: float = 0.0,
        pyramid_channels: int = 128,
        use_cbam: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = _create_timm_features_encoder(
            encoder_name,
            in_channels=in_channels,
            pretrained=pretrained,
            stochastic_depth=stochastic_depth,
        )

        channels = list(self.encoder.feature_info.channels())
        if len(channels) < 4:
            raise ValueError(f"Encoder {encoder_name} did not expose enough feature stages")

        self.laterals = nn.ModuleList(nn.Conv2d(channel, pyramid_channels, kernel_size=1) for channel in channels)
        self.smooth = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(pyramid_channels, pyramid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(pyramid_channels),
                nn.ReLU(inplace=True),
                CBAM(pyramid_channels) if use_cbam else nn.Identity(),
            )
            for _ in channels
        )
        fused_channels = pyramid_channels * len(channels)
        self.head = nn.Sequential(
            ConvBlock(fused_channels, pyramid_channels, dropout=dropout, use_cbam=use_cbam),
            nn.Conv2d(pyramid_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        pyramids = [lateral(feature) for lateral, feature in zip(self.laterals, features, strict=True)]
        for index in range(len(pyramids) - 2, -1, -1):
            pyramids[index] = pyramids[index] + F.interpolate(
                pyramids[index + 1],
                size=pyramids[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        pyramids = [smooth(feature) for smooth, feature in zip(self.smooth, pyramids, strict=True)]
        target_size = pyramids[0].shape[-2:]
        fused = torch.cat(
            [
                feature
                if feature.shape[-2:] == target_size
                else F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
                for feature in pyramids
            ],
            dim=1,
        )
        logits = self.head(fused)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


class TimmUNetPP(nn.Module):
    """A lightweight UNet++ style decoder on top of a timm encoder."""

    def __init__(
        self,
        encoder_name: str,
        *,
        in_channels: int,
        dropout: float = 0.0,
        pretrained: bool = False,
        stochastic_depth: float = 0.0,
        use_cbam: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = _create_timm_features_encoder(
            encoder_name,
            in_channels=in_channels,
            pretrained=pretrained,
            stochastic_depth=stochastic_depth,
        )

        channels = list(self.encoder.feature_info.channels())
        if len(channels) < 4:
            raise ValueError(f"Encoder {encoder_name} did not expose enough feature stages")

        self.row_channels = [max(channel, 32) for channel in channels]
        self.projections = nn.ModuleList(
            nn.Conv2d(in_channel, out_channel, kernel_size=1)
            for in_channel, out_channel in zip(channels, self.row_channels, strict=True)
        )
        node_blocks: dict[str, nn.Module] = {}
        for depth in range(1, len(self.row_channels)):
            for row in range(0, len(self.row_channels) - depth):
                in_channels_total = self.row_channels[row] * depth + self.row_channels[row + 1]
                node_blocks[f"{row}_{depth}"] = ConvBlock(
                    in_channels_total,
                    self.row_channels[row],
                    dropout=dropout,
                    use_cbam=use_cbam,
                )
        self.nodes = nn.ModuleDict(node_blocks)
        self.head = nn.Sequential(
            ConvBlock(self.row_channels[0], self.row_channels[0], dropout=dropout, use_cbam=use_cbam),
            nn.Conv2d(self.row_channels[0], 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        grid: dict[tuple[int, int], torch.Tensor] = {
            (index, 0): projection(feature)
            for index, (projection, feature) in enumerate(zip(self.projections, features, strict=True))
        }

        for depth in range(1, len(self.row_channels)):
            for row in range(0, len(self.row_channels) - depth):
                target_size = grid[(row, 0)].shape[-2:]
                concatenated = [grid[(row, step)] for step in range(depth)]
                upsampled = grid[(row + 1, depth - 1)]
                if upsampled.shape[-2:] != target_size:
                    upsampled = F.interpolate(upsampled, size=target_size, mode="bilinear", align_corners=False)
                concatenated.append(upsampled)
                grid[(row, depth)] = self.nodes[f"{row}_{depth}"](torch.cat(concatenated, dim=1))

        logits = self.head(grid[(0, len(self.row_channels) - 1)])
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


class TimmUPerNet(nn.Module):
    """A lightweight UPerNet-style head on top of a timm encoder."""

    def __init__(
        self,
        encoder_name: str,
        *,
        in_channels: int,
        dropout: float = 0.0,
        pretrained: bool = False,
        stochastic_depth: float = 0.0,
        pyramid_channels: int = 128,
        use_cbam: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = _create_timm_features_encoder(
            encoder_name,
            in_channels=in_channels,
            pretrained=pretrained,
            stochastic_depth=stochastic_depth,
        )
        channels = list(self.encoder.feature_info.channels())
        if len(channels) < 4:
            raise ValueError(f"Encoder {encoder_name} did not expose enough feature stages")

        self.ppm = PyramidPoolingModule(channels[-1], pyramid_channels)
        self.laterals = nn.ModuleList(nn.Conv2d(channel, pyramid_channels, kernel_size=1) for channel in channels[:-1])
        self.fpn_smooth = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(pyramid_channels, pyramid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(pyramid_channels),
                nn.ReLU(inplace=True),
                CBAM(pyramid_channels) if use_cbam else nn.Identity(),
            )
            for _ in channels[:-1]
        )
        merged_channels = pyramid_channels * len(channels)
        self.head = nn.Sequential(
            ConvBlock(merged_channels, pyramid_channels, dropout=dropout, use_cbam=use_cbam),
            nn.Conv2d(pyramid_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        top = self.ppm(features[-1])
        pyramids = [lateral(feature) for lateral, feature in zip(self.laterals, features[:-1], strict=True)]
        outputs = [None] * len(features)
        outputs[-1] = top
        running = top
        for index in range(len(pyramids) - 1, -1, -1):
            running = F.interpolate(running, size=pyramids[index].shape[-2:], mode="bilinear", align_corners=False)
            running = pyramids[index] + running
            outputs[index] = self.fpn_smooth[index](running)
        target_size = outputs[0].shape[-2:]
        fused = torch.cat(
            [
                feature
                if feature.shape[-2:] == target_size
                else F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
                for feature in outputs
            ],
            dim=1,
        )
        logits = self.head(fused)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


class TimmDeepLabV3Plus(nn.Module):
    """A lightweight DeepLabV3+ style head on top of a timm encoder."""

    def __init__(
        self,
        encoder_name: str,
        *,
        in_channels: int,
        dropout: float = 0.0,
        pretrained: bool = False,
        stochastic_depth: float = 0.0,
        aspp_channels: int = 128,
        low_level_channels: int = 48,
        use_cbam: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = _create_timm_features_encoder(
            encoder_name,
            in_channels=in_channels,
            pretrained=pretrained,
            stochastic_depth=stochastic_depth,
        )
        channels = list(self.encoder.feature_info.channels())
        if len(channels) < 4:
            raise ValueError(f"Encoder {encoder_name} did not expose enough feature stages")

        self.aspp = ASPP(channels[-1], aspp_channels)
        self.low_level = nn.Sequential(
            nn.Conv2d(channels[0], low_level_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_channels),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            ConvBlock(aspp_channels + low_level_channels, aspp_channels, dropout=dropout, use_cbam=use_cbam),
            nn.Conv2d(aspp_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        high = self.aspp(features[-1])
        low = self.low_level(features[0])
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.decoder(torch.cat([high, low], dim=1))
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


ENCODER_ALIASES = {
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "resnet50": "resnet50",
    "resnet101": "resnet101",
    "efficientnet_b0": "efficientnet_b0",
    "convnext_tiny": "convnext_tiny",
    "convnext_small": "convnext_small",
    "convnext_base": "convnext_base",
    "convnext_large": "convnext_large",
    "convnextv2_atto": "convnextv2_atto",
    "convnextv2_femto": "convnextv2_femto",
    "convnextv2_pico": "convnextv2_pico",
    "convnextv2_nano": "convnextv2_nano",
    "convnextv2_tiny": "convnextv2_tiny",
    "convnextv2_small": "convnextv2_small",
    "convnextv2_base": "convnextv2_base",
    "coatnet0": "coatnet_0_rw_224",
    "coatnet1": "coatnet_1_rw_224",
    "coatnet2": "coatnet_2_rw_224",
    "coatnet3": "coatnet_3_rw_224",
    "vgg11": "vgg11_bn",
    "vgg13": "vgg13_bn",
    "vgg16": "vgg16_bn",
    "vgg19": "vgg19_bn",
}

MODEL_HEADS = {
    "_unet": TimmUNet,
    "_unet_cbam": partial(TimmUNet, use_cbam=True),
    "_fpn": TimmFPN,
    "_fpn_cbam": partial(TimmFPN, use_cbam=True),
    "_unetpp": TimmUNetPP,
    "_unetpp_cbam": partial(TimmUNetPP, use_cbam=True),
    "_upernet": TimmUPerNet,
    "_upernet_cbam": partial(TimmUPerNet, use_cbam=True),
    "_deeplabv3plus": TimmDeepLabV3Plus,
    "_deeplabv3plus_cbam": partial(TimmDeepLabV3Plus, use_cbam=True),
}


def supported_model_names() -> list[str]:
    """Return the sorted list of supported model names."""

    names = ["small_unet", "unet", "coatnext_tiny_unet"]
    for alias in ENCODER_ALIASES:
        for suffix in MODEL_HEADS:
            names.append(f"{alias}{suffix}")
    return sorted(names)


def build_model(
    name: str,
    *,
    in_channels: int,
    dropout: float = 0.0,
    stochastic_depth: float = 0.0,
    pretrained: bool = False,
) -> nn.Module:
    """Build one of the supported segmentation models."""

    normalized = name.lower()
    if normalized in {"small_unet", "unet"}:
        return SmallUNet(in_channels=in_channels, dropout=dropout)
    if normalized in {"coatnext_tiny_unet", "coatnext_unet"}:
        from xylemx.models.coatnext_tiny import CoAtNeXtTinyUNet

        return CoAtNeXtTinyUNet(
            in_channels=in_channels,
            dropout=dropout,
            drop_path_rate=stochastic_depth,
        )
    for suffix, head_cls in MODEL_HEADS.items():
        if normalized.endswith(suffix):
            alias = normalized[: -len(suffix)]
            encoder_name = ENCODER_ALIASES.get(alias)
            if encoder_name is None:
                break
            return head_cls(
                encoder_name,
                in_channels=in_channels,
                dropout=dropout,
                pretrained=pretrained,
                stochastic_depth=stochastic_depth,
            )
    raise ValueError(f"Unsupported model: {name}")
