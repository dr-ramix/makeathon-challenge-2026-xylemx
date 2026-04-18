"""Segmentation model registry for the challenge baseline."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn

LOGGER = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """A compact convolutional block."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """One upsampling stage in a UNet decoder."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ConvBlock(out_channels + skip_channels, out_channels, dropout=dropout)

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
    ) -> None:
        super().__init__()
        try:
            import timm
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("timm is required for TimmUNet models") from exc

        self.encoder = timm.create_model(
            encoder_name,
            in_chans=in_channels,
            features_only=True,
            pretrained=pretrained,
            drop_path_rate=stochastic_depth,
        )
        channels = list(self.encoder.feature_info.channels())
        if len(channels) < 4:
            raise ValueError(f"Encoder {encoder_name} did not expose enough feature stages")

        c1, c2, c3, c4, c5 = channels[-5:]
        d1 = max(c1, 32)
        self.bottleneck = ConvBlock(c5, c5, dropout=dropout)
        self.decoder4 = DecoderBlock(c5, c4, c4, dropout=dropout)
        self.decoder3 = DecoderBlock(c4, c3, c3, dropout=dropout)
        self.decoder2 = DecoderBlock(c3, c2, c2, dropout=dropout)
        self.decoder1 = DecoderBlock(c2, c1, d1, dropout=dropout)
        self.head = nn.Sequential(
            ConvBlock(d1, d1, dropout=dropout),
            nn.Conv2d(d1, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        e1, e2, e3, e4, e5 = features[-5:]
        y = self.bottleneck(e5)
        y = self.decoder4(y, e4)
        y = self.decoder3(y, e3)
        y = self.decoder2(y, e2)
        y = self.decoder1(y, e1)
        logits = self.head(y)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


MODEL_ENCODERS = {
    "resnet18_unet": "resnet18",
    "resnet34_unet": "resnet34",
    "efficientnet_b0_unet": "efficientnet_b0",
}


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
    if normalized in MODEL_ENCODERS:
        return TimmUNet(
            MODEL_ENCODERS[normalized],
            in_channels=in_channels,
            dropout=dropout,
            pretrained=pretrained,
            stochastic_depth=stochastic_depth,
        )
    raise ValueError(f"Unsupported model: {name}")
