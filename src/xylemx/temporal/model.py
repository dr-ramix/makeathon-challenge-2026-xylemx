"""Dual-head temporal U-Net models."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    """Compact 2D conv block used throughout the temporal model."""

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
    """One U-Net decoder stage."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ConvBlock(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class TemporalFusionStem(nn.Module):
    """Temporal CNN stem for ``[B, T, C, H, W]`` inputs."""

    def __init__(self, in_channels_per_step: int, stem_channels: int, temporal_kernel_size: int = 3) -> None:
        super().__init__()
        padding = temporal_kernel_size // 2
        self.depthwise_temporal = nn.Conv3d(
            in_channels_per_step,
            in_channels_per_step,
            kernel_size=(temporal_kernel_size, 1, 1),
            padding=(padding, 0, 0),
            groups=in_channels_per_step,
            bias=False,
        )
        self.pointwise = nn.Sequential(
            nn.Conv3d(in_channels_per_step, stem_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(stem_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(stem_channels, stem_channels, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(stem_channels * 2, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4)
        x = self.depthwise_temporal(x)
        x = self.pointwise(x)
        mean_pool = torch.mean(x, dim=2)
        max_pool, _ = torch.max(x, dim=2)
        return self.project(torch.cat([mean_pool, max_pool], dim=1))


class DualHeadUNet(nn.Module):
    """A small dual-head U-Net with optional temporal fusion stem."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_time_classes: int,
        input_is_sequence: bool,
        stem_channels: int = 32,
        base_channels: int = 32,
        dropout: float = 0.0,
        temporal_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.input_is_sequence = input_is_sequence
        if input_is_sequence:
            self.stem = TemporalFusionStem(in_channels, stem_channels, temporal_kernel_size=temporal_kernel_size)
            encoder_in = stem_channels
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, stem_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=True),
            )
            encoder_in = stem_channels

        self.encoder1 = ConvBlock(encoder_in, base_channels, dropout=dropout)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2, dropout=dropout)
        self.encoder3 = ConvBlock(base_channels * 2, base_channels * 4, dropout=dropout)
        self.encoder4 = ConvBlock(base_channels * 4, base_channels * 8, dropout=dropout)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16, dropout=dropout)
        self.decoder4 = DecoderBlock(base_channels * 16, base_channels * 8, base_channels * 8, dropout=dropout)
        self.decoder3 = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4, dropout=dropout)
        self.decoder2 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2, dropout=dropout)
        self.decoder1 = DecoderBlock(base_channels * 2, base_channels, base_channels, dropout=dropout)
        self.mask_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.time_head = nn.Conv2d(base_channels, num_time_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.input_is_sequence:
            x = self.stem(x)
        else:
            x = self.stem(x)
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.pool(x1))
        x3 = self.encoder3(self.pool(x2))
        x4 = self.encoder4(self.pool(x3))
        bottleneck = self.bottleneck(self.pool(x4))
        y = self.decoder4(bottleneck, x4)
        y = self.decoder3(y, x3)
        y = self.decoder2(y, x2)
        y = self.decoder1(y, x1)
        return {
            "mask_logits": self.mask_head(y),
            "time_logits": self.time_head(y),
        }
