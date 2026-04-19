"""FiLM-conditioned dual-head U-Net models for temporal deforestation tasks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    """Compact 2D conv block used throughout the temporal model."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FiLMLayer(nn.Module):
    """Feature-wise linear modulation from sample-level conditioning vectors."""

    def __init__(self, channels: int, cond_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.enabled = cond_dim > 0
        if not self.enabled:
            self.mlp = None
            return

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, channels * 2),
        )
        final = self.mlp[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        if not self.enabled or cond is None or self.mlp is None:
            return x
        film = self.mlp(cond)
        gamma, beta = torch.chunk(film, chunks=2, dim=1)
        gamma = 1.0 + gamma
        return gamma[:, :, None, None] * x + beta[:, :, None, None]


class FiLMConvBlock(nn.Module):
    """Conv block followed by FiLM modulation."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, dropout=dropout)
        self.film = FiLMLayer(out_channels, cond_dim=cond_dim, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        x = self.conv(x)
        return self.film(x, cond)


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


class SqueezeExcite2d(nn.Module):
    """Channel attention for better modality mixing with low overhead."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // max(reduction, 1), 4)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = torch.mean(x, dim=(2, 3), keepdim=True)
        scale = torch.sigmoid(self.fc2(F.silu(self.fc1(context), inplace=True)))
        return x * scale


class ResidualSEBlock(nn.Module):
    """Residual conv block with squeeze-excitation attention."""

    def __init__(self, in_channels: int, out_channels: int, *, dropout: float = 0.0, se_reduction: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.se = SqueezeExcite2d(out_channels, reduction=se_reduction)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        y = self.conv1(x)
        y = F.silu(self.norm1(y), inplace=True)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.se(y)
        y = self.dropout(y)
        return F.silu(y + residual, inplace=True)


class FiLMResidualBlock(nn.Module):
    """Residual attention block with FiLM conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        cond_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        se_reduction: int = 8,
    ) -> None:
        super().__init__()
        self.block = ResidualSEBlock(
            in_channels,
            out_channels,
            dropout=dropout,
            se_reduction=se_reduction,
        )
        self.film = FiLMLayer(out_channels, cond_dim=cond_dim, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        return self.film(self.block(x), cond)


class DecoderResidualBlock(nn.Module):
    """Decoder stage using residual attention conv blocks."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        dropout: float = 0.0,
        se_reduction: int = 8,
    ) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ResidualSEBlock(
            out_channels + skip_channels,
            out_channels,
            dropout=dropout,
            se_reduction=se_reduction,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class ASPPContextBlock(nn.Module):
    """Lightweight ASPP-style context aggregation for the bottleneck."""

    def __init__(self, channels: int, dilations: tuple[int, ...] = (1, 2, 4)) -> None:
        super().__init__()
        branches = []
        for dilation in dilations:
            branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
                    nn.GroupNorm(_num_groups(channels), channels),
                    nn.SiLU(inplace=True),
                )
            )
        self.branches = nn.ModuleList(branches)
        merged_channels = channels * (len(dilations) + 1)
        self.project = nn.Sequential(
            nn.Conv2d(merged_channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_num_groups(channels), channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        for branch in self.branches:
            feats.append(branch(x))
        return self.project(torch.cat(feats, dim=1))


class _BaseTemporalUNet(nn.Module):
    """Shared utilities for temporal U-Net variants."""

    def __init__(self, cond_dim: int, input_is_sequence: bool, temporal_kernel_size: int) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.input_is_sequence = input_is_sequence
        self.temporal_kernel_size = temporal_kernel_size

    def _flatten_temporal(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x
        if x.ndim == 5:
            batch, time_steps, channels, height, width = x.shape
            return x.reshape(batch, time_steps * channels, height, width)
        raise ValueError(f"Expected 4D or 5D input tensor, got shape={tuple(x.shape)}")

    def _prepare_inputs(self, x_image: torch.Tensor, x_cond: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self._flatten_temporal(x_image)
        if self.cond_dim > 0 and x_cond is None:
            x_cond = x.new_zeros((x.shape[0], self.cond_dim))
        return x, x_cond


class DualHeadUNet(_BaseTemporalUNet):
    """FiLM-conditioned dual-head 2D U-Net for mask + time prediction."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_time_classes: int,
        cond_dim: int,
        input_is_sequence: bool = False,
        stem_channels: int = 32,
        base_channels: int = 32,
        dropout: float = 0.0,
        temporal_kernel_size: int = 3,
        film_hidden_dim: int = 128,
    ) -> None:
        super().__init__(cond_dim=cond_dim, input_is_sequence=input_is_sequence, temporal_kernel_size=temporal_kernel_size)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(stem_channels), stem_channels),
            nn.SiLU(inplace=True),
        )

        self.encoder1 = FiLMConvBlock(stem_channels, base_channels, cond_dim=cond_dim, hidden_dim=film_hidden_dim, dropout=dropout)
        self.encoder2 = FiLMConvBlock(base_channels, base_channels * 2, cond_dim=cond_dim, hidden_dim=film_hidden_dim, dropout=dropout)
        self.encoder3 = FiLMConvBlock(base_channels * 2, base_channels * 4, cond_dim=cond_dim, hidden_dim=film_hidden_dim, dropout=dropout)
        self.encoder4 = FiLMConvBlock(base_channels * 4, base_channels * 8, cond_dim=cond_dim, hidden_dim=film_hidden_dim, dropout=dropout)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = FiLMConvBlock(base_channels * 8, base_channels * 16, cond_dim=cond_dim, hidden_dim=film_hidden_dim, dropout=dropout)

        self.decoder4 = DecoderBlock(base_channels * 16, base_channels * 8, base_channels * 8, dropout=dropout)
        self.decoder3 = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4, dropout=dropout)
        self.decoder2 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2, dropout=dropout)
        self.decoder1 = DecoderBlock(base_channels * 2, base_channels, base_channels, dropout=dropout)

        self.mask_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.time_head = nn.Conv2d(base_channels, num_time_classes, kernel_size=1)

    def forward(self, x_image: torch.Tensor, x_cond: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x, x_cond = self._prepare_inputs(x_image, x_cond)

        x = self.stem(x)
        x1 = self.encoder1(x, x_cond)
        x2 = self.encoder2(self.pool(x1), x_cond)
        x3 = self.encoder3(self.pool(x2), x_cond)
        x4 = self.encoder4(self.pool(x3), x_cond)
        bottleneck = self.bottleneck(self.pool(x4), x_cond)

        y = self.decoder4(bottleneck, x4)
        y = self.decoder3(y, x3)
        y = self.decoder2(y, x2)
        y = self.decoder1(y, x1)
        return {
            "mask_logits": self.mask_head(y),
            "time_logits": self.time_head(y),
        }


class DualHeadUNetPlus(_BaseTemporalUNet):
    """Higher-capacity FiLM U-Net with residual attention and ASPP bottleneck."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_time_classes: int,
        cond_dim: int,
        input_is_sequence: bool = False,
        stem_channels: int = 40,
        base_channels: int = 48,
        dropout: float = 0.1,
        temporal_kernel_size: int = 3,
        film_hidden_dim: int = 192,
        se_reduction: int = 8,
    ) -> None:
        super().__init__(cond_dim=cond_dim, input_is_sequence=input_is_sequence, temporal_kernel_size=temporal_kernel_size)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(stem_channels), stem_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(stem_channels, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(stem_channels), stem_channels),
            nn.SiLU(inplace=True),
        )

        self.pool = nn.MaxPool2d(2)

        self.encoder1 = FiLMResidualBlock(
            stem_channels,
            base_channels,
            cond_dim=cond_dim,
            hidden_dim=film_hidden_dim,
            dropout=dropout,
            se_reduction=se_reduction,
        )
        self.encoder2 = FiLMResidualBlock(
            base_channels,
            base_channels * 2,
            cond_dim=cond_dim,
            hidden_dim=film_hidden_dim,
            dropout=dropout,
            se_reduction=se_reduction,
        )
        self.encoder3 = FiLMResidualBlock(
            base_channels * 2,
            base_channels * 4,
            cond_dim=cond_dim,
            hidden_dim=film_hidden_dim,
            dropout=dropout,
            se_reduction=se_reduction,
        )
        self.encoder4 = FiLMResidualBlock(
            base_channels * 4,
            base_channels * 8,
            cond_dim=cond_dim,
            hidden_dim=film_hidden_dim,
            dropout=dropout,
            se_reduction=se_reduction,
        )

        self.bottleneck = FiLMResidualBlock(
            base_channels * 8,
            base_channels * 16,
            cond_dim=cond_dim,
            hidden_dim=film_hidden_dim,
            dropout=dropout,
            se_reduction=se_reduction,
        )
        self.context = ASPPContextBlock(base_channels * 16, dilations=(1, 2, 4, 6))
        self.context_film = FiLMLayer(base_channels * 16, cond_dim=cond_dim, hidden_dim=film_hidden_dim)

        self.decoder4 = DecoderResidualBlock(
            base_channels * 16,
            base_channels * 8,
            base_channels * 8,
            dropout=dropout,
            se_reduction=se_reduction,
        )
        self.decoder3 = DecoderResidualBlock(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4,
            dropout=dropout,
            se_reduction=se_reduction,
        )
        self.decoder2 = DecoderResidualBlock(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
            dropout=dropout,
            se_reduction=se_reduction,
        )
        self.decoder1 = DecoderResidualBlock(
            base_channels * 2,
            base_channels,
            base_channels,
            dropout=dropout,
            se_reduction=se_reduction,
        )

        self.mask_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(base_channels), base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, 1, kernel_size=1),
        )
        self.time_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(base_channels), base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, num_time_classes, kernel_size=1),
        )

    def forward(self, x_image: torch.Tensor, x_cond: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x, x_cond = self._prepare_inputs(x_image, x_cond)

        x = self.stem(x)
        x1 = self.encoder1(x, x_cond)
        x2 = self.encoder2(self.pool(x1), x_cond)
        x3 = self.encoder3(self.pool(x2), x_cond)
        x4 = self.encoder4(self.pool(x3), x_cond)

        bottleneck = self.bottleneck(self.pool(x4), x_cond)
        bottleneck = self.context(bottleneck)
        bottleneck = self.context_film(bottleneck, x_cond)

        y = self.decoder4(bottleneck, x4)
        y = self.decoder3(y, x3)
        y = self.decoder2(y, x2)
        y = self.decoder1(y, x1)
        return {
            "mask_logits": self.mask_head(y),
            "time_logits": self.time_head(y),
        }
