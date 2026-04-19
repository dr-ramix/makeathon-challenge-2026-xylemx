"""Tiny CoAtNeXt-style U-Net for non-temporal segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DropPath(nn.Module):
    """Stochastic depth over residual branches."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerNorm2d(nn.Module):
    """LayerNorm applied over channels for NCHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class GRN(nn.Module):
    """Global response normalization used in ConvNeXtV2 blocks."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """ConvNeXtV2 block for the C stages."""

    def __init__(self, channels: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.pwconv1 = nn.Linear(channels, channels * 4)
        self.act = nn.GELU()
        self.grn = GRN(channels * 4)
        self.pwconv2 = nn.Linear(channels * 4, channels)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class RelativeSelfAttention2d(nn.Module):
    """Multi-head self-attention with learned 2D relative position bias."""

    def __init__(
        self,
        channels: int,
        *,
        num_heads: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(channels, channels * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        hidden = max(32, channels // 2)
        self.rel_pos_mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_heads),
        )

    def _relative_bias(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        coords_h = torch.arange(height, device=device, dtype=torch.float32)
        coords_w = torch.arange(width, device=device, dtype=torch.float32)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1).reshape(-1, 2)
        relative = coords[:, None, :] - coords[None, :, :]
        if height > 1:
            relative[..., 0] = relative[..., 0] / float(height - 1)
        if width > 1:
            relative[..., 1] = relative[..., 1] / float(width - 1)
        bias = self.rel_pos_mlp(relative).permute(2, 0, 1)
        return bias.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        num_tokens = tokens.shape[1]
        qkv = self.qkv(tokens).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn + self._relative_bias(height, width, x.device, attn.dtype).unsqueeze(0)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, channels)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out.transpose(1, 2).reshape(batch_size, channels, height, width)


class RelativeTransformerBlock(nn.Module):
    """Transformer block for the T stage."""

    def __init__(
        self,
        channels: int,
        *,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(channels * mlp_ratio)
        self.norm1 = LayerNorm2d(channels)
        self.attn = RelativeSelfAttention2d(channels, num_heads=num_heads, attn_drop=dropout, proj_drop=dropout)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = LayerNorm2d(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden_dim, channels, kernel_size=1),
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class Downsample(nn.Module):
    """2x downsampling between encoder stages."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """U-Net decoder stage."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class CoAtNeXtTinyUNet(nn.Module):
    """Tiny C-C-C-T encoder with a U-Net decoder for segmentation."""

    def __init__(
        self,
        in_channels: int,
        *,
        base_channels: int = 48,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        dims = [base_channels, base_channels * 2, base_channels * 4, base_channels * 4]
        depths = [2, 2, 2, 2]
        total_blocks = sum(depths)
        drop_paths = torch.linspace(0, drop_path_rate, steps=total_blocks).tolist()
        dp_index = 0

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )

        self.stage1 = nn.Sequential(
            *[ConvNeXtV2Block(dims[0], drop_path=drop_paths[dp_index + idx]) for idx in range(depths[0])]
        )
        dp_index += depths[0]

        self.down1 = Downsample(dims[0], dims[1])
        self.stage2 = nn.Sequential(
            *[ConvNeXtV2Block(dims[1], drop_path=drop_paths[dp_index + idx]) for idx in range(depths[1])]
        )
        dp_index += depths[1]

        self.down2 = Downsample(dims[1], dims[2])
        self.stage3 = nn.Sequential(
            *[ConvNeXtV2Block(dims[2], drop_path=drop_paths[dp_index + idx]) for idx in range(depths[2])]
        )
        dp_index += depths[2]

        self.down3 = Downsample(dims[2], dims[3])
        self.stage4 = nn.Sequential(
            *[
                RelativeTransformerBlock(
                    dims[3],
                    num_heads=num_heads,
                    drop_path=drop_paths[dp_index + idx],
                    dropout=dropout,
                )
                for idx in range(depths[3])
            ]
        )

        self.dec3 = DecoderBlock(dims[3], dims[2], dims[2], dropout=dropout)
        self.dec2 = DecoderBlock(dims[2], dims[1], dims[1], dropout=dropout)
        self.dec1 = DecoderBlock(dims[1], dims[0], dims[0], dropout=dropout)
        self.head = nn.Conv2d(dims[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.stage1(self.stem(x))
        x2 = self.stage2(self.down1(x1))
        x3 = self.stage3(self.down2(x2))
        x4 = self.stage4(self.down3(x3))

        y = self.dec3(x4, x3)
        y = self.dec2(y, x2)
        y = self.dec1(y, x1)
        logits = self.head(y)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


__all__ = ["CoAtNeXtTinyUNet"]
