"""ResNet-based UNet model for segmentation.

This uses `torchvision.models.resnet18` as an encoder and the
existing decoder blocks from `baseline.py` for a simple, stronger
segmentation backbone.
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn

LOGGER = logging.getLogger(__name__)


class ResNetUNet(nn.Module):
    """A UNet-like decoder on top of a ResNet-18 encoder.

    The implementation prefers `torchvision.models.resnet18`. If
    `torchvision` is not available, constructing this model will
    raise a clear error.
    """

    def __init__(self, in_channels: int = 3, pretrained: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        try:
            # Import lazily to avoid hard dependency until used.
            from torchvision import models
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("torchvision required for ResNetUNet") from exc

        # Build a resnet18 encoder
        try:
            resnet = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
        except Exception:
            # Fallback for older torchvision versions
            resnet = models.resnet18(pretrained=pretrained)

        # If input channels != 3, replace the first conv to accept custom channels
        if in_channels != 3:
            conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            resnet.conv1 = conv1

        # Encoder stages
        self.initial = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64
        self.layer2 = resnet.layer2  # 128
        self.layer3 = resnet.layer3  # 256
        self.layer4 = resnet.layer4  # 512

        # Decode from stride-32 features back toward the input resolution.
        from .baseline import ConvBlock, DecoderBlock

        self.bottleneck = ConvBlock(512, 512, dropout=dropout)
        self.decoder4 = DecoderBlock(512, 256, 256, dropout=dropout)
        self.decoder3 = DecoderBlock(256, 128, 128, dropout=dropout)
        self.decoder2 = DecoderBlock(128, 64, 64, dropout=dropout)
        self.decoder1 = DecoderBlock(64, 64, 32, dropout=dropout)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.initial(x)
        x1 = self.maxpool(x0)
        e1 = self.layer1(x1)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        b = self.bottleneck(e4)

        y = self.decoder4(b, e3)
        y = self.decoder3(y, e2)
        y = self.decoder2(y, e1)
        y = self.decoder1(y, x0)
        logits = self.head(y)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


__all__ = ["ResNetUNet"]
