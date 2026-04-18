"""CoAtNet-based UNet wrapper using timm feature extractor."""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn

LOGGER = logging.getLogger(__name__)


class CoAtNetUNet(nn.Module):
    """UNet decoder on top of a CoAtNet encoder from timm."""

    def __init__(self, in_channels: int = 3, pretrained: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        try:
            import timm
        except Exception as exc:
            raise RuntimeError("timm is required for CoAtNetUNet (pip install timm)") from exc

        encoder = timm.create_model("coatnet_0_rw_224", pretrained=pretrained, features_only=True, in_chans=in_channels)
        feature_channels = list(encoder.feature_info.channels())

        from .baseline import ConvBlock, DecoderBlock

        self.encoder = encoder
        self.encoder_input_size = (224, 224)
        deepest = feature_channels[-1]
        self.bottleneck = ConvBlock(deepest, deepest, dropout=dropout)

        decoders = []
        curr_in = deepest
        for skip_ch in reversed(feature_channels[:-1]):
            out_ch = max(skip_ch, 32)
            decoders.append(DecoderBlock(curr_in, skip_ch, out_ch, dropout=dropout))
            curr_in = out_ch

        self.decoders = nn.ModuleList(decoders)
        self.head = nn.Conv2d(curr_in, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_input = x
        if x.shape[-2:] != self.encoder_input_size:
            encoder_input = F.interpolate(x, size=self.encoder_input_size, mode="bilinear", align_corners=False)

        features = self.encoder(encoder_input)
        b = features[-1]
        b = self.bottleneck(b)

        y = b
        for decoder, skip in zip(self.decoders, reversed(features[:-1])):
            y = decoder(y, skip)

        logits = self.head(y)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


__all__ = ["CoAtNetUNet"]
