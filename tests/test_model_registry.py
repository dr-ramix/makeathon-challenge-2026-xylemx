"""Unit tests for the segmentation model registry."""

from __future__ import annotations

import unittest

import torch

from xylemx.models.baseline import build_model, supported_model_names


class ModelRegistryTests(unittest.TestCase):
    def test_registry_contains_requested_families(self) -> None:
        names = set(supported_model_names())
        expected = {
            "small_unet",
            "resnet18_unet",
            "resnet34_unet",
            "resnet50_unet",
            "resnet101_unet",
            "resnet34_unet_cbam",
            "resnet18_fpn",
            "resnet34_fpn",
            "resnet50_fpn",
            "resnet34_fpn_cbam",
            "resnet18_unetpp",
            "resnet50_unetpp",
            "resnet50_unetpp_cbam",
            "resnet34_deeplabv3plus",
            "resnet34_deeplabv3plus_cbam",
            "convnext_tiny_unet",
            "convnext_small_unet",
            "convnext_base_unet",
            "convnext_large_unet",
            "convnext_tiny_fpn",
            "convnext_tiny_unetpp",
            "convnext_tiny_upernet",
            "convnext_small_upernet",
            "convnext_tiny_deeplabv3plus",
            "convnext_tiny_deeplabv3plus_cbam",
            "convnextv2_atto_unet",
            "convnextv2_femto_unet",
            "convnextv2_pico_unet",
            "convnextv2_nano_unet",
            "convnextv2_tiny_unet",
            "convnextv2_small_unet",
            "convnextv2_base_unet",
            "convnextv2_tiny_fpn",
            "convnextv2_tiny_unetpp",
            "convnextv2_tiny_upernet",
            "convnextv2_tiny_deeplabv3plus",
            "coatnet0_unet",
            "coatnet1_unet",
            "coatnet2_unet",
            "coatnet3_unet",
            "coatnet0_fpn",
            "coatnet0_unetpp",
            "coatnet0_upernet",
            "vgg11_unet",
            "vgg13_unet",
            "vgg16_unet",
            "vgg19_unet",
            "vgg16_fpn",
            "vgg16_unetpp",
        }
        self.assertTrue(expected.issubset(names))

    def test_representative_models_forward(self) -> None:
        sample = torch.randn(1, 63, 128, 128)
        model_names = [
            "resnet18_unet",
            "resnet34_fpn",
            "resnet34_unet_cbam",
            "convnext_tiny_deeplabv3plus",
            "convnext_tiny_deeplabv3plus_cbam",
        ]
        for model_name in model_names:
            with self.subTest(model=model_name):
                model = build_model(model_name, in_channels=63, dropout=0.1)
                model.eval()
                with torch.no_grad():
                    logits = model(sample)
                self.assertEqual(tuple(logits.shape), (1, 1, 128, 128))


if __name__ == "__main__":
    unittest.main()
