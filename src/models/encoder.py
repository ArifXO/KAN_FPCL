"""ResNet-18 encoder for single-channel chest X-rays (Stage 2)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

FEATURE_DIM = 512


class ResNet18Encoder(nn.Module):
    """ResNet-18 trunk adapted to 1-channel input, producing [B, 512] features.

    The classification head is replaced with ``nn.Identity`` so the forward
    pass returns the global-average-pooled feature map. Pretrained weights
    are loaded from ImageNet when ``pretrained=True``; the first conv layer
    is replaced to accept single-channel input (mean of the RGB weights to
    preserve initialization scale).
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        backbone.conv1 = new_conv
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.feature_dim = FEATURE_DIM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
