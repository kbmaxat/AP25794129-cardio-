from __future__ import annotations

import torch
from torch import nn

from cardiac_image_system.models.unet2d import UNet2D


class BoundedPreprocessor(nn.Module):
    def __init__(self, temporal: bool, channels: int = 16, epsilon: float = 0.1):
        super().__init__()
        if not 0 < epsilon <= 1:
            raise ValueError("epsilon must be in (0, 1]")
        self.temporal = temporal
        self.epsilon = epsilon
        self.features = nn.Sequential(
            nn.Conv2d(7 if temporal else 1, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
        )
        self.gate = nn.Conv2d(channels, 1, 1)
        self.residual = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, raw, aligned, confidence, time_offsets):
        center = raw[:, :1]
        if self.temporal:
            trusted = confidence * aligned + (1 - confidence) * center
            times = time_offsets[:, :, None, None].expand(-1, -1, *center.shape[-2:])
            features = torch.cat([center, trusted, confidence, times * confidence], dim=1)
        else:
            features = center
        features = self.features(features)
        change = self.epsilon * torch.sigmoid(self.gate(features)) * torch.tanh(self.residual(features))
        output = (center + change).clamp(0, 1)
        return output, output - center


class TemporalSystem(nn.Module):
    def __init__(self, mode: str, config: dict):
        super().__init__()
        if mode not in {"none", "direct_temporal", "spatial", "temporal"}:
            raise ValueError(mode)
        self.mode = mode
        self.segmenter = UNet2D(
            in_channels=3 if mode == "direct_temporal" else 1,
            base_channels=config["unet_base_channels"],
        )
        self.preprocessor = None
        if mode in {"spatial", "temporal"}:
            self.preprocessor = BoundedPreprocessor(
                temporal=mode == "temporal", channels=config["adapter_channels"], epsilon=config["epsilon"]
            )

    def forward(self, raw, aligned, confidence, time_offsets):
        image = raw[:, :1]
        change = torch.zeros_like(image)
        if self.preprocessor is not None:
            image, change = self.preprocessor(raw, aligned, confidence, time_offsets)
        segmenter_input = raw if self.mode == "direct_temporal" else image
        return self.segmenter(segmenter_input), image, change
