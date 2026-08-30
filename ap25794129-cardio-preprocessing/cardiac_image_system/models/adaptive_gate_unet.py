from __future__ import annotations

import torch
from torch import nn

from cardiac_image_system.models.unet2d import UNet2D


class GateNet(nn.Module):
    """Small CNN that predicts a soft mixing weight over candidate preprocessing modes
    from the raw (unprocessed) image alone.

    This is the learned counterpart to the hand-picked Pθ compositions (Gaussian,
    Wavelet, NLM, CLAHE, Hybrid) evaluated elsewhere in this benchmark: instead of a
    fixed, dataset-wide choice of operator, Pθ becomes image-conditional,
    Pθ(x) = Σ_k G_φ(x)_k · P_k(x), with φ trained jointly with the downstream
    segmentation network on the segmentation loss itself -- so the selection is
    optimized directly for the task, not for an image-quality heuristic.
    """

    def __init__(self, num_modes: int, base_channels: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(base_channels * 4, num_modes)

    def forward(self, raw_image: torch.Tensor) -> torch.Tensor:
        """raw_image: [B, 1, H, W] -> mode weights [B, num_modes] (softmax, sums to 1)."""
        feat = self.features(raw_image).flatten(1)
        return torch.softmax(self.head(feat), dim=1)


class AdaptiveGateUNet(nn.Module):
    """Gate network + U-Net, trained end to end.

    Expects a [B, K, H, W] input stacking the K candidate-preprocessed versions of each
    image (index 0 must be the unprocessed "none" version -- the gate only ever looks at
    that channel, so its decision cannot depend on a classical filter it is meant to be
    an alternative to). The K channels are mixed into a single image by the predicted
    weights, then segmented by a standard compact U-Net. forward() returns only the
    segmentation logits (matching UNet2D's interface) so the existing training/eval loop
    is unchanged; call gate_weights() separately to inspect what the gate selected.
    """

    def __init__(self, num_modes: int, base_channels: int = 32, gate_base_channels: int = 8) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.gate = GateNet(num_modes=num_modes, base_channels=gate_base_channels)
        self.unet = UNet2D(in_channels=1, out_channels=1, base_channels=base_channels)
        self._last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_channel = x[:, 0:1, :, :]
        weights = self.gate(raw_channel)  # [B, K]
        self._last_weights = weights.detach()
        mixed = (x * weights[:, :, None, None]).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        return self.unet(mixed)

    def gate_weights(self) -> torch.Tensor | None:
        """Mode-mixing weights from the most recent forward() call, [B, num_modes]."""
        return self._last_weights
