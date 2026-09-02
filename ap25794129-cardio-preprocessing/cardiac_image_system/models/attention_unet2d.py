from __future__ import annotations

import torch
from torch import nn

from cardiac_image_system.models.unet2d import DoubleConv, Down


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018) applied to a skip connection."""

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.gate_conv(gate)
        s = self.skip_conv(skip)
        if g.shape[-2:] != s.shape[-2:]:
            g = nn.functional.interpolate(g, size=s.shape[-2:], mode="bilinear", align_corners=False)
        attention = self.psi(self.relu(g + s))
        return skip * attention


class AttentionUp(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.attention_gate = AttentionGate(
            gate_channels=in_channels // 2,
            skip_channels=skip_channels,
            inter_channels=max(skip_channels // 2, 1),
        )
        self.conv = DoubleConv((in_channels // 2) + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = nn.functional.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        skip = self.attention_gate(gate=x, skip=skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionUNet2D(nn.Module):
    """U-Net with additive attention gates on the skip connections (Oktay et al., 2018).

    Reuses the same encoder topology and channel widths as UNet2D so that the only structural
    difference between the two architectures is the attention mechanism on the skip connections,
    keeping the preprocessing-mode comparison meaningful across both backbones.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.bottleneck = Down(base_channels * 8, base_channels * 16)
        self.up1 = AttentionUp(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = AttentionUp(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = AttentionUp(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = AttentionUp(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.head(x)
