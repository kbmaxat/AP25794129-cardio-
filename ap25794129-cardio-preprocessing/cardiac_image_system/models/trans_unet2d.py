from __future__ import annotations

import torch
from torch import nn

from cardiac_image_system.models.unet2d import DoubleConv, Down, Up


class TransformerBottleneck(nn.Module):
    """Multi-head self-attention block operating on flattened spatial tokens
    (TransUNet-style; Chen et al., 2021), used in place of a plain convolutional bottleneck."""

    def __init__(
        self,
        channels: int,
        num_tokens: int,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, channels))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=int(channels * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        tokens = tokens + self.pos_embed
        tokens = self.transformer(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class TransUNet2D(nn.Module):
    """Compact TransUNet-style architecture (Chen et al., 2021): the same convolutional
    encoder/decoder as UNet2D, with a multi-head self-attention transformer block at the
    bottleneck in place of the plain convolutional bottleneck.

    Assumes a fixed input resolution (default 256x256, matching the rest of this benchmark)
    so that the bottleneck token count and positional embedding size are known at construction
    time rather than resized dynamically.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        image_size: int = 256,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)

        bottleneck_channels = base_channels * 16
        bottleneck_size = image_size // 16
        num_tokens = bottleneck_size * bottleneck_size
        self.transformer_bottleneck = TransformerBottleneck(
            channels=bottleneck_channels,
            num_tokens=num_tokens,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
        )

        self.up1 = Up(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.transformer_bottleneck(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.head(x)
