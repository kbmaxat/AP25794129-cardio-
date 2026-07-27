import torch

from cardiac_image_system.models import UNet2D


def test_unet2d_preserves_spatial_shape():
    model = UNet2D(in_channels=1, out_channels=1, base_channels=16)
    x = torch.randn(2, 1, 128, 128)
    y = model(x)
    assert y.shape == (2, 1, 128, 128)
