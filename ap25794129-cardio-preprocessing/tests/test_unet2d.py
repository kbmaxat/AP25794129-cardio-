import torch

from cardiac_image_system.models import UNet2D


def test_unet2d_preserves_spatial_shape():
    model = UNet2D(in_channels=1, out_channels=1, base_channels=16)
    x = torch.randn(2, 1, 128, 128)
    y = model(x)
    assert y.shape == (2, 1, 128, 128)


def test_unet2d_supports_multiclass_output_channels():
    model = UNet2D(in_channels=1, out_channels=4, base_channels=16)
    x = torch.randn(2, 1, 96, 96)
    y = model(x)
    assert y.shape == (2, 4, 96, 96)
