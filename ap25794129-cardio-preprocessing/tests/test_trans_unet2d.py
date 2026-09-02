import torch

from cardiac_image_system.models import TransUNet2D


def test_trans_unet2d_preserves_spatial_shape():
    model = TransUNet2D(in_channels=1, out_channels=1, base_channels=16, image_size=128)
    x = torch.randn(2, 1, 128, 128)
    y = model(x)
    assert y.shape == (2, 1, 128, 128)


def test_trans_unet2d_supports_multiclass_output_channels():
    model = TransUNet2D(in_channels=1, out_channels=4, base_channels=16, image_size=96)
    x = torch.randn(2, 1, 96, 96)
    y = model(x)
    assert y.shape == (2, 4, 96, 96)


def test_trans_unet2d_gradients_flow():
    model = TransUNet2D(in_channels=1, out_channels=1, base_channels=16, image_size=64)
    x = torch.randn(2, 1, 64, 64, requires_grad=True)
    y = model(x)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_trans_unet2d_default_matches_pipeline_resolution():
    # The rest of the benchmark always resizes to 256x256; the default image_size must match.
    model = TransUNet2D(in_channels=1, out_channels=1, base_channels=16)
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    assert y.shape == (1, 1, 256, 256)
