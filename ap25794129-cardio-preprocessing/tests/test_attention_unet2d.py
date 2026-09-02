import torch

from cardiac_image_system.models import AttentionUNet2D


def test_attention_unet2d_preserves_spatial_shape():
    model = AttentionUNet2D(in_channels=1, out_channels=1, base_channels=16)
    x = torch.randn(2, 1, 128, 128)
    y = model(x)
    assert y.shape == (2, 1, 128, 128)


def test_attention_unet2d_supports_multiclass_output_channels():
    model = AttentionUNet2D(in_channels=1, out_channels=4, base_channels=16)
    x = torch.randn(2, 1, 96, 96)
    y = model(x)
    assert y.shape == (2, 4, 96, 96)


def test_attention_unet2d_handles_odd_input_size():
    # Odd spatial sizes exercise the crop/pad alignment path in AttentionUp.
    model = AttentionUNet2D(in_channels=1, out_channels=1, base_channels=16)
    x = torch.randn(1, 1, 97, 101)
    y = model(x)
    assert y.shape == (1, 1, 97, 101)


def test_attention_unet2d_gradients_flow():
    model = AttentionUNet2D(in_channels=1, out_channels=1, base_channels=16)
    x = torch.randn(2, 1, 64, 64, requires_grad=True)
    y = model(x)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
