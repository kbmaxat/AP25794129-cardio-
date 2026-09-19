import numpy as np
import pytest
import torch

from cardiac_image_system.research_temporal.data import neighbors, select_patients, warp_neighbor
from cardiac_image_system.research_temporal.model import BoundedPreprocessor, TemporalSystem


def inputs():
    torch.manual_seed(12)
    return torch.rand(2, 3, 32, 32), torch.rand(2, 2, 32, 32), torch.rand(2, 2, 32, 32), torch.tensor([[-0.02, 0.02], [0.02, 0.04]])


@pytest.mark.parametrize("temporal", [False, True])
def test_initial_identity_and_trainable_residual(temporal):
    args = inputs()
    model = BoundedPreprocessor(temporal)
    output, change = model(*args)
    assert torch.equal(output, args[0][:, :1])
    assert torch.count_nonzero(change) == 0
    output.mean().backward()
    assert model.residual.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("temporal", [False, True])
def test_output_range_and_correction_bound(temporal):
    args = inputs()
    model = BoundedPreprocessor(temporal, epsilon=0.1)
    with torch.no_grad():
        model.residual.weight.fill_(100)
        model.residual.bias.fill_(100)
    output, change = model(*args)
    assert output.min() >= 0 and output.max() <= 1
    assert change.abs().max() <= 0.1 + 1e-6


def test_unreliable_neighbor_cannot_change_output():
    raw, aligned, confidence, times = inputs()
    model = BoundedPreprocessor(True)
    torch.nn.init.normal_(model.residual.weight)
    confidence.zero_()
    first, _ = model(raw, aligned, confidence, times)
    second, _ = model(raw, torch.zeros_like(aligned), confidence, times * 10)
    assert torch.equal(first, second)


def test_neighbors_distinct_and_boundary_policy():
    assert neighbors(0, 10) == [1, 2]
    assert neighbors(9, 10) == [7, 8]
    assert neighbors(4, 10) == [3, 5]
    with pytest.raises(ValueError):
        neighbors(0, 2)


def test_warp_direction_on_known_translation():
    center = np.zeros((24, 24), np.float32)
    center[8:14, 6:11] = 1
    neighbor = np.roll(center, 3, axis=1)
    flow = np.zeros((24, 24, 2), np.float32)
    flow[..., 0] = 3
    warped, _, _, valid = warp_neighbor(neighbor, flow)
    assert np.array_equal(warped[valid], center[valid])
    assert not valid[:, -3:].any()


def test_split_never_draws_from_official_holdouts():
    splits = {"training": [f"p{i}" for i in range(10)], "validation": ["v1"], "testing": ["t1"]}
    config = {"eligible_partition": "training", "external_test_access": False, "split_seed": 1, "train_patients": 6, "dev_patients": 2}
    selected = select_patients(splits, config)
    assert not set(selected["train"]) & set(selected["dev"])
    assert set(selected["train"] + selected["dev"]) <= set(splits["training"])
    assert selected == select_patients(splits, config)
    config["external_test_access"] = True
    with pytest.raises(ValueError):
        select_patients(splits, config)


@pytest.mark.parametrize("mode", ["none", "direct_temporal", "spatial", "temporal"])
def test_system_shape_and_finite_gradients(mode):
    config = {"unet_base_channels": 2, "adapter_channels": 4, "epsilon": 0.1}
    model = TemporalSystem(mode, config)
    logits, image, change = model(*inputs())
    assert logits.shape == image.shape == change.shape == (2, 1, 32, 32)
    logits.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
