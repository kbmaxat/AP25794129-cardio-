from __future__ import annotations

import cv2
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model, self.activations, self.gradients = model, None, None
        target_layer.register_forward_hook(lambda _m, _i, output: setattr(self, "activations", output))
        target_layer.register_full_backward_hook(lambda _m, _gi, go: setattr(self, "gradients", go[0]))

    def __call__(self, tensor):
        self.model.zero_grad(set_to_none=True)
        self.model(tensor).flatten()[0].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * self.activations).sum(dim=1).relu()[0].detach().cpu().numpy()
        heatmap -= heatmap.min()
        return heatmap / (heatmap.max() + 1e-8)


def target_layer(model, architecture: str):
    import torch.nn as nn
    if architecture == "resnet50": return model.layer4[-1]
    if architecture == "mobilenet_v2": return model.features[-1]
    return [layer for layer in model.features if isinstance(layer, nn.Conv2d)][-1]


def save_overlay(source, heatmap, destination):
    image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(destination), cv2.addWeighted(base, 0.55, color, 0.45, 0))
