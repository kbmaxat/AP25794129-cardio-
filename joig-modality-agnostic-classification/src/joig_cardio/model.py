from __future__ import annotations


def create_model(architecture: str = "vgg16", dropout: float = 0.5, pretrained: bool = True):
    import torch.nn as nn
    from torchvision import models

    architecture = architecture.lower()
    if architecture == "vgg16":
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg16(weights=weights)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 1))
    elif architecture == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 1))
    elif architecture == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.last_channel, 1))
    else:
        raise ValueError("architecture must be vgg16, resnet50, or mobilenet_v2")
    return model


def freeze_backbone(model, architecture: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    head = model.fc if architecture == "resnet50" else model.classifier
    for parameter in head.parameters():
        parameter.requires_grad = True


def unfreeze_for_finetuning(model, architecture: str, vgg_blocks: int = 2) -> None:
    if architecture == "vgg16":
        boundaries = [24, 17, 10, 5, 0]
        start = boundaries[max(0, min(vgg_blocks, 5)) - 1] if vgg_blocks else len(model.features)
        for parameter in model.features[start:].parameters():
            parameter.requires_grad = True
    elif architecture == "resnet50":
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
    else:
        for parameter in model.features[-4:].parameters():
            parameter.requires_grad = True
