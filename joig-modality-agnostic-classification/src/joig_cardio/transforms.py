from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance


class CardiacTransform:
    def __init__(self, image_size: int = 224, augment: bool = False):
        self.image_size = image_size
        self.augment = augment

    def __call__(self, image: Image.Image, modality: str):
        import torch

        array = np.asarray(image, dtype=np.uint8)
        if modality == "echo":
            array = cv2.medianBlur(array, 3)
        elif modality == "mri":
            array = cv2.GaussianBlur(array, (3, 3), 0.5)
        image = Image.fromarray(array)
        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = image.rotate(random.uniform(-15, 15), resample=Image.Resampling.BILINEAR)
            scale = random.uniform(0.9, 1.1)
            width, height = image.size
            resized = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
            canvas = Image.new("L", image.size)
            x = (width - resized.width) // 2 + random.randint(-max(1, width // 20), max(1, width // 20))
            y = (height - resized.height) // 2 + random.randint(-max(1, height // 20), max(1, height // 20))
            canvas.paste(resized, (x, y))
            image = ImageEnhance.Brightness(canvas).enhance(random.uniform(0.9, 1.1))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        if self.augment and random.random() < 0.25:
            array = np.clip(array + np.random.normal(0, 0.01, array.shape), 0, 1)
        tensor = torch.from_numpy(np.repeat(array[None, ...], 3, axis=0).copy())
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return (tensor - mean) / std
