from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    folds: int = 5
    epochs_frozen: int = 10
    epochs_finetune: int = 15
    learning_rate_frozen: float = 1e-4
    learning_rate_finetune: float = 1e-5
    early_stopping_patience: int = 5
    dropout: float = 0.5
    architecture: str = "vgg16"
    pretrained: bool = True
    vgg_blocks_to_unfreeze: int = 2

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
