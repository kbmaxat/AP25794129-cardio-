"""Axis adapter for the evaluated CAMUS-trained EchoNet checkpoint family.

This version records the post-hoc EchoNet VAL diagnostic convention only. It is not a
general medical-image orientation rule and does not establish a canonical physical
orientation for other datasets or checkpoints.
"""

from __future__ import annotations

import numpy as np


class CamusEchoNetAxisAdapterV1:
    """Transpose 2D EchoNet inputs for the evaluated CAMUS-trained checkpoints.

    Apply :meth:`prepare_input` to the prepared 2D image before inference, then apply
    :meth:`restore_prediction` to the model's 2D probability map before resizing it
    back to the native EchoNet grid. Keep the target mask in its original coordinates.
    """

    version = "camus-echonet-axis-transpose-v1"

    @staticmethod
    def _validate_2d(array: np.ndarray, name: str) -> None:
        if not isinstance(array, np.ndarray) or array.ndim != 2:
            raise ValueError(f"{name} must be a 2D NumPy array")
        if 0 in array.shape:
            raise ValueError(f"{name} must have non-empty dimensions")

    def prepare_input(self, image: np.ndarray) -> np.ndarray:
        """Transpose a prepared image and return a contiguous, independent copy."""
        self._validate_2d(image, "image")
        return image.T.copy()

    def restore_prediction(self, probability_map: np.ndarray) -> np.ndarray:
        """Undo the input axis permutation on a 2D probability map."""
        self._validate_2d(probability_map, "probability_map")
        return probability_map.T.copy()
