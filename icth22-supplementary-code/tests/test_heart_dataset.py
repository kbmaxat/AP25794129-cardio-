from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.augmentation import augment_pair, sample_parameters, validate_pair
from src.heart_database import build_label, read_p7_volume
from src.prepare_heart_dataset import build_folds, split_for_fold, stable_seed


class HeartDatabaseTests(unittest.TestCase):
    def test_reads_commented_p7_volume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.pgm"
            expected = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
            path.write_bytes(
                b"P7\n# xdim 1\n4 3 2\n255\n" + expected.tobytes()
            )
            np.testing.assert_array_equal(read_p7_volume(path), expected)

    def test_builds_three_class_label(self):
        endocardium = np.zeros((2, 4, 4), dtype=np.uint8)
        epicardium = np.zeros_like(endocardium)
        epicardium[:, 1:4, 1:4] = 255
        endocardium[:, 2:3, 2:3] = 255
        label = build_label(endocardium, epicardium)
        self.assertEqual(set(np.unique(label).tolist()), {0, 1, 2})
        self.assertEqual(int(np.sum(label == 1)), 2)
        self.assertEqual(int(np.sum(label == 2)), 16)

    def test_patient_folds_are_disjoint(self):
        patients = [f"Pat{index:02d}" for index in range(1, 19)]
        folds = build_folds(patients, 5, 42)
        split = split_for_fold(folds, 0)
        self.assertEqual(
            set(split["train"]) | set(split["validation"]) | set(split["test"]),
            set(patients),
        )
        self.assertFalse(set(split["train"]) & set(split["validation"]))
        self.assertFalse(set(split["train"]) & set(split["test"]))
        self.assertFalse(set(split["validation"]) & set(split["test"]))

    def test_augmentation_is_reproducible_and_valid(self):
        image = np.zeros((5, 32, 32), dtype=np.float32)
        label = np.zeros_like(image, dtype=np.uint8)
        image[:, 8:24, 8:24] = 0.8
        label[:, 10:22, 10:22] = 2
        label[:, 13:19, 13:19] = 1
        seed = stable_seed(42, "Pat01", "diastole", "1")

        rng1 = np.random.default_rng(seed)
        parameters1 = sample_parameters(rng1)
        result1 = augment_pair(image, label, parameters1, rng1)
        rng2 = np.random.default_rng(seed)
        parameters2 = sample_parameters(rng2)
        result2 = augment_pair(image, label, parameters2, rng2)

        np.testing.assert_array_equal(result1[0], result2[0])
        np.testing.assert_array_equal(result1[1], result2[1])
        validate_pair(*result1, reference_label=label)


if __name__ == "__main__":
    unittest.main()
