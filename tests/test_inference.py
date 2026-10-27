import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from covid_xray.inference import predict_image


class FixedOutputModel:
    """Return controlled scores to test inference decisions."""

    def __init__(self, output):
        self.output = np.asarray(output, dtype=np.float32)

    def __call__(self, batch, training=False):
        if batch.shape != (1, 224, 224, 1):
            raise AssertionError("Unexpected preprocessed image shape.")

        return self.output


class TestInference(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.image_path = Path(self.temp_dir.name) / "image.png"
        Image.new("RGB", (100, 200), color="gray").save(
            self.image_path
        )

    def test_threshold_boundary(self):
        for score, expected_class in [
            (0.49, 0),
            (0.50, 0),
            (0.51, 1),
        ]:
            with self.subTest(score=score):
                result = predict_image(
                    FixedOutputModel([[score]]),
                    self.image_path,
                )

                self.assertEqual(
                    result["predicted_class"], expected_class
                )
                self.assertEqual(
                    result["predicted_label"],
                    "positive" if expected_class else "negative",
                )
                self.assertAlmostEqual(
                    result["positive_score"], score, places=6
                )

    def test_invalid_outputs_are_rejected(self):
        for output in [
            [[np.nan]],
            [[np.inf]],
            [[-0.1]],
            [[1.1]],
            [[0.2, 0.8]],
        ]:
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    predict_image(
                        FixedOutputModel(output),
                        self.image_path,
                    )

    def test_invalid_thresholds_are_rejected(self):
        for threshold in [0, 1, -0.1, np.nan]:
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    predict_image(
                        FixedOutputModel([[0.7]]),
                        self.image_path,
                        threshold=threshold,
                    )


if __name__ == "__main__":
    unittest.main()