import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from covid_xray.data import make_dataset, preprocess_image


class TestDataPipeline(unittest.TestCase):
    def test_batches_preserve_images_labels_and_final_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            image_dir = Path(folder)

            # Distinct intensities reveal image/label ordering mistakes.
            values = [0, 128, 255]
            filenames = ["black.png", "gray.png", "white.png"]

            for filename, value in zip(filenames, values):
                Image.new("L", (20, 20), color=value).save(
                    image_dir / filename
                )

            manifest = pd.DataFrame({
                "filename": filenames,
                "label": ["negative", "positive", "negative"],
            })

            dataset = make_dataset(
                manifest,
                image_dir=image_dir,
                batch_size=2,
                training_mode=False,
            )

            batches = list(dataset.as_numpy_iterator())

            self.assertEqual(len(batches), 2)
            self.assertEqual(batches[0][0].shape, (2, 224, 224, 1))
            self.assertEqual(batches[1][0].shape, (1, 224, 224, 1))

            images = np.concatenate([batch[0] for batch in batches])
            labels = np.concatenate([batch[1] for batch in batches])

            np.testing.assert_array_equal(
                labels,
                [[0.0], [1.0], [0.0]],
            )

            for number, filename in enumerate(filenames):
                expected = preprocess_image(image_dir / filename)
                np.testing.assert_allclose(
                    images[number],
                    expected,
                    rtol=0,
                    atol=0,
                )

            self.assertEqual(images.dtype, np.float32)
            self.assertTrue(np.isfinite(images).all())

    def test_unknown_label_is_rejected(self):
        manifest = pd.DataFrame({
            "filename": ["unused.png"],
            "label": ["unknown"],
        })

        with self.assertRaisesRegex(ValueError, "unknown class"):
            make_dataset(manifest, image_dir=".")


if __name__ == "__main__":
    unittest.main()