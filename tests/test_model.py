import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from covid_xray.model import build_model


class TestModel(unittest.TestCase):
    def test_forward_training_and_save_reload(self):
        tf.keras.utils.set_random_seed(42)
        model = build_model()

        images = np.random.default_rng(42).random(
            (2, 224, 224, 1),
            dtype=np.float32,
        )
        labels = np.array([[0.0], [1.0]], dtype=np.float32)

        # Verify binary output shape and range.
        predictions = model(images, training=False).numpy()

        self.assertEqual(predictions.shape, (2, 1))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue((predictions >= 0).all())
        self.assertTrue((predictions <= 1).all())

        # Verify optimization runs and changes the output-layer weights.
        before = model.get_layer("covid_probability").kernel.numpy().copy()

        metrics = model.train_on_batch(
            images,
            labels,
            return_dict=True,
        )

        self.assertTrue(np.isfinite(metrics["loss"]))

        after = model.get_layer("covid_probability").kernel.numpy()
        self.assertFalse(np.array_equal(before, after))

        # Verify saved models preserve inference results.
        expected = model(images, training=False).numpy()

        with tempfile.TemporaryDirectory() as folder:
            model_path = Path(folder) / "smoke_model.keras"
            model.save(model_path)

            restored = tf.keras.models.load_model(
                model_path,
                compile=False,
            )
            actual = restored(images, training=False).numpy()

            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-5,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()