"""A real CPU optimizer step and Keras save/load round trip on synthetic pixels."""

import numpy as np
import tensorflow as tf

from covid_xray.config import CLASSES


def test_custom_cnn_normalization_and_inference_are_consistent(synthetic_model_files):
    _, model, pixels = synthetic_model_files
    required = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.MaxPooling2D,
        tf.keras.layers.BatchNormalization,
        tf.keras.layers.Dropout,
        tf.keras.layers.Dense,
    )
    assert all(any(isinstance(layer, kind) for layer in model.layers) for kind in required)
    assert int(model.optimizer.iterations.numpy()) == 1
    white = np.full((1, 224, 224, 3), 255, dtype=np.float32)
    normalized = model.get_layer("normalize")(white).numpy()
    np.testing.assert_allclose(normalized, 1.0)

    first = model(pixels, training=False).numpy()
    second = model(pixels, training=False).numpy()
    assert first.shape == (2, len(CLASSES))
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(first, second)


def test_saved_keras_model_preserves_predictions(synthetic_model_files):
    path, model, pixels = synthetic_model_files
    restored = tf.keras.models.load_model(path, compile=False, safe_mode=True)
    before = model(pixels, training=False).numpy()
    after = restored(pixels, training=False).numpy()

    np.testing.assert_allclose(after, before, rtol=1e-5, atol=1e-6)
