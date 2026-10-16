"""A small, readable CNN with preprocessing saved inside the Keras model."""

import tensorflow as tf

from covid_xray.config import CLASSES, IMAGE_SIZE


def build_model(seed: int = 42, learning_rate: float = 1e-3) -> tf.keras.Model:
    """Build a compiled CNN accepting RGB pixels in the original [0, 255] range.

    Keras augmentation layers are active during ``fit`` only. Keeping these
    layers and normalization inside the model makes web predictions use the
    same preprocessing as training, without augmenting evaluation images.
    """
    tf.keras.utils.set_random_seed(seed)

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="normalize")(inputs)
    # Small transformations mimic positioning changes without flipping anatomy.
    x = tf.keras.layers.RandomRotation(0.02, fill_mode="nearest", seed=seed)(x)
    x = tf.keras.layers.RandomTranslation(
        0.04, 0.04, fill_mode="nearest", seed=seed + 1
    )(x)
    x = tf.keras.layers.RandomZoom(
        (-0.08, 0.08), fill_mode="nearest", seed=seed + 2
    )(x)

    for block, filters in enumerate((32, 64, 128), start=1):
        # Batch normalization stabilizes training. Bias is redundant before it.
        x = tf.keras.layers.Conv2D(
            filters, 3, padding="same", use_bias=False, name=f"conv_{block}"
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{block}")(x)
        x = tf.keras.layers.Activation("relu", name=f"relu_{block}")(x)
        x = tf.keras.layers.MaxPooling2D(name=f"pool_{block}")(x)
        x = tf.keras.layers.Dropout(0.15, name=f"dropout_{block}")(x)

    # Global pooling avoids the large parameter count of a Flatten + Dense head.
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense")(x)
    x = tf.keras.layers.Dropout(0.4, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(
        len(CLASSES), activation="softmax", name="probabilities"
    )(x)

    model = tf.keras.Model(inputs, outputs, name="covid_xray_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
