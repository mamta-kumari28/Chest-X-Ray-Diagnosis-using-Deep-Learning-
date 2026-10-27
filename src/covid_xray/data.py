"""Shared image preprocessing for training and inference."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from covid_xray.config import IMAGE_SIZE


def preprocess_image(
    path: str | Path,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Return a padded grayscale image normalized to [0, 1].

    image_size uses Pillow's (width, height) convention.
    Output shape is (height, width, 1).
    """
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)

        # Composite transparent images onto a black background.
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new(
                "RGBA",
                rgba.size,
                (0, 0, 0, 255),
            )
            image = Image.alpha_composite(background, rgba)

        image = image.convert("L")

        # Preserve aspect ratio and pad to the target dimensions.
        image = ImageOps.pad(
            image,
            image_size,
            method=Image.Resampling.BILINEAR,
            color=0,
        )

        pixels = np.asarray(image, dtype=np.float32) / 255.0

    return pixels[..., np.newaxis]


LABEL_MAP = {"negative": 0, "positive": 1}


def make_dataset(
    table,
    image_dir: str | Path,
    batch_size: int = 32,
    training_mode: bool = False,
    seed: int = 42,
):
    """Build batches using the shared image preprocessing function."""
    import tensorflow as tf

    if table.empty:
        raise ValueError("The image manifest is empty.")

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    image_dir = Path(image_dir)
    labels = table["label"].map(LABEL_MAP)

    if labels.isna().any():
        raise ValueError("The manifest contains unknown class labels.")

    paths = [
        str(image_dir / filename)
        for filename in table["filename"]
    ]

    def load_image(path):
        filename = path.numpy().decode("utf-8")
        return preprocess_image(filename)

    def prepare_example(path, label):
        image = tf.py_function(
            func=load_image,
            inp=[path],
            Tout=tf.float32,
        )
        image.set_shape((IMAGE_SIZE[1], IMAGE_SIZE[0], 1))

        label = tf.reshape(tf.cast(label, tf.float32), (1,))
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels.to_numpy(dtype=np.float32))
    )

    # Shuffle filenames before decoding to limit memory use.
    if training_mode:
        dataset = dataset.shuffle(
            buffer_size=len(table),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        prepare_example,
        num_parallel_calls=2,
        deterministic=True,
    )

    return dataset.batch(batch_size).prefetch(1)