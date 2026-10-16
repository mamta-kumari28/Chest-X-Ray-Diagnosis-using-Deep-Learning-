"""Small, synthetic fixtures; these images are never clinical training data."""

import os
import hashlib
import json

# Keep the verification suite usable on small cloud CPU instances.
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from covid_xray.config import CLASSES


@pytest.fixture
def image_index(tmp_path):
    """Use distinct pixels and patients so leakage checks have real files to read."""
    rows = []
    rng = np.random.default_rng(42)
    for label in CLASSES:
        for number in range(12):
            path = tmp_path / f"{label}-{number}.png"
            pixels = rng.integers(0, 256, size=(28, 36, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(path)
            rows.append(
                {
                    "path": str(path),
                    "label": label,
                    "patient_id": f"{label}-patient-{number}",
                    "source": "synthetic-test-fixture",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def manifest(image_index):
    from covid_xray.data import build_manifest

    return build_manifest(image_index)


@pytest.fixture(scope="module")
def synthetic_model_files(tmp_path_factory):
    """Optimize one synthetic batch to test serialization, never medical quality."""
    import tensorflow as tf

    from covid_xray.model import build_model

    folder = tmp_path_factory.mktemp("synthetic-model-only")
    model = build_model(seed=42)
    pixels = np.random.default_rng(42).uniform(0, 255, (2, 224, 224, 3)).astype("float32")
    labels = np.array([0, 1], dtype="int32")
    loss, accuracy = model.train_on_batch(pixels, labels)
    assert np.isfinite(loss)
    assert np.isfinite(accuracy)
    model_path = folder / "model.keras"
    model.save(model_path)
    checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata = {
        "trained": True,
        "class_names": list(CLASSES),
        "input_size": [224, 224],
        "input_range": [0, 255],
        "sha256": checksum,
        "model_sha256": checksum,
        "training_note": "Synthetic unit-test batch only; not a diagnostic model.",
    }
    (folder / "class_names.json").write_text(json.dumps(list(CLASSES)))
    (folder / "model_metadata.json").write_text(json.dumps(metadata))
    yield model_path, model, pixels
    tf.keras.backend.clear_session()
