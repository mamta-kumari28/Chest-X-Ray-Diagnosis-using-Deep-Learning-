"""Validate hostile/invalid uploads and model-artifact integrity boundaries."""

import io
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from covid_xray import inference
from covid_xray.config import CLASSES


def _encoded_image(format="PNG", mode="L"):
    buffer = io.BytesIO()
    Image.new(mode, (32, 48), color=160).save(buffer, format=format)
    return buffer.getvalue()


@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_supported_images_decode_to_rgb_with_model_input_range(format):
    image = inference.validate_image(_encoded_image(format=format))
    pixels = inference.image_to_array(image)

    assert image.mode == "RGB"
    assert pixels.shape == (1, 224, 224, 3)
    assert pixels.dtype == np.float32
    assert pixels.min() >= 0 and pixels.max() <= 255
    assert pixels.max() > 1  # Prevent accidentally normalizing twice.


@pytest.mark.parametrize("data", [b"", b"not an image", b"%PDF-1.7", b"\x89PNG\r\n\x1a\n"])
def test_invalid_or_truncated_uploads_are_rejected(data):
    with pytest.raises(inference.ImageValidationError):
        inference.validate_image(data)


def test_unsupported_content_is_rejected_even_if_named_as_an_image():
    with pytest.raises(inference.ImageValidationError):
        inference.validate_image(_encoded_image(format="GIF"))


def test_upload_byte_and_pixel_limits_are_enforced(monkeypatch):
    data = _encoded_image()
    monkeypatch.setattr(inference, "MAX_UPLOAD_BYTES", len(data) - 1)
    with pytest.raises(inference.ImageValidationError):
        inference.validate_image(data)
    monkeypatch.setattr(inference, "MAX_UPLOAD_BYTES", len(data) + 1)
    monkeypatch.setattr(inference, "MAX_IMAGE_PIXELS", 32 * 48 - 1)
    with pytest.raises(inference.ImageValidationError):
        inference.validate_image(data)


def test_missing_model_bundle_explains_unavailability(tmp_path):
    with pytest.raises(inference.ModelUnavailableError, match="model bundle|not installed"):
        inference.inspect_model_artifacts(tmp_path / "missing.keras")


@pytest.fixture
def copied_model_files(synthetic_model_files, tmp_path):
    source, _, _ = synthetic_model_files
    for name in ("model.keras", "class_names.json", "model_metadata.json"):
        shutil.copyfile(source.parent / name, tmp_path / name)
    return tmp_path / "model.keras"


def test_modified_model_is_rejected(copied_model_files):
    with copied_model_files.open("ab") as file:
        file.write(b"tampered")
    with pytest.raises(inference.ModelUnavailableError, match="checksum"):
        inference.inspect_model_artifacts(copied_model_files)


@pytest.mark.parametrize(
    "change",
    [
        {"trained": False},
        {"class_names": list(reversed(CLASSES))},
        {"input_size": [128, 128]},
        {"input_size": [224.0, 224.0]},
        {"input_range": [False, 255]},
        {"sha256": "invalid", "model_sha256": "invalid"},
    ],
)
def test_incompatible_metadata_is_rejected(copied_model_files, change):
    path = copied_model_files.parent / "model_metadata.json"
    metadata = json.loads(path.read_text())
    metadata.update(change)
    path.write_text(json.dumps(metadata))

    with pytest.raises(inference.ModelUnavailableError):
        inference.inspect_model_artifacts(copied_model_files)


def test_wrong_external_class_order_is_rejected(copied_model_files):
    path = copied_model_files.parent / "class_names.json"
    path.write_text(json.dumps(list(reversed(CLASSES))))
    with pytest.raises(inference.ModelUnavailableError):
        inference.inspect_model_artifacts(copied_model_files)


def test_real_keras_bundle_load_and_prediction(synthetic_model_files, monkeypatch):
    path, _, _ = synthetic_model_files
    monkeypatch.setenv("MODEL_PATH", str(path))
    bundle = inference.load_model_bundle()
    image = inference.validate_image(_encoded_image())
    result = inference.predict_image(bundle, image)

    assert result.class_name in CLASSES
    assert result.class_names == tuple(CLASSES)
    assert len(result.scores) == len(CLASSES)
    assert all(0 <= value <= 1 for value in result.scores)
    assert result.confidence == max(result.scores)
    assert np.isclose(sum(result.scores), 1.0)
    # This asserts numerical plumbing only, not medical performance.


@pytest.mark.parametrize("scores", [[[np.nan, 0, 1]], [[0.1, 0.2, 0.3]], [[-1, 1, 1]], [[0.5, 0.5]]])
def test_invalid_model_scores_do_not_become_predictions(scores):
    class InvalidOutputModel:
        def predict(self, images, verbose=0):
            return np.asarray(scores)

    bundle = SimpleNamespace(
        model=InvalidOutputModel(), artifacts=SimpleNamespace(class_names=tuple(CLASSES))
    )
    with pytest.raises(inference.PredictionError):
        inference.predict_image(bundle, Image.new("RGB", (32, 32)))
