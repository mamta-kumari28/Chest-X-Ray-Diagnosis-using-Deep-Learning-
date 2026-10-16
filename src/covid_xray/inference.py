"""Validate local model artifacts and run cautious, reproducible image inference.

TensorFlow is imported only when a complete, checksum-verified artifact bundle
is available. This keeps the app useful before training and makes malformed
upload handling independent of a heavyweight machine-learning installation.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from covid_xray.config import CLASSES, IMAGE_SIZE

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "model.keras"


class ImageValidationError(ValueError):
    """An uploaded file cannot be safely decoded as a supported image."""


class ModelUnavailableError(RuntimeError):
    """A complete, compatible, verified trained model is not available."""


class PredictionError(RuntimeError):
    """The model could not produce valid class probabilities."""


@dataclass(frozen=True)
class ModelArtifacts:
    """A validated artifact bundle; extra metadata records training provenance."""

    model_path: Path
    class_names: tuple[str, ...]
    metadata: dict[str, Any]
    sha256: str


@dataclass(frozen=True)
class ModelBundle:
    """A loaded Keras model and the artifacts that describe it."""

    model: Any
    artifacts: ModelArtifacts


@dataclass(frozen=True)
class Prediction:
    """The model's relative class scores; these are not clinical probabilities."""

    class_name: str
    confidence: float
    scores: tuple[float, ...]
    class_names: tuple[str, ...]


def validate_image(data: bytes) -> Image.Image:
    """Decode one PNG or JPEG safely and return an independent RGB image.

    File extensions and browser MIME types are not proof of file content. Pillow
    checks the actual format, verifies the file, and then decodes it in a second
    pass. A pixel limit also protects against a tiny compressed image expanding
    into an unreasonable amount of memory.
    """
    if not isinstance(data, bytes) or not data:
        raise ImageValidationError("Upload a nonempty PNG or JPEG image.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageValidationError("The image exceeds the 10 MB upload limit.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"PNG", "JPEG"}:
                    raise ImageValidationError("Only PNG and JPEG image content is supported.")
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ImageValidationError("The image exceeds the 25 megapixel safety limit.")
                if getattr(image, "n_frames", 1) != 1:
                    raise ImageValidationError("Upload a single image, not an animated file.")
                image.verify()

            # Reopen after verify(): it deliberately consumes the original file.
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                image = ImageOps.exif_transpose(image)
                return image.convert("RGB").copy()
    except ImageValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ImageValidationError("The image is too large to decode safely.") from error
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as error:
        raise ImageValidationError("The image is corrupt or is not a supported PNG/JPEG file.") from error


def image_to_array(image: Image.Image) -> np.ndarray:
    """Resize to 224 × 224 and retain 0..255 pixels for embedded normalization."""
    image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    # Training uses a Rescaling layer inside the saved model. Dividing by 255
    # here would normalize twice and silently change the input distribution.
    return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)


def _read_json(path: Path) -> Any:
    """Read small metadata files and reject duplicate JSON object keys."""
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON keys are not allowed.")
            result[key] = value
        return result

    try:
        if path.stat().st_size > 100_000:
            raise ValueError("Metadata file is unexpectedly large.")
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    except (OSError, UnicodeError, ValueError) as error:
        raise ModelUnavailableError(f"Cannot read valid model metadata: {path.name}.") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ModelUnavailableError("The model file cannot be read.") from error
    return digest.hexdigest()


def inspect_model_artifacts(model_path: str | Path | None = None) -> ModelArtifacts:
    """Require the model, exact class ordering, input contract, and checksum.

    Checksums establish file integrity relative to the provided metadata. Obtain
    both files from a source you trust; a matching checksum is not evidence that
    an arbitrary model or its predictions are safe for medical use.
    """
    path = Path(model_path or os.environ.get("MODEL_PATH") or DEFAULT_MODEL_PATH).expanduser().resolve()
    if path.suffix != ".keras":
        raise ModelUnavailableError("MODEL_PATH must point to a .keras model file.")
    required_paths = (path, path.parent / "class_names.json", path.parent / "model_metadata.json")
    if any(not item.is_file() for item in required_paths):
        raise ModelUnavailableError(
            "A trained model bundle is not installed. Add the .keras model, "
            "class_names.json, and model_metadata.json to the same directory."
        )

    class_names = _read_json(required_paths[1])
    metadata = _read_json(required_paths[2])
    if not isinstance(class_names, list) or class_names != list(CLASSES):
        raise ModelUnavailableError("class_names.json does not match the project's class order.")
    if not isinstance(metadata, dict) or metadata.get("trained") is not True:
        raise ModelUnavailableError("Metadata must explicitly identify a trained model.")
    if metadata.get("class_names") != list(CLASSES):
        raise ModelUnavailableError("Model metadata has an incompatible class order.")
    size = metadata.get("input_size")
    if not isinstance(size, list) or any(type(value) is not int for value in size) or size != list(IMAGE_SIZE):
        raise ModelUnavailableError("Model metadata must declare input_size [224, 224].")
    value_range = metadata.get("input_range")
    if (
        not isinstance(value_range, list)
        or any(type(value) not in (int, float) for value in value_range)
        or value_range != [0, 255]
    ):
        raise ModelUnavailableError("Model metadata must declare input_range [0, 255].")
    checksum = metadata.get("sha256")
    if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        raise ModelUnavailableError("Model metadata must contain a lowercase SHA-256 checksum.")
    if _sha256(path) != checksum:
        raise ModelUnavailableError("The model checksum does not match model_metadata.json.")
    return ModelArtifacts(path, tuple(class_names), metadata, checksum)


def load_model_bundle(model_path: str | Path | None = None) -> ModelBundle:
    """Load only verified Keras artifacts; do not download or create a model."""
    artifacts = inspect_model_artifacts(model_path)
    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(artifacts.model_path, compile=False, safe_mode=True)
        if tuple(model.input_shape) != (None, *IMAGE_SIZE, 3):
            raise ModelUnavailableError("The model input shape must be (None, 224, 224, 3).")
        if tuple(model.output_shape) != (None, len(CLASSES)):
            raise ModelUnavailableError("The model must output one score for each of the three classes.")
    except ModelUnavailableError:
        raise
    except Exception as error:
        # Dependency and deserialization failures should keep the uploader page
        # usable instead of crashing the application with a TensorFlow traceback.
        raise ModelUnavailableError(
            "The verified model could not be loaded. Check the TensorFlow version and model export."
        ) from error
    if _sha256(artifacts.model_path) != artifacts.sha256:
        raise ModelUnavailableError("The model file changed while it was loading.")
    return ModelBundle(model, artifacts)


def predict_image(bundle: ModelBundle, image: Image.Image) -> Prediction:
    """Run inference and reject nonfinite, malformed, or nonprobability output."""
    try:
        scores = np.asarray(bundle.model.predict(image_to_array(image), verbose=0), dtype=np.float64)
    except Exception as error:
        raise PredictionError("Prediction failed. Check the model and its input contract.") from error
    if (
        scores.shape != (1, len(CLASSES))
        or not np.isfinite(scores).all()
        or np.any(scores < 0)
        or np.any(scores > 1)
        or not np.isclose(scores[0].sum(), 1.0, atol=1e-3)
    ):
        raise PredictionError("The model did not return three valid softmax probabilities.")
    values = tuple(float(value) for value in scores[0])
    index = int(np.argmax(scores[0]))
    return Prediction(bundle.artifacts.class_names[index], values[index], values, bundle.artifacts.class_names)
