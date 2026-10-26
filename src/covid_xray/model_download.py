"""Download and verify the published model checkpoint."""

import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

from covid_xray.config import MODEL_SHA256


MODEL_URL = (
    "https://github.com/mamta-kumari28/"
    "Chest-X-Ray-Diagnosis-using-Deep-Learning-/"
    "releases/download/model-v1/best_model.keras"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def ensure_model(destination: str | Path) -> Path:
    """Reuse a valid checkpoint or download a verified copy."""
    destination = Path(destination)

    if destination.exists():
        if file_sha256(destination) != MODEL_SHA256:
            raise ValueError("Existing model checksum mismatch.")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            suffix=".download",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)

            with urlopen(MODEL_URL, timeout=60) as response:
                total_bytes = 0

                while chunk := response.read(1024 * 1024):
                    total_bytes += len(chunk)

                    if total_bytes > 20 * 1024 * 1024:
                        raise ValueError("Unexpectedly large model download.")

                    temporary.write(chunk)

        if file_sha256(temporary_path) != MODEL_SHA256:
            raise ValueError("Downloaded model checksum mismatch.")

        # Publish the file only after successful verification.
        os.replace(temporary_path, destination)
        return destination

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)