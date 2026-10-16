"""Download a reviewed model bundle from HTTPS, verifying its published checksum.

Example: python scripts/download_model.py --base-url https://host/release \
    --sha256 <trusted-published-model-sha256> --output artifacts/models
Never load model files from unknown publishers. A checksum proves integrity,
not model quality or clinical validity.
"""

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from covid_xray.inference import inspect_model_artifacts


class HTTPSRedirects(HTTPRedirectHandler):
    """Reject an HTTP hop instead of checking only the final response URL."""

    def redirect_request(self, request, response, code, message, headers, new_url):
        if urlsplit(new_url).scheme != "https":
            raise ValueError("Artifact redirects must remain on HTTPS.")
        return super().redirect_request(request, response, code, message, headers, new_url)


def download_bundle(base_url, sha256, output):
    parts = urlsplit(base_url)
    if parts.scheme != "https" or not parts.hostname:
        raise ValueError("Use a trusted HTTPS release URL.")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("Use a public release URL without embedded credentials, query, or fragment.")
    if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
        raise ValueError("Supply the publisher's lowercase SHA-256 checksum.")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = ["model.keras", "class_names.json", "model_metadata.json"]
    if any((output / name).exists() for name in names):
        raise FileExistsError("Model bundle already exists; use a new output directory.")
    opener = build_opener(HTTPSRedirects())
    with tempfile.TemporaryDirectory(dir=output) as directory:
        folder = Path(directory)
        for name in names:
            limit = 512 * 1024 * 1024 if name == "model.keras" else 1024 * 1024
            with opener.open(f"{base_url.rstrip('/')}/{name}", timeout=60) as response:
                if urlsplit(response.url).scheme != "https":
                    raise ValueError("Download redirected away from HTTPS.")
                size = 0
                with (folder / name).open("wb") as file:
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > limit:
                            raise ValueError(f"Artifact exceeds size limit: {name}")
                        file.write(chunk)
        with (folder / "model.keras").open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        if digest != sha256:
            raise ValueError("Downloaded model does not match the trusted published checksum.")
        inspect_model_artifacts(folder / "model.keras")
        # Only promote files once all integrity and metadata checks pass.
        for name in names:
            os.replace(folder / name, output / name)
    print(f"Verified model bundle saved under {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", default="artifacts/models")
    args = parser.parse_args()
    try:
        download_bundle(args.base_url, args.sha256, args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Download failed: {error}\n")


if __name__ == "__main__":
    main()
