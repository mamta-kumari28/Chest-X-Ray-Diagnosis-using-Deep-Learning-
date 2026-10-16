"""Check hosted startup and download failures without making network requests."""

import importlib
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request

import pytest
from streamlit.testing.v1 import AppTest

from covid_xray.config import CLASSES


@pytest.fixture
def downloader(monkeypatch):
    module = importlib.import_module("scripts.download_model")

    def fail_if_network_is_used(*args, **kwargs):
        pytest.fail("Invalid or missing model configuration must not access the network.")

    monkeypatch.setattr(module, "build_opener", fail_if_network_is_used)
    return module


@pytest.mark.parametrize("base_url", ["http://example.com/models", "file:///tmp/models", "", "https://"])
def test_downloader_rejects_non_https_before_network(downloader, tmp_path, base_url):
    output = tmp_path / "new-output"
    with pytest.raises(ValueError, match="HTTPS"):
        downloader.download_bundle(base_url, "0" * 64, output)
    assert not output.exists()


@pytest.mark.parametrize("checksum", ["", "0" * 63, "0" * 65, "A" * 64, "x" * 64])
def test_downloader_rejects_bad_checksum_before_network(downloader, tmp_path, checksum):
    output = tmp_path / "new-output"
    with pytest.raises(ValueError, match="checksum"):
        downloader.download_bundle("https://example.com/models", checksum, output)
    assert not output.exists()


@pytest.mark.parametrize("name", ["model.keras", "class_names.json", "model_metadata.json"])
def test_downloader_preserves_existing_artifact_and_user_files(downloader, tmp_path, name):
    existing = tmp_path / name
    existing.write_bytes(b"existing user-owned artifact")
    note = tmp_path / "notes.txt"
    note.write_text("Preserve this file too.")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    with pytest.raises(FileExistsError):
        downloader.download_bundle("https://example.com/models", "0" * 64, tmp_path)

    after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert after == before


def _mock_release_transport(downloader, monkeypatch):
    # This placeholder checks transport/integrity only; Keras never loads it.
    model_bytes = b"synthetic transport-test placeholder; not a trained model"
    checksum = hashlib.sha256(model_bytes).hexdigest()
    metadata = {
        "trained": True,
        "class_names": list(CLASSES),
        "input_size": [224, 224],
        "input_range": [0, 255],
        "sha256": checksum,
        "purpose": "Unit-test transport fixture only, never deploy.",
    }
    files = {
        "model.keras": model_bytes,
        "class_names.json": json.dumps(list(CLASSES)).encode(),
        "model_metadata.json": json.dumps(metadata).encode(),
    }
    requests = []

    def open_fixture(url, timeout):
        assert timeout > 0
        name = url.rsplit("/", 1)[-1]
        requests.append(name)
        response = io.BytesIO(files[name])
        response.url = url
        return response

    monkeypatch.setattr(
        downloader, "build_opener", lambda *handlers: SimpleNamespace(open=open_fixture)
    )
    return checksum, files, requests


def test_mocked_download_verifies_all_files_before_promotion(downloader, monkeypatch, tmp_path):
    checksum, files, requests = _mock_release_transport(downloader, monkeypatch)
    output = tmp_path / "verified-test-bundle"
    downloader.download_bundle("https://example.com/models", checksum, output)

    assert requests == list(files)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == files


def test_mocked_download_checksum_failure_preserves_files_without_partial_bundle(
    downloader, monkeypatch, tmp_path
):
    _, files, requests = _mock_release_transport(downloader, monkeypatch)
    note = tmp_path / "notes.txt"
    note.write_text("User notes must survive failed downloads.")
    before = note.read_bytes()

    with pytest.raises(ValueError, match="published checksum"):
        downloader.download_bundle("https://example.com/models", "0" * 64, tmp_path)

    assert requests == list(files)
    assert list(tmp_path.iterdir()) == [note]
    assert note.read_bytes() == before


def test_downloader_rejects_https_to_http_redirect_before_following(downloader):
    handler = downloader.HTTPSRedirects()
    with pytest.raises(ValueError, match="remain on HTTPS"):
        handler.redirect_request(
            Request("https://example.com/models"), None, 302, "Found", {}, "http://example.com/model.keras"
        )


def test_hosted_app_missing_configuration_starts_without_network(
    downloader, monkeypatch, tmp_path
):
    monkeypatch.delenv("MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_SHA256", raising=False)
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.keras"))
    path = Path(__file__).resolve().parents[1] / "deploy_app.py"
    app = AppTest.from_file(str(path), default_timeout=30).run()

    assert len(app.exception) == 0
    assert app.title[0].value == "COVID-19 Chest X-Ray Detection"
    assert any("Predictions are unavailable" in item.value for item in app.info)
    assert app.button[0].disabled
    assert len(app.metric) == 0


@pytest.mark.parametrize(
    "base_url,checksum",
    [
        ("http://example.com/models", "0" * 64),
        ("https://example.com/models", "invalid-checksum"),
        ("", "0" * 64),
        ("https://example.com/models", ""),
    ],
)
def test_hosted_app_bad_model_configuration_is_graceful_without_network(
    downloader, monkeypatch, tmp_path, base_url, checksum
):
    monkeypatch.setenv("MODEL_BASE_URL", base_url)
    monkeypatch.setenv("MODEL_SHA256", checksum)
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.keras"))
    # Keep this test independent of any model caches from other sessions.
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    path = Path(__file__).resolve().parents[1] / "deploy_app.py"
    app = AppTest.from_file(str(path), default_timeout=30).run()

    assert len(app.exception) == 0
    assert app.title[0].value == "Model setup needs attention"
    assert any("could not be verified" in item.value for item in app.error)
    assert any("not a medical diagnosis" in item.value for item in app.info)
    assert len(app.metric) == 0
