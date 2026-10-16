"""Optional hosted entry point: explicitly configured, verified model download.

Use ``streamlit run deploy_app.py`` when the host cannot provision local files.
Set public release base_url and its trusted sha256 in Streamlit's [model] secrets
section. No configuration means the same honest, model-free preview as app.py.
"""

import os
from pathlib import Path
import tempfile

import streamlit as st

import app
from covid_xray.inference import ModelUnavailableError, inspect_model_artifacts
from scripts.download_model import download_bundle


@st.cache_resource(show_spinner=False)
def provision_model(base_url, sha256):
    # A checksum-specific cache also separates model versions after redeployment.
    if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
        raise ValueError("Configure a trusted lowercase SHA-256 checksum.")
    folder = Path(tempfile.gettempdir()) / "covid-xray-models" / sha256
    model = folder / "model.keras"
    if not model.exists():
        download_bundle(base_url, sha256, folder)
    artifacts = inspect_model_artifacts(model)
    if artifacts.sha256 != sha256:
        raise ValueError("Cached model does not match the configured checksum.")
    return str(model)


def main():
    # Streamlit 1.43 requires page configuration before accessing secrets/cache.
    st.set_page_config(
        page_title="COVID-19 Chest X-Ray | Deep Learning Portfolio",
        page_icon="🩻",
        layout="wide",
    )
    # Streamlit Community Cloud injects this section securely. Both values are
    # public artifact configuration, not dataset or patient credentials.
    try:
        settings = dict(st.secrets.get("model", {}))
    except FileNotFoundError:
        settings = {}
    base_url = os.environ.get("MODEL_BASE_URL") or settings.get("base_url", "")
    checksum = os.environ.get("MODEL_SHA256") or settings.get("sha256", "")
    if base_url or checksum:
        try:
            if not isinstance(base_url, str) or not isinstance(checksum, str) or not base_url or not checksum:
                raise ValueError("Configure both a public HTTPS base_url and its trusted sha256.")
            os.environ["MODEL_PATH"] = provision_model(base_url, checksum)
        except (OSError, ValueError, ModelUnavailableError):
            st.title("Model setup needs attention")
            st.error("The release bundle could not be verified. Check the configured HTTPS release, checksum, and host logs.")
            st.info("Predictions remain unavailable. This experimental project is not a medical diagnosis.")
            st.stop()
    app.main(configure_page=False)


if __name__ == "__main__":
    main()
