"""Streamlit interface for the experimental COVID-19 chest X-ray project."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit executes this file directly. Adding src also supports a clean clone
# before an editable installation; no model or dataset is downloaded here.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from covid_xray.inference import (  # noqa: E402 - import the local src package after bootstrap
    DEFAULT_MODEL_PATH,
    ImageValidationError,
    ModelBundle,
    ModelUnavailableError,
    PredictionError,
    inspect_model_artifacts,
    load_model_bundle,
    predict_image,
    validate_image,
)


@st.cache_resource(show_spinner=False)
def cached_model(path: str, signature: tuple[tuple[int, int], ...]) -> ModelBundle:
    """Reuse an unchanged model between reruns without caching user uploads.

    The signature participates in Streamlit's cache key. Replacing any artifact
    invalidates this resource. Full checksum verification happens before lookup
    and again when the model is first loaded.
    """
    return load_model_bundle(path)


def main(configure_page: bool = True) -> None:
    if configure_page:
        st.set_page_config(
            page_title="COVID-19 Chest X-Ray | Deep Learning Portfolio",
            page_icon="🩻",
            layout="wide",
        )
    st.markdown(
        """
        <style>
        .block-container {max-width: 1180px; padding-top: 2.3rem;}
        .project-label {color: #117a8b; font-size: .8rem; letter-spacing: .12em;
                        text-transform: uppercase; font-weight: 700;}
        </style>
        <p class="project-label">Deep learning · Research portfolio</p>
        """,
        unsafe_allow_html=True,
    )
    st.title("COVID-19 Chest X-Ray Detection")
    st.markdown(
        "Explore a custom convolutional neural network that classifies an image "
        "as **COVID-19**, **Normal**, or **Pneumonia**."
    )
    st.warning(
        "Experimental educational project — not a medical diagnosis. This model "
        "has no established clinical validity. Do not use its output to make "
        "health or treatment decisions."
    )

    model_path = Path(os.environ.get("MODEL_PATH") or DEFAULT_MODEL_PATH).expanduser().resolve()
    bundle = None
    model_error = None
    try:
        artifacts = inspect_model_artifacts(model_path)
        files = (
            artifacts.model_path,
            artifacts.model_path.parent / "class_names.json",
            artifacts.model_path.parent / "model_metadata.json",
        )
        signature = tuple((path.stat().st_mtime_ns, path.stat().st_size) for path in files)
        with st.spinner("Loading the verified trained model…"):
            bundle = cached_model(str(artifacts.model_path), signature)
    except (ModelUnavailableError, OSError) as error:
        model_error = str(error)

    with st.sidebar:
        st.subheader("Model readiness")
        if bundle is None:
            st.info("Awaiting a trained model")
            st.caption("The interface is ready. Predictions require a verified model bundle.")
        else:
            st.success("Trained model loaded")
            st.caption(f"Artifact SHA-256: {bundle.artifacts.sha256[:12]}…")
            st.caption("This confirms artifact integrity, not clinical validity.")
        st.divider()
        st.markdown("**Accepted input**")
        st.caption("PNG or JPEG · up to 10 MB · one image · up to 25 megapixels")
        st.caption("Grayscale and color images are converted to RGB and resized to 224 × 224.")
        st.markdown("**Interpret the scores carefully**")
        st.caption(
            "Scores are the network's relative class outputs. They are not "
            "calibrated probabilities of disease or measures of clinical certainty."
        )
        st.caption("Source, training, evaluation, and deployment instructions are in the GitHub README.")

    if model_error:
        st.info(f"Predictions are unavailable. {model_error}")
        with st.expander("Install a trained model"):
            st.markdown(
                "1. Train and evaluate the model using the repository's Google Colab notebook.\n"
                "2. Export `model.keras`, `class_names.json`, and `model_metadata.json`.\n"
                "3. Place the three files in `artifacts/`, or set `MODEL_PATH` to your `.keras` file.\n"
                "4. Restart this app. See `docs/DEPLOYMENT.md` for a checksum-verified download workflow."
            )
            st.caption("No demonstration or random predictions are substituted for a trained model.")

    upload_column, result_column = st.columns([1, 1], gap="large")
    with upload_column:
        st.subheader("1 · Upload an image")
        upload = st.file_uploader("Choose a chest X-ray image", type=["png", "jpg", "jpeg"])
        st.caption(
            "The app checks file integrity, not image anatomy. It cannot verify "
            "that an uploaded image is a chest X-ray. Remove patient identifiers "
            "and use only images you are authorized to upload."
        )
        image = None
        if upload is not None:
            try:
                image = validate_image(upload.getvalue())
                st.image(image, caption="Uploaded image preview", use_container_width=True)
                st.caption(f"Decoded dimensions: {image.width} × {image.height} pixels")
            except ImageValidationError as error:
                st.error(str(error))
        predict = st.button(
            "Analyze image",
            type="primary",
            disabled=image is None or bundle is None,
            use_container_width=True,
        )
        st.caption("This application processes uploads in memory and does not save uploaded files.")

    with result_column:
        st.subheader("2 · Review model output")
        if not predict:
            st.markdown("Upload a valid image and select **Analyze image** to display model output.")
            if bundle is None:
                st.caption("No disease classification will appear until a trained model is installed.")
        elif image is not None and bundle is not None:
            try:
                with st.spinner("Running inference…"):
                    result = predict_image(bundle, image)
                st.metric("Predicted class", result.class_name)
                st.metric("Top model score", f"{result.confidence:.1%}")
                scores = pd.DataFrame(
                    {"Model score": result.scores}, index=pd.Index(result.class_names, name="Class")
                )
                st.bar_chart(scores, horizontal=True, color="#117a8b")
                table = scores.reset_index()
                st.dataframe(
                    table,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Model score": st.column_config.NumberColumn("Model score", format="%.4f")
                    },
                )
                st.caption(
                    "A high score can still be incorrect. A Normal prediction does not "
                    "rule out illness; a COVID-19 or Pneumonia prediction does not establish it."
                )
            except PredictionError as error:
                st.error(str(error))

    with st.expander("How the model works and where it can fail"):
        st.markdown(
            "The custom CNN uses convolution, pooling, batch normalization, dropout, "
            "and dense layers. Training augmentation applies only to training images; "
            "the exported model includes pixel normalization.\n\n"
            "Public datasets can contain source, age, acquisition, and patient-overlap biases. "
            "Patient grouping and duplicate checks reduce leakage, but do not establish "
            "generalization to a hospital or population. Review the saved test metrics, "
            "confusion matrix, ROC curves, and errors before discussing model performance. "
            "Clinical claims require independent validation that this portfolio does not provide."
        )
    st.divider()
    st.caption("COVID-19 Chest X-Ray Detection using Deep Learning · Educational use")


if __name__ == "__main__":
    main()
