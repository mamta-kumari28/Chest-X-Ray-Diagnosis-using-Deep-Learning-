"""Streamlit interface for the trained chest X-ray classifier."""

import json
import tempfile
from pathlib import Path
import os
from covid_xray.model_download import ensure_model

import streamlit as st
from PIL import Image, UnidentifiedImageError


from covid_xray.inference import load_model, predict_image

# Resolve app assets relative to the repository's entry-point file.
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "artifacts/models/best_model.keras"
RESULTS_DIR = PROJECT_ROOT / "artifacts/results"

st.set_page_config(
    page_title="Chest X-Ray Classification",
    page_icon="🩻",
    layout="centered",
)

st.title("Chest X-Ray Classification")
st.caption("Custom TensorFlow/Keras CNN · Educational project")

st.warning(
    "For education and research only. This model is not validated "
    "for clinical diagnosis and must not guide medical decisions."
)

if os.getenv("SHOWCASE_MODE") != "1":
    st.info(
        "Upload a chest X-ray in PNG or JPEG format. The application "
        "does not verify whether an uploaded image is a chest X-ray. "
        "Other images can also receive a prediction."
    )


@st.cache_resource
def get_model():
    verified_path = ensure_model(MODEL_PATH)
    return load_model(verified_path)


with st.expander("Official test results"):
    metrics_path = RESULTS_DIR / "test_metrics.json"

    if metrics_path.is_file():
        metrics = json.loads(
            metrics_path.read_text(encoding="utf-8")
        )

        st.write(
            "Recorded evaluation on 400 official test images "
            "using a fixed threshold of 0.5."
        )

        first, second, third = st.columns(3)
        first.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        second.metric("Positive recall", f"{metrics['recall']:.2%}")
        third.metric("Positive F1", f"{metrics['f1']:.2%}")

        st.write(
            f"False negatives: {metrics['false_negative']} "
            "out of 200 positive images."
        )

        matrix_path = RESULTS_DIR / "test_confusion_matrix.png"
        if matrix_path.is_file():
            with Image.open(matrix_path) as matrix_image:
                st.image(
                    matrix_image.convert("RGB"),
                    caption="Recorded official test confusion matrix",
                )

        st.caption(
            "These are saved dataset-level results, not metrics "
            "calculated from the image you upload."
        )
    else:
        st.info("Saved test metrics are not available.")


# Show project results without requiring publicly hosted model weights.
if os.getenv("SHOWCASE_MODE") == "1":
    st.subheader("Project Showcase")
    st.info(
        "Interactive inference is currently available locally. "
        "This public showcase displays recorded project results."
    )

    st.write(
        "Custom CNN: three convolution blocks with 32, 64, and "
        "128 filters, global average pooling, a dense layer, "
        "dropout, and a sigmoid output. Total parameters: 101,665."
    )

    for filename, caption in [
        ("accuracy_curve.png", "Recorded training and validation accuracy"),
        ("loss_curve.png", "Recorded training and validation loss"),
        ("app_screenshot.png", "Locally tested Streamlit application"),
    ]:
        path = RESULTS_DIR / filename

        if path.is_file():
            with Image.open(path) as image:
                st.image(
                    image.convert("RGB"),
                    caption=caption,
                )

    st.stop()


uploaded = st.file_uploader(
    "Choose a chest X-ray",
    type=["png", "jpg", "jpeg"],
)

if uploaded is not None:
    # Limit uploaded file size before attempting image decoding.
    if uploaded.size > 10 * 1024 * 1024:
        st.error("Please upload an image smaller than 10 MB.")
        st.stop()

    try:
        uploaded.seek(0)
        with Image.open(uploaded) as image:
            if image.format not in {"PNG", "JPEG"}:
                raise ValueError("Only PNG and JPEG images are supported.")

            image.load()
            preview = image.convert("RGB")

        st.image(preview, caption="Uploaded image")

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ):
        st.error("The file could not be read as a supported image.")
        st.stop()

    if st.button("Classify image", type="primary"):
        
        try:
            with st.spinner("Processing image..."):
                model = get_model()

                # Temporary image files are removed after inference.
                with tempfile.TemporaryDirectory() as folder:
                    suffix = Path(uploaded.name).suffix.lower()
                    image_path = Path(folder) / f"upload{suffix}"
                    image_path.write_bytes(uploaded.getvalue())

                    result = predict_image(
                        model,
                        image_path,
                        threshold=0.5,
                    )

            st.subheader(
                f"Model classification: {result['predicted_label']}"
            )
            st.metric(
                "COVID-positive model score",
                f"{result['positive_score']:.4f}",
            )

            st.caption(
                "Decision rule: positive when score > 0.5. "
                "The score is not a calibrated medical probability "
                "or a measure of diagnostic certainty."
            )

        except Exception:
            st.error(
                "Prediction failed. Check the terminal for details."
            )
            import logging
            logging.exception("Image prediction failed")