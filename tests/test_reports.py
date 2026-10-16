"""Check saved EDA and metric mathematics using explicit temporary fixtures."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import covid_xray.evaluate as evaluation
from covid_xray.config import CLASSES
from covid_xray.eda import create_report


def test_eda_saves_accurate_counts_and_readable_figures(manifest, tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    output = tmp_path / "synthetic-eda-only"
    summary = create_report(manifest_path, output)
    counts = pd.read_csv(output / "class_counts.csv", index_col=0)
    dimensions = pd.read_csv(output / "sample_dimensions.csv")

    assert summary["images"].sum() == len(manifest)
    assert counts.sum().sum() == len(manifest)
    assert counts.sum(axis=1).to_dict() == manifest["label"].value_counts().to_dict()
    assert len(dimensions) == len(manifest[manifest["split"] == "train"])
    assert set(dimensions["width"]) == {36}
    assert set(dimensions["height"]) == {28}
    for name in ("class_distribution.png", "training_samples_PRIVATE.png", "image_dimensions.png"):
        with Image.open(output / name) as image:
            image.verify()


def _metric_fixture(image_index, tmp_path, monkeypatch, probabilities):
    """Isolate evaluation math from fitting; no clinical result is produced."""
    frame = image_index.groupby("label", sort=False).head(2).copy()
    frame["split"] = "test"
    frame["source"] = "unit-test-metric-fixture-not-clinical"
    manifest_path = tmp_path / "fixture-manifest.csv"
    frame.to_csv(manifest_path, index=False)
    model_path = tmp_path / "fixture.keras"
    model_path.write_bytes(b"unit-test placeholder; never loaded or deployed")
    metadata = {
        "trained": True,
        "class_names": list(CLASSES),
        "input_size": [224, 224],
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    (tmp_path / "model_metadata.json").write_text(json.dumps(metadata))

    class KnownFixtureOutputs:
        def predict(self, dataset, verbose=0):
            return np.asarray(probabilities)

    monkeypatch.setattr(evaluation, "load_manifest", lambda path: frame)
    monkeypatch.setattr(evaluation, "make_dataset", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        evaluation.tf.keras.models, "load_model", lambda *args, **kwargs: KnownFixtureOutputs()
    )
    return manifest_path, model_path


def test_evaluation_saves_correct_metrics_roc_and_false_prediction_analysis(
    image_index, tmp_path, monkeypatch
):
    # Ground truth: COVID, COVID, Normal, Normal, Pneumonia, Pneumonia.
    # Known errors: one COVID -> Pneumonia and one Pneumonia -> COVID.
    probabilities = [
        [0.8, 0.1, 0.1],
        [0.1, 0.2, 0.7],
        [0.1, 0.8, 0.1],
        [0.2, 0.6, 0.2],
        [0.7, 0.1, 0.2],
        [0.1, 0.2, 0.7],
    ]
    manifest, model = _metric_fixture(image_index, tmp_path, monkeypatch, probabilities)
    output = tmp_path / "unit-test-metrics-only"
    result = evaluation.evaluate(manifest, model, output)

    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["confusion_matrix"] == [[1, 0, 1], [0, 2, 0], [1, 0, 1]]
    for average in ("macro", "weighted"):
        for name in ("precision", "recall", "f1"):
            assert result[average][name] == pytest.approx(2 / 3)
    assert result["roc_auc_ovr"] == pytest.approx(
        {"COVID-19": 0.625, "Normal": 1.0, "Pneumonia": 0.75}
    )
    assert result["roc_auc_macro_ovr"] == pytest.approx(19 / 24)
    assert result["test_images"] == result["test_patients"] == 6
    errors = json.loads((output / "error_analysis.json").read_text())
    assert errors["COVID-19"]["false_negatives"] == 1
    assert errors["COVID-19"]["false_positives"] == 1
    assert errors["Normal"]["false_negatives"] == 0
    assert errors["Normal"]["false_positives"] == 0
    assert errors["Pneumonia"]["false_negatives"] == 1
    assert errors["Pneumonia"]["false_positives"] == 1
    assert len(pd.read_csv(output / "predictions.csv")) == 6
    for name in ("confusion_matrix.png", "roc_curves.png", "false_predictions.png"):
        with Image.open(output / name) as image:
            image.verify()
    assert json.loads((output / "metrics.json").read_text())["accuracy"] == pytest.approx(4 / 6)
    assert (output / "classification_report.txt").is_file()
    assert (output / "classification_report.json").is_file()


def test_undefined_precision_is_explicitly_null_in_saved_metrics(
    image_index, tmp_path, monkeypatch
):
    probabilities = [[0.2, 0.6, 0.2]] * 6
    manifest, model = _metric_fixture(image_index, tmp_path, monkeypatch, probabilities)
    output = tmp_path / "undefined-metrics-test-only"
    result = evaluation.evaluate(manifest, model, output)
    saved = (output / "metrics.json").read_text()

    assert result["classification_report"]["COVID-19"]["precision"] is None
    assert result["classification_report"]["Pneumonia"]["precision"] is None
    assert len(result["undefined_metrics"]) == 2
    assert "NaN" not in saved
    assert "Infinity" not in saved
    json.loads(saved)
