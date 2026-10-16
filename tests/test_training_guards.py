"""Synthetic verification must never create publishable portfolio results."""

import pytest

from covid_xray.evaluate import evaluate
from covid_xray.train import train


def test_training_rejects_synthetic_fixtures_without_writing_artifacts(manifest, tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    output = tmp_path / "portfolio-artifacts"

    with pytest.raises(ValueError, match="Synthetic"):
        train(manifest_path, output, epochs=1, batch_size=2)
    assert not output.exists()


def test_evaluation_rejects_synthetic_portfolio_results(
    manifest, synthetic_model_files, tmp_path
):
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    model_path, _, _ = synthetic_model_files
    output = tmp_path / "portfolio-evaluation"

    with pytest.raises(ValueError, match="Synthetic"):
        evaluate(manifest_path, model_path, output, batch_size=2)
    assert not output.exists()
