"""The deployment page must remain useful before any real model is trained."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_missing_model_startup_is_graceful_and_prediction_is_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "not-trained.keras"))
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()

    assert len(app.exception) == 0
    assert app.title[0].value == "COVID-19 Chest X-Ray Detection"
    assert any("not a medical diagnosis" in item.value for item in app.warning)
    assert any("Predictions are unavailable" in item.value for item in app.info)
    button = next(item for item in app.button if item.label == "Analyze image")
    assert button.disabled
    assert len(app.metric) == 0
