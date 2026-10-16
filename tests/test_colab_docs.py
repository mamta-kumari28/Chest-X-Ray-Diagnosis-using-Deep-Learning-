"""Keep Phase 2 Colab paths aligned with the documented clone command."""

from pathlib import Path


def test_phase2_checklist_uses_actual_clone_directory_and_persistent_data_root():
    text = Path("docs/PHASE2_FINAL_CHECKLIST.md").read_text()

    assert "%cd /content/Chest-X-Ray-Diagnosis-using-Deep-Learning-" in text
    assert "/content/covid-xray-project" not in text
    assert '--data-dir "/content/drive/MyDrive/covidx-cxr3a-v7"' in text
