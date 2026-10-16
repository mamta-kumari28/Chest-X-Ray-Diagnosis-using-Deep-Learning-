"""Phase 2 metadata auditing and EDA use only tiny local fixtures."""

import json

import pandas as pd
import pytest
from PIL import Image

from covid_xray.config import CLASSES
from covid_xray.data import audit_metadata, audit_metadata_file, normalize_metadata
from covid_xray.eda import create_dataset_report


def _metadata(rows):
    return pd.DataFrame(rows, columns=["image_path", "label", "patient_id", "original_partition"])


def test_metadata_normalization_supports_legacy_path_without_inventing_labels(tmp_path):
    frame = pd.DataFrame(
        [{"path": "scan.png", "label": "Normal", "patient_id": "p1", "original_partition": "train"}]
    )
    normalized = normalize_metadata(frame)

    assert normalized.loc[0, "image_path"] == "scan.png"
    assert "path" not in normalized
    with pytest.raises(ValueError, match="Unknown labels"):
        normalize_metadata(normalized.assign(label="Other"))


def test_audit_counts_metadata_accessible_valid_missing_corrupt_and_unsupported(tmp_path):
    Image.new("L", (31, 27), color=80).save(tmp_path / "valid.png")
    (tmp_path / "corrupt.jpg").write_bytes(b"not a real image")
    (tmp_path / "unsupported.txt").write_text("not an image")
    frame = _metadata(
        [
            ("valid.png", CLASSES[0], "p1", "train"),
            ("missing.png", CLASSES[1], "p2", "train"),
            ("corrupt.jpg", CLASSES[2], "p3", "test"),
            ("unsupported.txt", CLASSES[1], "p4", "test"),
        ]
    )

    audited, report = audit_metadata(frame, tmp_path)

    assert report["metadata_records"] == 4
    assert report["accessible_image_files"] == 3
    assert report["valid_images"] == 1
    assert report["missing_files"] == 1
    assert report["corrupt_files"] == 1
    assert report["unsupported_files"] == 1
    assert audited.set_index("image_path").loc["valid.png", "width"] == 31
    assert audited.set_index("image_path").loc["valid.png", "height"] == 27
    assert set(audited.validation_status) == {"valid", "missing", "corrupt", "unsupported"}


def test_audit_flags_patient_overlap_and_exact_duplicate_content(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (20, 18), color=(15, 40, 80)).save(first)
    Image.open(first).save(second)
    frame = _metadata(
        [
            (first.name, CLASSES[0], "same-patient", "train"),
            (second.name, CLASSES[0], "same-patient", "test"),
        ]
    )

    _, report = audit_metadata(frame, tmp_path)

    assert report["patient_overlap_count"] == 1
    assert report["exact_duplicate_groups"] == 1
    assert report["exact_duplicate_records"] == 2
    assert report["perceptual_match_groups"] == 1


def test_audit_retains_ambiguous_importer_status_without_selecting_a_file(tmp_path):
    Image.new("L", (12, 12)).save(tmp_path / "same.png")
    frame = _metadata([("same.png", CLASSES[0], "p1", "train")]).assign(
        resolution_status="ambiguous", candidate_count="2"
    )

    audited, report = audit_metadata(frame, tmp_path)

    assert audited.loc[0, "validation_status"] == "ambiguous"
    assert not audited.loc[0, "file_exists"]
    assert report["ambiguous_records"] == 1
    assert report["accessible_image_files"] == 0


def test_phase2_eda_saves_only_observed_counts_and_real_fixture_figures(tmp_path):
    rows = []
    for class_number, label in enumerate(CLASSES):
        path = tmp_path / f"{class_number}.png"
        Image.new("L", (24 + class_number, 30 + class_number), color=50 + class_number).save(path)
        rows.append((path.name, label, f"patient-{class_number}", "train"))
    metadata = _metadata(rows)
    metadata_path = tmp_path / "metadata.csv"
    validated_path = tmp_path / "validated.csv"
    report_path = tmp_path / "quality.json"
    output = tmp_path / "eda"
    metadata.to_csv(metadata_path, index=False)

    audited, report = audit_metadata_file(
        metadata_path, tmp_path, validated_path, report_path
    )
    summary = create_dataset_report(audited, report, output)

    assert summary.metadata_records.sum() == 3
    assert summary.valid_image_records.sum() == 3
    assert len(pd.read_csv(output / "dataset_summary.csv")) == 3
    assert json.loads((output / "data_quality_report.json").read_text())["valid_images"] == 3
    for name in ("class_distribution.png", "sample_images.png", "image_dimensions.png"):
        with Image.open(output / name) as image:
            image.verify()


def test_phase2_eda_does_not_create_fake_samples_when_every_image_is_missing(tmp_path):
    frame = _metadata([("missing.png", CLASSES[0], "p1", "train")])
    audited, report = audit_metadata(frame, tmp_path)
    output = tmp_path / "eda"

    create_dataset_report(audited, report, output)

    assert (output / "class_distribution.png").is_file()
    assert not (output / "sample_images.png").exists()
    assert not (output / "image_dimensions.png").exists()
