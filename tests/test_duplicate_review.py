"""Duplicate review stays private, explicit, and fail-closed at test boundaries."""

import pandas as pd
import pytest
from PIL import Image

from covid_xray.data import audit_metadata, build_manifest, image_hash, perceptual_hash
from covid_xray.review import (
    build_duplicate_review,
    create_contact_sheets,
    enforce_official_duplicate_boundary,
    summarize_duplicate_review,
    write_duplicate_review,
)


def _official_index(image_index):
    frame = image_index.copy()
    frame["sha256"] = frame.path.map(image_hash)
    frame["perceptual_hash"] = frame.path.map(perceptual_hash)
    frame["original_split"] = "train"
    for _, group in frame.groupby("label"):
        frame.loc[group.index[:3], "original_split"] = "test"
    return frame


def test_exact_review_identifies_cross_partition_conflicts(image_index):
    frame = _official_index(image_index)
    train_index = frame[frame.original_split.eq("train")].index[0]
    test_index = frame[frame.original_split.eq("test")].index[0]
    frame.loc[train_index, "sha256"] = frame.loc[test_index, "sha256"]
    frame.loc[train_index, "label"] = frame.loc[test_index, "label"]

    review = build_duplicate_review(frame)
    exact = review[review.evidence_type.eq("exact")]

    assert exact.crosses_official_partition.all()
    assert exact.differing_patient_ids.all()
    with pytest.raises(ValueError, match="Exact duplicate"):
        enforce_official_duplicate_boundary(review)


def test_unresolved_cross_partition_perceptual_candidate_blocks_split(image_index):
    frame = _official_index(image_index)
    train_index = frame[frame.original_split.eq("train")].index[0]
    test_index = frame[frame.original_split.eq("test")].index[0]
    frame.loc[train_index, "perceptual_hash"] = frame.loc[
        test_index, "perceptual_hash"
    ]

    with pytest.raises(ValueError, match="Unresolved perceptual"):
        build_manifest(frame)


def test_rejected_cross_partition_candidate_does_not_link_images(image_index):
    frame = _official_index(image_index)
    train_index = frame[frame.original_split.eq("train")].index[0]
    test_index = frame[frame.original_split.eq("test")].index[0]
    value = frame.loc[test_index, "perceptual_hash"]
    frame.loc[train_index, "perceptual_hash"] = value
    decisions = pd.DataFrame(
        [{"group_id": f"perceptual:{value}", "review_status": "rejected"}]
    )

    result = build_manifest(frame, review=decisions)

    assert set(result[result.original_split.eq("test")].split) == {"test"}


def test_accepted_cross_partition_candidate_blocks_split(image_index):
    frame = _official_index(image_index)
    train_index = frame[frame.original_split.eq("train")].index[0]
    test_index = frame[frame.original_split.eq("test")].index[0]
    value = frame.loc[test_index, "perceptual_hash"]
    frame.loc[train_index, "perceptual_hash"] = value
    decisions = pd.DataFrame(
        [{"group_id": f"perceptual:{value}", "review_status": "accepted"}]
    )

    with pytest.raises(ValueError, match="Human-accepted duplicate"):
        build_manifest(frame, review=decisions)


def test_conflicting_human_decisions_are_rejected(image_index):
    frame = _official_index(image_index)
    value = frame.loc[0, "perceptual_hash"]
    frame.loc[1, "perceptual_hash"] = value
    decisions = pd.DataFrame(
        [
            {"group_id": f"perceptual:{value}", "review_status": "accepted"},
            {"group_id": f"perceptual:{value}", "review_status": "rejected"},
        ]
    )

    with pytest.raises(ValueError, match="conflicting review decisions"):
        build_duplicate_review(frame, decisions)


def test_review_reports_label_and_patient_disagreement(image_index):
    frame = _official_index(image_index)
    frame.loc[1, "sha256"] = frame.loc[0, "sha256"]
    frame.loc[1, "label"] = next(
        label for label in frame.label.unique() if label != frame.loc[0, "label"]
    )

    review = build_duplicate_review(frame)
    exact = review[review.evidence_type.eq("exact")]

    assert exact.conflicting_labels.all()
    assert exact.differing_patient_ids.all()


def test_contact_sheet_uses_only_existing_authorized_local_images(
    image_index, tmp_path
):
    frame = _official_index(image_index)
    frame.loc[1, "perceptual_hash"] = frame.loc[0, "perceptual_hash"]
    review = build_duplicate_review(frame)

    written = create_contact_sheets(review, tmp_path / "private", max_groups=1)

    assert written == 1
    output = next((tmp_path / "private").glob("*.jpg"))
    with Image.open(output) as image:
        image.verify()


def test_missing_dataset_is_reported_without_inventing_validation(tmp_path):
    metadata = pd.DataFrame(
        [
            {
                "image_path": "not-downloaded.png",
                "label": "Normal",
                "patient_id": "patient-1",
                "original_partition": "train",
            }
        ]
    )

    audited, report = audit_metadata(metadata, tmp_path / "unavailable")

    assert report["metadata_records"] == 1
    assert report["valid_images"] == 0
    assert report["missing_files"] == 1
    assert audited.loc[0, "validation_status"] == "missing"


def test_review_summary_contains_no_private_identifiers(image_index):
    frame = _official_index(image_index)
    frame.loc[1, "perceptual_hash"] = frame.loc[0, "perceptual_hash"]

    summary = summarize_duplicate_review(build_duplicate_review(frame))

    assert summary["perceptual_candidate_groups"] >= 1
    assert "patient_id" not in summary
    assert "image_path" not in summary


def test_private_review_command_writes_csv_and_safe_summary(image_index, tmp_path):
    frame = _official_index(image_index)
    frame.loc[1, "perceptual_hash"] = frame.loc[0, "perceptual_hash"]
    validated = tmp_path / "validated.csv"
    output = tmp_path / "duplicate_review.csv"
    summary_path = tmp_path / "summary.json"
    frame.rename(
        columns={"path": "image_path", "original_split": "original_partition"}
    ).to_csv(validated, index=False)

    review, summary = write_duplicate_review(
        validated, output, summary_path=summary_path
    )

    assert not review.empty
    assert output.is_file()
    assert summary_path.is_file()
    assert summary["perceptual_candidate_groups"] >= 1
