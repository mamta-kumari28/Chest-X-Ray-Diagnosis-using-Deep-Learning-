"""Verify pinned labels and safe image-path resolution without dataset access."""

import importlib.util
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def importer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "import_covidx_metadata.py"
    spec = importlib.util.spec_from_file_location("covidx_metadata_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(name="example.png"):
    return {
        "path": name,
        "patient_id": "test-patient",
        "source": "unit-test-only",
        "label": "Normal",
        "original_split": "train",
    }


def test_importer_rejects_altered_official_label_files(importer, tmp_path):
    files = []
    for name in importer.LABEL_FILES:
        path = tmp_path / name
        path.write_text("test-patient example.png normal rsna\n")
        files.append(path)

    with pytest.raises(ValueError, match="checksum mismatch"):
        importer.read_labels(files)


def test_importer_rejects_ambiguous_duplicate_filenames(importer, tmp_path):
    for split in ("train", "test"):
        folder = tmp_path / split
        folder.mkdir()
        Image.new("L", (16, 20)).save(folder / "example.png")

    with pytest.raises(ValueError, match="found 2"):
        importer.resolve_image_paths([_row()], tmp_path)


def test_importer_rejects_image_symlink_outside_dataset(importer, tmp_path):
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    outside = tmp_path / "outside.png"
    Image.new("L", (16, 20)).save(outside)
    (data_dir / "example.png").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink leaves"):
        importer.resolve_image_paths([_row()], data_dir)


def test_importer_resolves_one_nested_file_without_changing_patient_metadata(importer, tmp_path):
    folder = tmp_path / "train"
    folder.mkdir()
    Image.new("L", (16, 20)).save(folder / "example.png")
    original = _row()
    result = importer.resolve_image_paths([original], tmp_path)

    assert result == [{**original, "path": "train/example.png"}]
    assert original["path"] == "example.png"


def test_phase2_mapping_retains_missing_records_for_quality_reporting(importer, tmp_path):
    found = tmp_path / "train" / "found.png"
    found.parent.mkdir()
    Image.new("L", (16, 20)).save(found)
    rows = [_row("found.png"), {**_row("missing.png"), "patient_id": "p2"}]

    result = importer.map_image_paths(rows, tmp_path)

    assert result[0]["image_path"] == "train/found.png"
    assert result[0]["resolution_status"] == "found"
    assert result[1]["image_path"] == "missing.png"
    assert result[1]["resolution_status"] == "missing"
    assert result[1]["candidate_count"] == 0
