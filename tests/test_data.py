"""Exercise leakage protection with real, temporary image files."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, PngImagePlugin

from covid_xray.config import CLASSES
from covid_xray.data import (
    build_manifest,
    image_hash,
    load_manifest,
    make_dataset,
    validate_manifest,
)
from covid_xray.inference import image_to_array, validate_image


def _copy_pixels(path, target):
    # Change the PNG container while preserving the decoded image pixels.
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("fixture", "a different file with identical image pixels")
    with Image.open(path) as image:
        image.save(target, pnginfo=metadata, compress_level=0)
    return str(target)


def test_split_is_reproducible_and_all_classes_reach_each_split(image_index):
    first = build_manifest(image_index, seed=42)
    second = build_manifest(image_index, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert "split" not in image_index  # The caller's input remains unchanged.
    assert set(first["split"]) == {"train", "val", "test"}
    for _, group in first.groupby("split"):
        assert set(group["label"]) == set(CLASSES)
    assert first.groupby("patient_id")["split"].nunique().max() == 1
    assert first.groupby("sha256")["split"].nunique().max() == 1
    validate_manifest(first)


def test_hash_uses_decoded_pixels_not_png_metadata(image_index, tmp_path):
    path = Path(image_index.iloc[0]["path"])
    duplicate = Path(_copy_pixels(path, tmp_path / "metadata-copy.png"))

    assert path.read_bytes() != duplicate.read_bytes()
    assert image_hash(path) == image_hash(duplicate)


def test_duplicate_and_patient_connections_are_transitive(image_index, tmp_path):
    # A duplicate joins patient 0 to patient 1. A second duplicate joins 1 to 2.
    # Their other, different images must therefore stay in the same partition.
    extra = []
    for original_number, patient_number in [(0, 1), (1, 2)]:
        original = image_index.iloc[original_number]
        row = original.to_dict()
        row["path"] = _copy_pixels(
            original["path"], tmp_path / f"duplicate-{original_number}.png"
        )
        row["patient_id"] = image_index.iloc[patient_number]["patient_id"]
        extra.append(row)
    frame = build_manifest(pd.concat([image_index, pd.DataFrame(extra)], ignore_index=True))
    patients = set(image_index.iloc[:3]["patient_id"])

    assert frame.loc[frame["patient_id"].isin(patients), "split"].nunique() == 1


def test_same_pixels_with_conflicting_labels_are_rejected(image_index, tmp_path):
    row = image_index.iloc[0].to_dict()
    row["path"] = _copy_pixels(row["path"], tmp_path / "conflicting-label.png")
    row["label"] = CLASSES[1]
    row["patient_id"] = "conflicting-patient"

    with pytest.raises(ValueError):
        build_manifest(pd.concat([image_index, pd.DataFrame([row])], ignore_index=True))


@pytest.mark.parametrize("missing", [None, "", "   ", pd.NA])
def test_unknown_patient_fails_closed(image_index, missing):
    image_index.loc[0, "patient_id"] = missing
    with pytest.raises(ValueError):
        build_manifest(image_index)


@pytest.mark.parametrize("patient_id", ["unknown", "None", " NuLL ", "NaN", "-1"])
def test_placeholder_patient_identity_cannot_establish_isolation(image_index, patient_id):
    image_index.loc[0, "patient_id"] = patient_id
    with pytest.raises(ValueError, match="patient_id"):
        build_manifest(image_index)


@pytest.mark.parametrize("column", ["path", "label", "patient_id", "source"])
def test_required_index_metadata_is_not_invented(image_index, column):
    with pytest.raises(ValueError):
        build_manifest(image_index.drop(columns=column))


def test_missing_class_is_rejected(image_index):
    with pytest.raises(ValueError):
        build_manifest(image_index[image_index["label"] != CLASSES[0]])


def test_not_enough_patients_for_three_class_holdouts_is_rejected(image_index):
    tiny = image_index.groupby("label", sort=False).head(2)
    with pytest.raises(ValueError):
        build_manifest(tiny)


def test_cross_split_patient_leakage_is_rejected(manifest):
    first = manifest[manifest["split"] == "train"].index[0]
    second = manifest[manifest["split"] == "test"].index[0]
    manifest.loc[second, "patient_id"] = manifest.loc[first, "patient_id"]

    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_cross_split_duplicate_pixels_are_rejected(manifest, tmp_path):
    first = manifest[manifest["split"] == "train"].iloc[0]
    duplicate = first.to_dict()
    duplicate.update(
        path=_copy_pixels(first["path"], tmp_path / "holdout-duplicate.png"),
        patient_id="different-patient-same-pixels",
        split="test",
    )

    with pytest.raises(ValueError):
        validate_manifest(pd.concat([manifest, pd.DataFrame([duplicate])], ignore_index=True))


def test_missing_file_is_rejected(manifest):
    Path(manifest.iloc[0]["path"]).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_manifest(manifest)


def test_load_manifest_resolves_paths_relative_to_manifest(manifest, tmp_path):
    manifest["path"] = manifest["path"].map(lambda value: Path(value).name)
    path = tmp_path / "manifest.csv"
    manifest.to_csv(path, index=False)
    loaded = load_manifest(path)

    assert loaded["path"].map(lambda value: Path(value).is_absolute()).all()
    assert loaded["path"].map(lambda value: Path(value).is_file()).all()


def test_official_test_partition_is_preserved(image_index):
    image_index["original_split"] = "train"
    for _, group in image_index.groupby("label"):
        image_index.loc[group.index[:3], "original_split"] = "test"
    official_test_paths = set(
        image_index.loc[image_index["original_split"] == "test", "path"]
    )
    result = build_manifest(image_index)

    assert set(result.loc[result["split"] == "test", "path"]) == official_test_paths


def test_official_patient_overlap_requires_manual_resolution(image_index):
    image_index["original_split"] = "train"
    for _, group in image_index.groupby("label"):
        image_index.loc[group.index[:3], "original_split"] = "test"
    image_index.loc[3, "patient_id"] = image_index.loc[0, "patient_id"]

    with pytest.raises(ValueError):
        build_manifest(image_index)


def test_real_tensorflow_pipeline_has_rgb_224_inputs_and_stable_holdout(manifest):
    frame = manifest[manifest["split"] == "test"].reset_index(drop=True)
    first = list(make_dataset(frame, batch_size=4, training=False).as_numpy_iterator())
    second = list(make_dataset(frame, batch_size=4, training=False).as_numpy_iterator())
    first_images = np.concatenate([images for images, _ in first])
    second_images = np.concatenate([images for images, _ in second])
    labels = np.concatenate([labels for _, labels in first])

    assert first_images.shape == (len(frame), 224, 224, 3)
    assert first_images.dtype == np.float32
    assert first_images.min() >= 0 and first_images.max() <= 255
    assert first_images.max() > 1  # The model owns the normalization layer.
    np.testing.assert_array_equal(first_images, second_images)
    expected = np.array([CLASSES.index(label) for label in frame["label"]])
    np.testing.assert_array_equal(labels, expected)


@pytest.mark.parametrize("mode,format", [("L", "PNG"), ("RGB", "PNG"), ("RGBA", "PNG"), ("RGB", "JPEG")])
def test_training_decode_and_serving_preprocessing_match_exactly(tmp_path, mode, format):
    rng = np.random.default_rng(314)
    channels = {"L": (), "RGB": (3,), "RGBA": (4,)}[mode]
    pixels = rng.integers(0, 256, size=(41, 67, *channels), dtype=np.uint8)
    image = Image.fromarray(pixels)
    path = tmp_path / ("non-square.jpg" if format == "JPEG" else "non-square.png")
    if format == "JPEG":
        # Orientation metadata must also be handled identically in both paths.
        exif = image.getexif()
        exif[274] = 6
        image.save(path, exif=exif)
    else:
        image.save(path)
    frame = pd.DataFrame([{"path": str(path), "label": CLASSES[0]}])

    train_pixels, labels = next(
        make_dataset(frame, batch_size=1, training=False).as_numpy_iterator()
    )
    serving_pixels = image_to_array(validate_image(path.read_bytes()))

    assert train_pixels.shape == serving_pixels.shape == (1, 224, 224, 3)
    np.testing.assert_array_equal(train_pixels, serving_pixels)
    np.testing.assert_array_equal(labels, [0])
