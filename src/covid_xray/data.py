"""Metadata-driven indexing, patient-safe splits, and TensorFlow input pipelines.

No patient IDs are inferred from filenames. ``sha256`` is the hash of decoded
RGB pixels plus dimensions, so PNG metadata changes cannot hide exact duplicates.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.model_selection import GroupShuffleSplit

from covid_xray.config import CLASSES, IMAGE_SIZE, SEED

REQUIRED = {"path", "label", "patient_id", "source", "sha256"}
METADATA_REQUIRED = {"image_path", "label", "patient_id", "original_partition"}
EXTENSIONS = {".png", ".jpg", ".jpeg"}


def image_hash(path):
    """Hash orientation-corrected RGB pixels, rather than compressed file bytes."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        digest = hashlib.sha256(str(image.size).encode("ascii"))
        digest.update(image.tobytes())
        return digest.hexdigest()


def perceptual_hash(path):
    """64-bit perceptual hash flags many resized/recompressed duplicate scans.

    Matching hashes are conservatively grouped; they are not proof of identity.
    Different hashes do not establish that two images or patients are unrelated.
    """
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("L")
        pixels = np.asarray(image.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    # The low-frequency part of an orthonormal 2D DCT is a compact image signature.
    positions = np.arange(32) + 0.5
    basis = np.cos(np.pi / 32 * np.outer(np.arange(8), positions)) * np.sqrt(2 / 32)
    basis[0] /= np.sqrt(2)
    low = (basis @ pixels @ basis.T).ravel()
    bits = low > np.median(low[1:])
    bits[0] = False  # Ignore overall brightness (the DC coefficient).
    return np.packbits(bits).tobytes().hex()


def normalize_metadata(frame):
    """Return canonical Phase 2 metadata without inventing patient information."""
    metadata = frame.copy()
    if "image_path" not in metadata and "path" in metadata:
        metadata = metadata.rename(columns={"path": "image_path"})
    missing = METADATA_REQUIRED - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    if metadata.empty:
        raise ValueError("Dataset metadata is empty.")
    for column in METADATA_REQUIRED:
        values = metadata[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing or empty metadata value: {column}.")
        metadata[column] = values.astype(str).str.strip()
    placeholders = {"unknown", "none", "null", "nan", "-1"}
    if metadata.patient_id.str.lower().isin(placeholders).any():
        raise ValueError("Genuine patient IDs are required for leakage checks.")
    if not set(metadata.label).issubset(CLASSES):
        raise ValueError(f"Unknown labels; use exactly {CLASSES}.")
    if not set(metadata.original_partition).issubset({"train", "test"}):
        raise ValueError("original_partition must contain only train or test.")
    return metadata.reset_index(drop=True)


def _inspect_image(path):
    """Decode one image once and return dimensions plus duplicate signatures."""
    with Image.open(path) as image:
        image.load()
        image_format = image.format or "unknown"
        image_mode = image.mode
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        rgb = image.convert("RGB")
        digest = hashlib.sha256(str(rgb.size).encode("ascii"))
        digest.update(rgb.tobytes())
        gray = rgb.convert("L")
        pixels = np.asarray(
            gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64
        )
        positions = np.arange(32) + 0.5
        basis = np.cos(np.pi / 32 * np.outer(np.arange(8), positions)) * np.sqrt(2 / 32)
        basis[0] /= np.sqrt(2)
        low = (basis @ pixels @ basis.T).ravel()
        bits = low > np.median(low[1:])
        bits[0] = False
    return {
        "width": width,
        "height": height,
        "image_mode": image_mode,
        "image_format": image_format,
        "sha256": digest.hexdigest(),
        "perceptual_hash": np.packbits(bits).tobytes().hex(),
    }


def _count_duplicate_groups(frame, column):
    values = frame.loc[frame[column].astype(str).ne(""), column]
    counts = values.value_counts()
    repeated = counts[counts.gt(1)]
    return int(len(repeated)), int(repeated.sum())


def audit_metadata(frame, data_dir=None):
    """Inspect metadata and locally accessible images without creating data splits.

    The returned table contains one row for every upstream metadata record, even
    when its image is missing, unsupported, or corrupt. Counts in the report keep
    upstream records separate from files that were actually decoded locally.
    """
    metadata = normalize_metadata(frame)
    root = Path(data_dir).expanduser().resolve() if data_dir else None
    records = []
    for row in metadata.to_dict("records"):
        source_path = Path(row["image_path"]).expanduser()
        resolved = source_path.resolve() if source_path.is_absolute() else (
            (root / source_path).resolve() if root else source_path.resolve()
        )
        result = {
            **row,
            "resolved_path": str(resolved),
            "file_exists": False,
            "supported_format": source_path.suffix.lower() in EXTENSIONS,
            "readable": False,
            "width": pd.NA,
            "height": pd.NA,
            "image_mode": "",
            "image_format": "",
            "sha256": "",
            "perceptual_hash": "",
            "validation_status": "missing",
            "validation_message": "Image file does not exist.",
        }
        if row.get("resolution_status") == "ambiguous":
            result.update(
                validation_status="ambiguous",
                validation_message=(
                    "Multiple local files have this upstream filename; "
                    "no image was selected."
                ),
            )
        elif root and not resolved.is_relative_to(root):
            result.update(
                validation_status="invalid_path",
                validation_message="Relative image path leaves the configured data directory.",
            )
        elif not resolved.is_file():
            pass
        elif not result["supported_format"]:
            result.update(
                file_exists=True,
                validation_status="unsupported",
                validation_message=f"Unsupported extension: {resolved.suffix or '(none)'}.",
            )
        else:
            result["file_exists"] = True
            try:
                result.update(_inspect_image(resolved))
                result.update(
                    readable=True,
                    validation_status="valid",
                    validation_message="",
                )
            except (OSError, UnidentifiedImageError, ValueError, SyntaxError) as error:
                result.update(
                    validation_status="corrupt",
                    validation_message=f"{type(error).__name__}: image could not be decoded.",
                )
        records.append(result)

    audited = pd.DataFrame(records)
    exact_groups, exact_records = _count_duplicate_groups(audited, "sha256")
    perceptual_groups, perceptual_records = _count_duplicate_groups(
        audited, "perceptual_hash"
    )
    patient_partitions = audited.groupby("patient_id").original_partition.nunique()
    overlapping_patients = sorted(patient_partitions[patient_partitions.gt(1)].index.tolist())
    path_label_counts = audited.groupby("image_path").label.nunique()
    content_label_counts = (
        audited[audited.sha256.ne("")].groupby("sha256").label.nunique()
    )
    status_counts = audited.validation_status.value_counts()
    accessible = audited[audited.file_exists]
    valid = audited[audited.readable]

    def class_counts(mask):
        return {
            label: int((mask & audited.label.eq(label)).sum()) for label in CLASSES
        }

    warnings = []
    if int(status_counts.get("missing", 0)):
        warnings.append("Some metadata records do not have an accessible local image.")
    if int(status_counts.get("corrupt", 0)):
        warnings.append("Some image files could not be decoded by Pillow.")
    if int(status_counts.get("unsupported", 0)):
        warnings.append("Some files use unsupported image extensions.")
    if int(status_counts.get("ambiguous", 0)):
        warnings.append("Some upstream filenames match multiple local image files.")
    if int(status_counts.get("invalid_path", 0)):
        warnings.append("Some image paths leave the configured data directory.")
    if overlapping_patients:
        warnings.append("Patient IDs overlap the official train/test boundary.")
    if int(path_label_counts.gt(1).sum()) or int(content_label_counts.gt(1).sum()):
        warnings.append("Conflicting labels were found for an image path or exact image content.")
    if perceptual_groups:
        warnings.append(
            "Matching perceptual hashes require review; they are duplicate candidates, not proof."
        )
    if exact_groups:
        warnings.append("Exact decoded-image duplicate groups require leakage review.")
    report = {
        "metadata_records": int(len(audited)),
        "unique_metadata_records": int(
            len(audited.drop_duplicates(list(METADATA_REQUIRED)))
        ),
        "records_with_accessible_images": int(audited.file_exists.sum()),
        "accessible_image_files": int(accessible.resolved_path.nunique()),
        "valid_image_records": int(audited.readable.sum()),
        "valid_images": int(valid.resolved_path.nunique()),
        "missing_files": int(status_counts.get("missing", 0)),
        "invalid_paths": int(status_counts.get("invalid_path", 0)),
        "ambiguous_records": int(status_counts.get("ambiguous", 0)),
        "unsupported_files": int(status_counts.get("unsupported", 0)),
        "corrupt_files": int(status_counts.get("corrupt", 0)),
        "duplicate_metadata_records": int(
            audited.duplicated(list(METADATA_REQUIRED), keep=False).sum()
        ),
        "duplicate_image_path_records": int(
            audited.duplicated("image_path", keep=False).sum()
        ),
        "exact_duplicate_groups": exact_groups,
        "exact_duplicate_records": exact_records,
        "perceptual_match_groups": perceptual_groups,
        "perceptual_match_records": perceptual_records,
        "patient_overlap_count": len(overlapping_patients),
        "patient_overlap_ids_sample": overlapping_patients[:20],
        "image_path_label_conflicts": int(path_label_counts.gt(1).sum()),
        "exact_content_label_conflicts": int(content_label_counts.gt(1).sum()),
        "class_metadata_records": class_counts(pd.Series(True, index=audited.index)),
        "class_accessible_files": class_counts(audited.file_exists),
        "class_valid_images": class_counts(audited.readable),
        "partition_metadata_records": {
            key: int(value)
            for key, value in audited.original_partition.value_counts().to_dict().items()
        },
        "warnings": warnings,
    }
    return audited, report


def audit_metadata_file(metadata_path, data_dir, output, report_path):
    """Audit one metadata CSV and persist portable local observations."""
    metadata_path = Path(metadata_path).expanduser().resolve()
    frame = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    audited, report = audit_metadata(frame, data_dir)
    output = Path(output).expanduser().resolve()
    report_path = Path(report_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    audited.to_csv(output, index=False)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    return audited, report


def _check_rows(frame, check_files=True):
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("The image manifest is empty.")
    for column in REQUIRED:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing or empty {column}; genuine patient metadata is required.")
    if frame.patient_id.astype(str).str.strip().str.lower().isin({"unknown", "none", "null", "nan", "-1"}).any():
        raise ValueError("Missing patient_id: placeholder identities cannot establish patient isolation.")
    if not frame.patient_id.map(lambda value: isinstance(value, str) and value == value.strip()).all():
        raise ValueError("patient_id must be consistent strings without surrounding whitespace.")
    if not set(frame.label).issubset(CLASSES):
        raise ValueError(f"Unknown labels; use exactly {CLASSES}.")
    if frame.path.duplicated().any():
        raise ValueError("Duplicate image paths in manifest.")
    if not frame.sha256.astype(str).str.fullmatch(r"[a-f0-9]{64}").all():
        raise ValueError("sha256 must contain canonical decoded-image SHA-256 hashes.")
    if frame.groupby("sha256").label.nunique().gt(1).any():
        raise ValueError("Duplicate images have conflicting labels; resolve source annotations.")
    if "perceptual_hash" in frame and not frame.perceptual_hash.astype(str).str.fullmatch(r"[a-f0-9]{16}").all():
        raise ValueError("perceptual_hash must contain 64-bit image hashes.")
    if check_files:
        for path in frame.path:
            if not Path(path).is_file():
                raise ValueError(f"Image file not found: {path}")
            if Path(path).suffix.lower() not in EXTENSIONS:
                raise ValueError(f"Unsupported image type: {path}")


def validate_manifest(frame, check_files=True):
    """Reject unknown splits, missing classes, patient leaks, and pixel duplicates."""
    _check_rows(frame, check_files)
    if "split" not in frame or frame.split.isna().any():
        raise ValueError("Manifest needs explicit train, val, and test splits.")
    if set(frame.split) != {"train", "val", "test"}:
        raise ValueError("Manifest must contain exactly train, val, and test splits.")
    if "original_split" in frame:
        if not set(frame.original_split).issubset({"train", "test"}):
            raise ValueError("Invalid original_split assignments.")
        if not (frame.original_split.eq("test") == frame.split.eq("test")).all():
            raise ValueError("The official test boundary was changed; restore its original assignments.")
    columns = ["patient_id", "sha256"]
    if "perceptual_hash" in frame:
        columns.append("perceptual_hash")
    for column in columns:
        if frame.groupby(column).split.nunique().gt(1).any():
            raise ValueError(f"Leakage: {column} occurs across splits.")
    for name, part in frame.groupby("split"):
        if set(part.label) != set(CLASSES):
            raise ValueError(f"Split {name} does not contain all three classes.")


def _groups(frame):
    # Union-find connects repeated patients AND duplicates with different IDs.
    # Connected groups stay together, including transitive duplicate links.
    parent = list(range(len(frame)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    columns = ["patient_id", "sha256"]
    if "perceptual_hash" in frame:
        columns.append("perceptual_hash")
    for column in columns:
        first = {}
        for i, value in enumerate(frame[column]):
            if value in first:
                parent[find(i)] = find(first[value])
            else:
                first[value] = i
    return np.array([find(i) for i in range(len(frame))])


def _holdout(frame, fraction, seed):
    """Choose a reproducible group split close to image/class target proportions."""
    if len(set(frame._group)) < 3:
        raise ValueError("Too few independent patient/duplicate groups to split safely.")
    best = None
    best_score = float("inf")
    splitter = GroupShuffleSplit(n_splits=128, test_size=fraction, random_state=seed)
    for left, right in splitter.split(frame, frame.label, groups=frame._group):
        train, held = frame.iloc[left], frame.iloc[right]
        if set(train.label) != set(CLASSES) or set(held.label) != set(CLASSES):
            continue
        # Large connected patient groups mean exact ratios are not always possible.
        score = abs(len(held) / len(frame) - fraction)
        for label in CLASSES:
            score += abs(held.label.eq(label).sum() / frame.label.eq(label).sum() - fraction)
        if score < best_score:
            best, best_score = (train.copy(), held.copy()), score
    if best is None:
        raise ValueError("Cannot retain all classes with disjoint patients; add independent patients.")
    return best


def build_manifest(index, seed=SEED, test_size=0.15, val_size=0.15):
    """Preserve an official test set when present; otherwise group-split 70/15/15."""
    if not (0 < test_size < 1 and 0 < val_size < 1 and test_size + val_size < 1):
        raise ValueError("Split fractions must be positive and sum to less than one.")
    frame = index.copy().reset_index(drop=True)
    missing = (REQUIRED - {"sha256"}) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    if "sha256" not in frame:
        frame["sha256"] = frame.path.map(image_hash)
    if "perceptual_hash" not in frame:
        frame["perceptual_hash"] = frame.path.map(perceptual_hash)
    _check_rows(frame)
    if set(frame.label) != set(CLASSES):
        raise ValueError("All three classes are required before splitting.")
    frame["_group"] = _groups(frame)
    if "original_split" in frame:
        if frame.original_split.isna().any() or not set(frame.original_split).issubset({"train", "test"}):
            raise ValueError("original_split must contain only official train/test assignments.")
        if set(frame.original_split) != {"train", "test"}:
            raise ValueError("Official metadata must include both training and testing partitions.")
        if frame.groupby("_group").original_split.nunique().gt(1).any():
            raise ValueError("Official split leaks patients or duplicate images; resolve before training.")
        rest = frame[frame.original_split.eq("train")].copy()
        test = frame[frame.original_split.eq("test")].copy()
        # In this mode val_size is the fraction of the official TRAIN pool.
        train, val = _holdout(rest, val_size, seed)
    else:
        rest, test = _holdout(frame, test_size, seed)
        train, val = _holdout(rest, val_size / (1 - test_size), seed + 1)
    result = pd.concat([part.assign(split=name) for part, name in
                        [(train, "train"), (val, "val"), (test, "test")]], ignore_index=True)
    result = result.drop(columns="_group")
    validate_manifest(result)
    return result


def load_manifest(path):
    """Resolve portable CSV paths and recompute hashes to detect changed image data."""
    path = Path(path).resolve()
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "path" not in frame:
        raise ValueError("Manifest is missing path column.")
    frame["path"] = frame.path.map(lambda p: str((path.parent / p).resolve()))
    validate_manifest(frame)
    for row in frame.itertuples():
        if image_hash(row.path) != row.sha256:
            raise ValueError(f"Image content changed since indexing: {row.path}")
        if "perceptual_hash" in frame and perceptual_hash(row.path) != row.perceptual_hash:
            raise ValueError(f"Perceptual image signature changed: {row.path}")
    return frame


def make_dataset(frame, batch_size=32, training=False, seed=SEED):
    """Decode/resize with parallel I/O; augmentation and scaling live in the model."""
    import tensorflow as tf

    if batch_size < 1 or frame.empty:
        raise ValueError("Use a positive batch size and a nonempty image partition.")
    labels = frame.label.map({name: i for i, name in enumerate(CLASSES)})
    if labels.isna().any():
        raise ValueError("Unknown image label.")
    dataset = tf.data.Dataset.from_tensor_slices((frame.path.astype(str).tolist(), labels.to_numpy(np.int32)))
    if training:
        dataset = dataset.shuffle(min(len(frame), 4096), seed=seed, reshuffle_each_iteration=True)

    def decode(path, label):
        # PIL aligns EXIF handling with duplicate hashes and the upload pipeline.
        def read(value):
            with Image.open(value.numpy().decode()) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                # Use the exact same bilinear resize as web inference.
                image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
                return np.asarray(image, dtype=np.uint8)

        image = tf.py_function(read, [path], Tout=tf.uint8)
        image.set_shape([*IMAGE_SIZE, 3])
        return tf.cast(image, tf.float32), label

    options = tf.data.Options()
    options.experimental_deterministic = True
    # Bound workers for resource-constrained Streamlit/CI machines; no full-image cache.
    options.threading.private_threadpool_size = min(os.cpu_count() or 2, 4)
    dataset = dataset.with_options(options)
    return dataset.map(decode, num_parallel_calls=4).batch(batch_size).prefetch(2)


def index_images(data_dir, patients, output):
    """Create an index only from an explicit path/patient/source/label mapping."""
    data_dir, output = Path(data_dir).resolve(), Path(output).resolve()
    meta = pd.read_csv(patients, dtype=str, keep_default_na=False)
    if "path" not in meta and "image_path" in meta:
        meta = meta.rename(columns={"image_path": "path"})
    required = {"path", "label", "patient_id", "source"}
    if not required.issubset(meta):
        raise ValueError(f"Patient CSV requires {sorted(required)}.")
    if meta.empty or meta.path.duplicated().any():
        raise ValueError("Patient CSV is empty or has duplicate paths.")
    rows = []
    for row in meta.to_dict("records"):
        image = (data_dir / row["path"]).resolve()
        if not image.is_relative_to(data_dir):
            raise ValueError("Image paths must remain within --data-dir.")
        row["path"] = str(image)
        row["sha256"] = image_hash(image)
        row["perceptual_hash"] = perceptual_hash(image)
        rows.append(row)
    frame = pd.DataFrame(rows)
    _check_rows(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame["path"] = frame.path.map(lambda p: os.path.relpath(p, output.parent))
    frame.to_csv(output, index=False)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="Index genuine patient mappings and hash images")
    index.add_argument("--data-dir", required=True)
    index.add_argument("--patients", required=True)
    index.add_argument("--output", default="data/processed/index.csv")
    split = commands.add_parser("split", help="Create leakage-checked train/val/test manifest")
    split.add_argument("--index", required=True)
    split.add_argument("--output", default="data/processed/manifest.csv")
    split.add_argument("--seed", type=int, default=SEED)
    split.add_argument("--val-size", type=float, default=0.15)
    audit = commands.add_parser(
        "audit", help="Validate metadata and accessible images without creating splits"
    )
    audit.add_argument("--metadata", required=True)
    audit.add_argument("--data-dir", required=True)
    audit.add_argument("--output", default="data/processed/validated_metadata.csv")
    audit.add_argument(
        "--report", default="artifacts/results/eda/data_quality_report.json"
    )
    args = parser.parse_args()
    if args.command == "index":
        frame = index_images(args.data_dir, args.patients, args.output)
        result = {
            "rows": len(frame),
            "output": args.output,
            "class_counts": frame.label.value_counts().to_dict(),
        }
    elif args.command == "split":
        path = Path(args.index).resolve()
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame["path"] = frame.path.map(lambda p: str((path.parent / p).resolve()))
        frame = build_manifest(frame, seed=args.seed, val_size=args.val_size)
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        frame["path"] = frame.path.map(lambda p: os.path.relpath(p, output.parent))
        frame.to_csv(output, index=False)
        result = {
            "rows": len(frame),
            "output": args.output,
            "class_counts": frame.label.value_counts().to_dict(),
        }
    else:
        _, report = audit_metadata_file(
            args.metadata, args.data_dir, args.output, args.report
        )
        result = {**report, "output": args.output, "report": args.report}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
