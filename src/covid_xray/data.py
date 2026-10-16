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
from PIL import Image, ImageOps
from sklearn.model_selection import GroupShuffleSplit

from covid_xray.config import CLASSES, IMAGE_SIZE, SEED

REQUIRED = {"path", "label", "patient_id", "source", "sha256"}
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
    args = parser.parse_args()
    if args.command == "index":
        frame = index_images(args.data_dir, args.patients, args.output)
    else:
        path = Path(args.index).resolve()
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame["path"] = frame.path.map(lambda p: str((path.parent / p).resolve()))
        frame = build_manifest(frame, seed=args.seed, val_size=args.val_size)
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        frame["path"] = frame.path.map(lambda p: os.path.relpath(p, output.parent))
        frame.to_csv(output, index=False)
    print(json.dumps({"rows": len(frame), "output": args.output, "class_counts": frame.label.value_counts().to_dict()}, indent=2))


if __name__ == "__main__":
    main()
