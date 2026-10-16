"""Convert verified COVIDx CXR-3A patient labels into the project metadata CSV.

The official labels supply patient IDs; image names never stand in for patients.
Download the exact label files documented in docs/DATASET.md before running this
script. No dataset image or private metadata is committed by this helper.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


# Pin the inspected three-class metadata, rather than accepting a later binary
# release under a similar name. These are file integrity checks, not image hashes.
LABEL_FILES = {
    "train_COVIDx_CXR-3A.txt": (
        "train",
        "a7e7ef10b00df1bdeb05c5028b04fbe99fe7011593382e2adff8a2550705835a",
    ),
    "test_COVIDx_CXR-3A.txt": (
        "test",
        "048629233e9426f1b42ce1d86e898bdb1691ff8544d81fa44b1d6a3657d88871",
    ),
}
LABEL_MAP = {"COVID-19": "COVID-19", "normal": "Normal", "pneumonia": "Pneumonia"}
SOURCE_NAMES = {"actmed", "bimcv", "cohen", "fig1", "ricord", "rsna", "sirm", "stonybrook"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def read_labels(label_files: list[Path]) -> list[dict[str, str]]:
    """Validate both pinned label files and retain the genuine upstream IDs."""
    names = [path.name for path in label_files]
    if len(names) != 2 or set(names) != set(LABEL_FILES):
        raise ValueError("Supply exactly the official train_COVIDx_CXR-3A.txt and test_COVIDx_CXR-3A.txt files.")

    rows: list[dict[str, str]] = []
    image_names: set[str] = set()
    patient_sources: dict[str, str] = {}
    patient_splits: dict[str, str] = {}
    for file in label_files:
        data = file.read_bytes()
        split, expected_hash = LABEL_FILES[file.name]
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError(f"Label checksum mismatch: {file.name}. Download the pinned file in docs/DATASET.md.")
        for line_number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f"Expected four columns in {file.name}:{line_number}.")
            patient_id, name, label, source = fields
            if label not in LABEL_MAP or source not in SOURCE_NAMES:
                raise ValueError(f"Unexpected label or source in {file.name}:{line_number}.")
            if patient_id.casefold() in {"", "none", "nan", "null", "unknown", "-1"}:
                raise ValueError(f"Missing patient ID in {file.name}:{line_number}.")
            if Path(name).name != name or "\\" in name:
                raise ValueError(f"The pinned labels must contain plain image filenames: {name}.")
            if name in image_names:
                raise ValueError(f"Image filename appears more than once in the label files: {name}.")
            # The inspected release contains no raw ID reused across sources. A
            # changed release must not silently merge or separate such identities.
            if patient_id in patient_sources and patient_sources[patient_id] != source:
                raise ValueError(f"Patient ID {patient_id} collides across sources; reconcile provenance first.")
            if patient_id in patient_splits and patient_splits[patient_id] != split:
                raise ValueError(f"Patient ID {patient_id} crosses the official train/test boundary.")
            patient_sources[patient_id] = source
            patient_splits[patient_id] = split
            image_names.add(name)
            rows.append({
                "path": name,
                "patient_id": patient_id,
                "source": source,
                "label": LABEL_MAP[label],
                "original_split": split,
            })
    return rows


def resolve_image_paths(rows: list[dict[str, str]], data_dir: Path) -> list[dict[str, str]]:
    """Resolve each official filename to one local image, rejecting ambiguity."""
    root = data_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Not an image directory: {data_dir}.")
    files: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"Image symlink leaves the dataset directory: {path.name}.")
            files.setdefault(path.name, []).append(path)

    result: list[dict[str, str]] = []
    for row in rows:
        matches = files.get(row["path"], [])
        if len(matches) != 1:
            raise ValueError(
                f"Expected one file named {row['path']}, found {len(matches)}. "
                "Use the version-7 archive with one extracted copy of each image."
            )
        result.append({**row, "path": matches[0].relative_to(root).as_posix()})
    return result


def main() -> None:
    """Write a manifest only after every image and upstream ID passes checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", nargs="+", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError(f"Output already exists: {args.output}. Choose a new path to preserve existing metadata.")
        rows = resolve_image_paths(read_labels(args.labels), args.data_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation avoids overwriting a concurrently created manifest.
        with args.output.open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["path", "patient_id", "source", "label", "original_split"])
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.exit(2, f"Metadata import failed: {exc}\n")
    print(f"Wrote {len(rows):,} image records to {args.output}.")
    print("Patient IDs are upstream identities; verify source provenance and licenses before training.")


if __name__ == "__main__":
    main()
