"""Build the four lightweight Kaggle entry notebooks shipped with the project."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "kaggle_notebooks"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    for position, cell in enumerate(cells):
        source = "".join(cell.get("source", []))
        cell["id"] = hashlib.sha256(f"{position}:{source}".encode()).hexdigest()[:12]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


BOOTSTRAP = r'''from pathlib import Path
import os
import shutil
import subprocess
import sys
import zipfile

REPO_URL = "https://github.com/mamta-kumari28/Chest-X-Ray-Diagnosis-using-Deep-Learning-.git"
REPO_REF = "phase2/final-validation-fixes"  # Change to main after the PR is merged.
ROOT = Path("/kaggle/working/covid-xray-project")
OUTPUT_ROOT = Path("/kaggle/working/covid_xray_outputs")

if not ROOT.exists():
    source_archives = sorted(Path("/kaggle/input").glob("**/covid_xray_project_source.zip"))
    if source_archives:
        with zipfile.ZipFile(source_archives[0]) as archive:
            archive.extractall("/kaggle/working")
        print("Loaded project source from:", source_archives[0])
    else:
        extracted_sources = [
            path.parent for path in Path("/kaggle/input").glob("**/pyproject.toml")
            if (path.parent / "src" / "covid_xray").is_dir()
        ]
        if extracted_sources:
            shutil.copytree(extracted_sources[0], ROOT)
            print("Loaded extracted project source from:", extracted_sources[0])
        else:
            subprocess.run(["git", "clone", "--branch", REPO_REF, "--depth", "1", REPO_URL, str(ROOT)], check=True)
os.chdir(ROOT)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], check=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print("Repository:", ROOT)
print("Outputs:", OUTPUT_ROOT)'''


FIND_DATA = r'''# The attached Kaggle dataset must contain train/ and test/ image folders.
def find_data_root():
    candidates = [Path("/kaggle/input/covidx-cxr2")]
    candidates += sorted(Path("/kaggle/input").glob("**/train"))
    for item in candidates:
        root = item.parent if item.name == "train" else item
        if (root / "train").is_dir() and (root / "test").is_dir():
            return root
    raise FileNotFoundError("Attach Kaggle dataset andyczhao/covidx-cxr2 version 7 first.")

DATA_DIR = find_data_root()
print("Dataset:", DATA_DIR)'''


FIND_FILE = r'''def find_one(name, roots=(Path("/kaggle/working"), Path("/kaggle/input"))):
    matches = []
    for root in roots:
        matches.extend(root.glob(f"**/{name}"))
    matches = [path for path in matches if ".git" not in path.parts]
    if not matches:
        raise FileNotFoundError(f"Could not find {name}. Attach the previous notebook output as a Kaggle input.")
    print(name, "->", matches[0])
    return matches[0]
'''


def write(name: str, cells: list[dict]) -> None:
    path = OUTPUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook(cells), indent=1) + "\n", encoding="utf-8")


write(
    "02_preprocessing_kaggle.ipynb",
    [
        markdown("""# 02 · Kaggle metadata preprocessing

Run this notebook **first**. Attach `andyczhao/covidx-cxr2`, select version 7, and enable Internet so the two small pinned official CXR-3A label files can be downloaded. This notebook does not train a model or alter the official test partition."""),
        code(BOOTSTRAP),
        code(FIND_DATA),
        code(r'''from urllib.request import urlretrieve

LABEL_DIR = OUTPUT_ROOT / "labels_cxr3a"
PROCESSED = OUTPUT_ROOT / "processed"
LABEL_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

base = "https://raw.githubusercontent.com/lindawangg/COVID-Net/9645505646161c7eae5fc1a8ecbfa34a77551f2f/labels"
for name in ("train_COVIDx_CXR-3A.txt", "test_COVIDx_CXR-3A.txt"):
    target = LABEL_DIR / name
    if not target.exists():
        urlretrieve(f"{base}/{name}", target)
    print(name, target.stat().st_size, "bytes")'''),
        code(r'''METADATA = PROCESSED / "patients.csv"
if METADATA.exists():
    METADATA.unlink()  # Kaggle working output from this run only.
subprocess.run([
    sys.executable, "scripts/import_covidx_metadata.py",
    "--labels", str(LABEL_DIR / "train_COVIDx_CXR-3A.txt"), str(LABEL_DIR / "test_COVIDx_CXR-3A.txt"),
    "--data-dir", str(DATA_DIR), "--output", str(METADATA), "--require-all-images",
], check=True)

import pandas as pd
metadata = pd.read_csv(METADATA)
display(metadata.head())
display(pd.crosstab(metadata["original_partition"], metadata["label"], margins=True))
print("Saved:", METADATA)'''),
        markdown("Use **Save Version → Save & Run All**. For notebook 01, either keep the same session or attach this notebook's saved output as an input."),
    ],
)

write(
    "01_data_exploration_kaggle.ipynb",
    [
        markdown("""# 01 · Kaggle validation and EDA

Run notebook 02 first. This performs the real image audit and creates plots plus the private duplicate-review files. It can take considerable time for 30,034 images."""),
        code(BOOTSTRAP),
        code(FIND_DATA),
        code(FIND_FILE + '\nMETADATA = find_one("patients.csv")'),
        code(r'''from covid_xray.data import audit_metadata_file
from covid_xray.eda import create_dataset_report
from covid_xray.review import write_duplicate_review
import json
import pandas as pd

PROCESSED = OUTPUT_ROOT / "processed"
EDA = OUTPUT_ROOT / "eda"
PRIVATE = EDA / "private"
PROCESSED.mkdir(parents=True, exist_ok=True)
PRIVATE.mkdir(parents=True, exist_ok=True)

VALIDATED = PROCESSED / "validated_metadata.csv"
REPORT = EDA / "data_quality_report.json"
audited, report = audit_metadata_file(METADATA, DATA_DIR, VALIDATED, REPORT)
create_dataset_report(audited, report, EDA)
write_duplicate_review(
    VALIDATED,
    PRIVATE / "duplicate_review.csv",
    summary_path=EDA / "duplicate_review_summary.json",
    contact_sheets=PRIVATE / "contact_sheets",
    data_dir=DATA_DIR,
)
print(json.dumps(report, indent=2))'''),
        code(r'''from IPython.display import Image as DisplayImage, display

for name in ("class_distribution.png", "sample_images.png", "image_dimensions.png"):
    path = EDA / name
    if path.exists():
        display(DisplayImage(filename=str(path)))
print("Private review:", PRIVATE / "duplicate_review.csv")
print("Save the full covid_xray_outputs folder as this notebook's output.")'''),
        markdown("""## Required human review before training

Download `duplicate_review.csv`. For every `perceptual:*` group, inspect the contact sheet and set one consistent `review_status` (`accepted` or `rejected`) plus a note. Keep patient IDs and paths private. Upload the completed file to a **private Kaggle dataset** named `duplicate_decisions.csv`. Do not mark candidates automatically."""),
    ],
)

write(
    "03_cnn_training_kaggle.ipynb",
    [
        markdown("""# 03 · Kaggle CNN training

Enable a GPU accelerator. Attach the saved output from notebook 01 and your private reviewed `duplicate_decisions.csv`. The notebook stops if review is incomplete or an official-boundary conflict exists."""),
        code(BOOTSTRAP),
        code(FIND_FILE + '\nVALIDATED = find_one("validated_metadata.csv")\nDECISIONS = find_one("duplicate_decisions.csv")'),
        code(r'''import pandas as pd
from covid_xray.review import build_duplicate_review, enforce_official_duplicate_boundary

audited = pd.read_csv(VALIDATED, dtype=str, keep_default_na=False)
decisions = pd.read_csv(DECISIONS, dtype=str, keep_default_na=False)
review = build_duplicate_review(audited, decisions)
groups = review.drop_duplicates("group_id")
unresolved = groups[(groups["evidence_type"] == "perceptual") & (groups["review_status"] == "unresolved")]
if not unresolved.empty:
    raise ValueError(f"Human duplicate review is incomplete: {len(unresolved)} perceptual groups remain unresolved.")
enforce_official_duplicate_boundary(review)
print("Duplicate review complete and official boundary guard passed.")'''),
        code(r'''# Confirm this only after checking source-specific permissions documented in docs/DATASET.md.
SOURCE_PERMISSIONS_CONFIRMED = False
if not SOURCE_PERMISSIONS_CONFIRMED:
    raise RuntimeError("Set SOURCE_PERMISSIONS_CONFIRMED=True only after recording the source permission decision.")'''),
        code(r'''from covid_xray.data import build_manifest

valid = audited[audited["validation_status"].eq("valid")].copy()
index = valid.rename(columns={
    "resolved_path": "path",
    "original_partition": "original_split",
})[["path", "patient_id", "label", "sha256", "perceptual_hash", "original_split"]]

PROCESSED = OUTPUT_ROOT / "processed"
MODELS = OUTPUT_ROOT / "models"
PROCESSED.mkdir(parents=True, exist_ok=True)
MANIFEST = PROCESSED / "manifest.csv"
manifest = build_manifest(index, val_size=0.15, review=decisions)
manifest.to_csv(MANIFEST, index=False)
display(pd.crosstab(manifest["split"], manifest["label"], margins=True))'''),
        code(r'''import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))
if not tf.config.list_physical_devices("GPU"):
    raise RuntimeError("Enable a Kaggle GPU accelerator before training.")

subprocess.run([
    sys.executable, "-m", "covid_xray.train",
    "--manifest", str(MANIFEST), "--output", str(MODELS),
    "--epochs", "30", "--batch-size", "32",
], check=True)
print("Best model:", MODELS / "model.keras")'''),
        markdown("Use **Save Version → Save & Run All**, then attach this saved output to notebook 04. Keep the model private until its redistribution permission is confirmed."),
    ],
)

write(
    "04_model_evaluation_kaggle.ipynb",
    [
        markdown("""# 04 · Kaggle held-out evaluation

Attach the saved output from notebook 03. Evaluate the official held-out test set only once after model selection."""),
        code(BOOTSTRAP),
        code(FIND_FILE + '\nMANIFEST = find_one("manifest.csv")\nMODEL = find_one("model.keras")'),
        code(r'''RESULTS = OUTPUT_ROOT / "evaluation"
subprocess.run([
    sys.executable, "-m", "covid_xray.evaluate",
    "--manifest", str(MANIFEST), "--model", str(MODEL),
    "--output", str(RESULTS), "--batch-size", "32",
], check=True)'''),
        code(r'''import json
from IPython.display import Image as DisplayImage, display

metrics = json.loads((RESULTS / "metrics.json").read_text())
display(metrics)
for name in ("confusion_matrix.png", "roc_curves.png", "error_analysis.png"):
    path = RESULTS / name
    if path.exists():
        display(DisplayImage(filename=str(path)))
print("Evaluation outputs:", RESULTS)
print("Experimental research output; not a medical diagnosis.")'''),
        markdown("Use **Save Version → Save & Run All** and download the executed notebook plus `covid_xray_outputs`. Upload them for repository review; large/private artifacts will remain outside Git."),
    ],
)

print(f"Built Kaggle notebooks in {OUTPUT}")

# Kaggle can load this source archive as an Input without GitHub credentials.
archive_base = OUTPUT / "covid_xray_project_source"
if archive_base.with_suffix(".zip").exists():
    archive_base.with_suffix(".zip").unlink()
staging = ROOT / ".kaggle_source_staging" / "covid-xray-project"
if staging.parent.exists():
    shutil.rmtree(staging.parent)
staging.mkdir(parents=True)
for name in ("src", "scripts", "pyproject.toml", "requirements.txt", "README.md", "docs"):
    source = ROOT / name
    target = staging / name
    if source.is_dir():
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(source, target)
shutil.make_archive(str(archive_base), "zip", staging.parent, "covid-xray-project")
shutil.rmtree(staging.parent)
print(f"Built Kaggle source archive: {archive_base.with_suffix('.zip')}")
