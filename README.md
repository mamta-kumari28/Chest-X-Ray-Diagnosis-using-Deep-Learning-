# COVID-19 Chest X-Ray Detection using Deep Learning

[![Open training notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamta-kumari28/Chest-X-Ray-Diagnosis-using-Deep-Learning-/blob/main/notebooks/train_colab.ipynb)

For the remaining account-owned actions, use the short [completion checklist](docs/NEXT_STEPS.md).

A reproducible, three-class image-classification project with a custom TensorFlow/Keras CNN, patient-aware data preparation, exploratory analysis, held-out evaluation, and a Streamlit interface.

**Project status:** implementation and local functional checks are available. No real chest X-ray dataset has been downloaded in this workspace, no clinical model has been trained, and no medical performance numbers or live deployment are claimed. The app opens without a model and keeps predictions disabled until a verified trained artifact bundle is installed.

> Educational and experimental use only. Predictions are not medical diagnoses. A Normal prediction cannot rule out illness. Use only de-identified images you are permitted to process.

## Dataset and scope

The recommended dataset is **COVIDx CXR-3A**, with **30,034 chest X-rays** across COVID-19, Normal, and Pneumonia, documented by the COVID-Net authors. Its version-specific metadata supplies patient IDs, filenames, labels, sources, and official train/test assignments. This project uses three mutually exclusive dataset labels; real COVID-19 can itself cause pneumonia, so these labels are not a complete clinical differential diagnosis.

| Class | Official training pool | Official test | Total |
| --- | ---: | ---: | ---: |
| COVID-19 | 15,994 | 200 | 16,194 |
| Normal | 8,085 | 100 | 8,185 |
| Pneumonia | 5,555 | 100 | 5,655 |
| Total | 29,634 | 400 | 30,034 |

Read [the dataset guide](docs/DATASET.md) before downloading. It records verified author documentation, exact release links, metadata checksums, download instructions, and source-specific licenses. Dataset access and all image-reuse permissions remain your responsibility. COVID-Net's software license does **not** license its component images. Some source images restrict nonacademic use and redistribution; never publish generated sample-image figures without permission.

## Project structure

```text
app.py                       Streamlit interface, no fallback predictions
deploy_app.py                Optional release-download entry point for hosted Streamlit
src/covid_xray/
  config.py                  Shared class order and image size
  data.py                    Indexing, hashing, group splitting, tf.data pipelines
  eda.py                     Counts, training samples, image-dimension plots
  model.py                   Custom CNN with saved preprocessing
  train.py                   Callbacks, best checkpoint, provenance, learning curves
  evaluate.py                Held-out metrics, ROC, confusion matrix, FP/FN analysis
  inference.py               Upload checks, model integrity checks, predictions
scripts/
  import_covidx_metadata.py  Version-specific genuine patient metadata importer
  download_model.py         Explicit HTTPS artifact download with checksum checks
notebooks/train_colab.ipynb  Reproducible Colab GPU workflow, no saved outputs
tests/                       Data integrity, model, inference, app, evaluation tests
docs/                        Dataset, deployment, and model-card documentation
data/                        Ignored local data and manifests
artifacts/                   Ignored model, figures, and evaluation output
```

## Local setup

Use **Python 3.11 or 3.12**; the tested runtime is Python 3.12 on Linux. From this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m pytest -q
ruff check .
streamlit run app.py
```

For serving alone, install `requirements.txt` instead of `requirements-dev.txt`. The CPU package is appropriate for local development and serving. Colab uses **`requirements-colab.txt`**, which installs GPU-capable TensorFlow instead; do not install both TensorFlow distributions in the same runtime. No API key is needed to run the local app. Kaggle may require an account or accepted dataset terms for downloads.

For the exact tested **Linux/Python 3.12 CPU development environment**, replace the first dependency-install command with `python -m pip install --require-hashes -r requirements-lock.txt`. This includes development tools and pins transitive dependencies with artifact hashes. CI and the cloud setup use this lock; the Colab requirements remain separate because they use the GPU distribution.

## Prepare data and explore

1. Follow [docs/DATASET.md](docs/DATASET.md) to download the exact CXR-3A images and its two pinned label files. Keep downloaded images outside Git under `data/raw/covidx/`. The latest COVIDx release is not interchangeable with CXR-3A.
2. Convert the original patient metadata, index images, and build the manifest:

```bash
python scripts/import_covidx_metadata.py \
  --labels data/metadata/train_COVIDx_CXR-3A.txt data/metadata/test_COVIDx_CXR-3A.txt \
  --data-dir data/raw/covidx --output data/metadata/patients.csv

python -m covid_xray.data index \
  --data-dir data/raw/covidx --patients data/metadata/patients.csv \
  --output data/processed/index.csv

python -m covid_xray.data split \
  --index data/processed/index.csv --output data/processed/manifest.csv

python -m covid_xray.eda \
  --manifest data/processed/manifest.csv --output artifacts/eda
```

The DataFrame contains `path`, `label`, `patient_id`, `source`, decoded-pixel `sha256`, `perceptual_hash`, `original_split`, and `split`. Paths are stored relative to the manifest so the data tree can move together. The importer fails if metadata checksums differ, filenames are missing or ambiguous, or IDs have conflicting source identities. Custom data must supply genuine, consistently namespaced patient identifiers; filenames and row numbers are never accepted as inferred IDs.

The official 400-image test set is preserved. Approximately 15% of the official training pool becomes validation data, with whole connected patient/duplicate groups kept together. Group sizes can change exact proportions. Without official assignments, custom data uses approximately 70/15/15 train/validation/test. Every split must contain all three classes.

Exact duplicates are detected using decoded RGB pixel hashes, including duplicates whose file metadata differs. A 64-bit perceptual hash additionally groups matching low-frequency signatures, which can catch resized/recompressed copies; such matches are grouped conservatively, not treated as proof of identity. Patient IDs and duplicate hashes connect into groups, including transitive connections. Cross-label exact-duplicate conflicts or official train/test leakage stop the workflow. Loading a manifest recomputes hashes to detect altered images. These checks cannot recover missing patient identities or prove that different perceptual hashes represent unrelated images: manually audit suspicious source overlap, near-matching hashes, and transformed duplicates before making generalization claims. Source-specific preprocessing and hospital artifacts can also leak the label despite disjoint patients.

EDA saves Pandas count tables, Seaborn class/split plots, training sample previews, and a dimension plot for up to 1,000 training images. No real-data plots are included until you run this step. Sample previews remain local because source-image redistribution rights vary.

## Train on a Colab GPU

Open [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb) in Google Colab. Choose **Runtime → Change runtime type → GPU** and use Python 3.11/3.12. The notebook explains cloning this repository, dependency installation, version-pinned metadata downloads, private dataset placement, preprocessing, training, evaluation, and private model export. Use the repository's actual GitHub URL and branch after you publish these files.

The same commands run from a configured GPU checkout:

```bash
python -m covid_xray.train --manifest data/processed/manifest.csv \
  --output artifacts --epochs 30 --batch-size 32

python -m covid_xray.evaluate --manifest data/processed/manifest.csv \
  --model artifacts/model.keras --output artifacts/evaluation
```

The CNN includes Conv2D, BatchNormalization, MaxPooling2D, Dropout, global average pooling, and Dense layers. Images become RGB at **224 × 224**. The saved model normalizes pixels with `Rescaling(1/255)` and applies small rotation, translation, and zoom **only during training**. Training, validation, and serving use the same pixel-range contract. Class weights are computed on training data only. Bounded parallel `tf.data` workers, batching, and prefetching avoid caching the full dataset in memory.

Training uses EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, and TerminateOnNaN. `model.keras` is the checkpoint with the lowest validation loss. The run also saves `history.json`, `training_curves.png`, `class_names.json`, and `model_metadata.json` with a checksum, class order, input contract, actual epoch counts, and manifest provenance. Existing models are protected unless `--overwrite` is explicit. Training and evaluation reject synthetic test fixtures as portfolio data. Fixed seeds improve repeatability; hardware/software differences can still affect results.

## Evaluate and interpret

The test set is used by the separate evaluation command after model selection. Keep its manifest unchanged: evaluation checks both the model checksum and the exact training-manifest checksum.

Outputs include accuracy; macro/weighted precision, recall, and F1; a per-class classification report; confusion matrix; one-vs-rest and macro ROC-AUC where defined; ROC curves; per-image predictions; FP/FN counts and examples for every class; and a local grid of incorrect predictions. Undefined metrics are recorded with reasons rather than invented values. Scores are uncalibrated softmax outputs. Metrics describe images on held-out source-identified patients, not prospective clinical performance.

Investigate false-negative COVID-19 cases and false positives separately, review source/hospital and acquisition confounding, inspect class imbalance, and document limitations in [the model card](docs/MODEL_CARD.md). Retuning after inspecting test results makes that set development data and requires a new independent test cohort.

## Web app and deployment

Run `streamlit run app.py` to preview the interface. Install the actual trained bundle into `artifacts/`, or configure `MODEL_PATH` to a `.keras` file with its JSON companions. The app verifies checksums and class order, checks actual image content and size, previews uploads, and plots all three class scores. It cannot determine whether an arbitrary valid image is a chest X-ray.

For large files, publish a permitted, evaluated model bundle separately and use:

```bash
python scripts/download_model.py --base-url https://YOUR-HOST/VERSION \
  --sha256 YOUR_TRUSTED_MODEL_SHA256 --output artifacts
```

The URL and checksum above are placeholders; no trained model download is presently published. [Deployment instructions](docs/DEPLOYMENT.md) cover Streamlit Community Cloud and container/VM hosting, optional explicit artifact provisioning, and manual deployment actions. No deployment is performed automatically by this repository.

## Verification and remaining work

Local tests use synthetic temporary fixtures to verify patient/hash isolation, deterministic preprocessing, real TensorFlow optimization and Keras save/load, upload handling, artifact verification, app startup, and metric/report logic. Fixture results are never presented as medical model results. GitHub Actions runs lint and tests on Python 3.12.

To complete the real-data experiment: obtain dataset access and required source permissions, download the exact images, run preprocessing and EDA, train on a Colab GPU, evaluate once on the untouched test set, document actual results, and then explicitly publish permitted code/model artifacts and deploy. Patient metadata, images, secrets, notebooks with outputs, trained weights, and generated reports belong outside Git.
