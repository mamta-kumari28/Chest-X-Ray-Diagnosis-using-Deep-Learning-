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
notebooks/
  01_data_exploration.ipynb  Validated Phase 2 EDA through the package
  02_preprocessing.ipynb     Metadata import and audit; no final split
  03_cnn_training.ipynb      GPU training through covid_xray.train
  04_model_evaluation.ipynb  Held-out evaluation through covid_xray.evaluate
  train_colab.ipynb          Retained end-to-end Colab workflow
tests/                       Data integrity, model, inference, app, evaluation tests
docs/                        Dataset, deployment, and model-card documentation
data/raw/                    Ignored local licensed medical data
data/processed/              Ignored local indexes and manifests
artifacts/models/            Ignored model checkpoints and training history
artifacts/results/           Ignored EDA and evaluation reports
sample_images/               Ignored licensed demo-image location
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
2. Convert the official labels into metadata, audit local files, and generate EDA. Phase 2 preserves the official partitions and does not create final training splits:

```bash
python scripts/import_covidx_metadata.py \
  --labels data/raw/train_COVIDx_CXR-3A.txt data/raw/test_COVIDx_CXR-3A.txt \
  --data-dir data/raw/covidx --output data/processed/patients.csv

python -m covid_xray.eda \
  --metadata data/processed/patients.csv \
  --data-dir data/raw/covidx \
  --validated-output data/processed/validated_metadata.csv \
  --output artifacts/results/eda
```

The Phase 2 metadata contains `image_path`, `label`, `patient_id`, `original_partition`, `source`, local resolution status, and candidate count. The audited CSV adds actual file availability, decode status, dimensions, exact pixel hashes, and perceptual hashes. Missing or ambiguous files remain visible for reporting. Custom data must supply genuine, consistently namespaced patient identifiers; filenames and row numbers are never accepted as inferred IDs.

The official 400-record test partition is preserved exactly in Phase 2. Patient overlap, exact duplicates, perceptual-hash candidates, label conflicts, missing files, corrupt images, unsupported formats, and dimensions are reported without moving records. Phase 3 must review these findings before deriving a validation split from the official training pool.

An authorized Google Colab run validated all **30,034** CXR-3A records against
**30,034** readable local image files. It found no missing, corrupt, unsupported,
ambiguous, or path-conflicting files; no conflicting labels; and no patient ID
overlap across the official train/test boundary. It also found 8 exact decoded-image
duplicate groups (17 records) and 242 matching perceptual-hash groups (581 records).
Those groups remain Phase 3 review inputs rather than evidence of independent
images. See [the verified Phase 2 results](docs/PHASE2_RESULTS.md).

Exact duplicates are detected using decoded RGB pixel hashes, including duplicates whose file metadata differs. A 64-bit perceptual hash additionally groups matching low-frequency signatures, which can catch resized/recompressed copies; such matches are grouped conservatively, not treated as proof of identity. Patient IDs and duplicate hashes connect into groups, including transitive connections. Cross-label exact-duplicate conflicts or official train/test leakage stop the workflow. Loading a manifest recomputes hashes to detect altered images. These checks cannot recover missing patient identities or prove that different perceptual hashes represent unrelated images: manually audit suspicious source overlap, near-matching hashes, and transformed duplicates before making generalization claims. Source-specific preprocessing and hospital artifacts can also leak the label despite disjoint patients.

When readable local images are available, EDA saves `class_distribution.png`, `sample_images.png`, `image_dimensions.png`, `dataset_summary.csv`, and `data_quality_report.json` under `artifacts/results/eda/`. Metadata-record counts and actually validated local-image counts are shown separately. No real-data plots are included in Git; sample previews remain local because source-image redistribution rights vary.

For Google Colab, open `notebooks/02_preprocessing.ipynb`, optionally mount Drive, and set `COVIDX_DATA_DIR` and `COVIDX_LABEL_DIR` to the extracted version-7 images and pinned label-file folder. Then run `notebooks/01_data_exploration.ipynb` with the same `COVIDX_DATA_DIR`. Dataset-host login, acceptance of terms, and any private credentials remain manual account actions.

## Train on a Colab GPU

Use the focused notebooks in numeric order for portfolio review. The retained [end-to-end Colab notebook](notebooks/train_colab.ipynb) provides the same complete workflow in one place. Choose **Runtime → Change runtime type → GPU** for training and use Python 3.11/3.12. All notebooks call the existing `covid_xray` package rather than carrying separate implementations.

The same commands run from a configured GPU checkout:

```bash
python -m covid_xray.train --manifest data/processed/manifest.csv \
  --output artifacts/models --epochs 30 --batch-size 32

python -m covid_xray.evaluate --manifest data/processed/manifest.csv \
  --model artifacts/models/model.keras --output artifacts/results
```

The CNN includes Conv2D, BatchNormalization, MaxPooling2D, Dropout, global average pooling, and Dense layers. Images become RGB at **224 × 224**. The saved model normalizes pixels with `Rescaling(1/255)` and applies small rotation, translation, and zoom **only during training**. Training, validation, and serving use the same pixel-range contract. Class weights are computed on training data only. Bounded parallel `tf.data` workers, batching, and prefetching avoid caching the full dataset in memory.

Training uses EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, and TerminateOnNaN. `model.keras` is the checkpoint with the lowest validation loss. The run also saves `history.json`, `training_curves.png`, `class_names.json`, and `model_metadata.json` with a checksum, class order, input contract, actual epoch counts, and manifest provenance. Existing models are protected unless `--overwrite` is explicit. Training and evaluation reject synthetic test fixtures as portfolio data. Fixed seeds improve repeatability; hardware/software differences can still affect results.

## Evaluate and interpret

The test set is used by the separate evaluation command after model selection. Keep its manifest unchanged: evaluation checks both the model checksum and the exact training-manifest checksum.

Outputs include accuracy; macro/weighted precision, recall, and F1; a per-class classification report; confusion matrix; one-vs-rest and macro ROC-AUC where defined; ROC curves; per-image predictions; FP/FN counts and examples for every class; and a local grid of incorrect predictions. Undefined metrics are recorded with reasons rather than invented values. Scores are uncalibrated softmax outputs. Metrics describe images on held-out source-identified patients, not prospective clinical performance.

Investigate false-negative COVID-19 cases and false positives separately, review source/hospital and acquisition confounding, inspect class imbalance, and document limitations in [the model card](docs/MODEL_CARD.md). Retuning after inspecting test results makes that set development data and requires a new independent test cohort.

## Web app and deployment

Run `streamlit run app.py` to preview the interface. Install the actual trained bundle into `artifacts/models/`, or configure `MODEL_PATH` to a `.keras` file with its JSON companions. The app verifies checksums and class order, checks actual image content and size, previews uploads, and plots all three class scores. It cannot determine whether an arbitrary valid image is a chest X-ray.

For large files, publish a permitted, evaluated model bundle separately and use:

```bash
python scripts/download_model.py --base-url https://YOUR-HOST/VERSION \
  --sha256 YOUR_TRUSTED_MODEL_SHA256 --output artifacts/models
```

The URL and checksum above are placeholders; no trained model download is presently published. [Deployment instructions](docs/DEPLOYMENT.md) cover Streamlit Community Cloud and container/VM hosting, optional explicit artifact provisioning, and manual deployment actions. No deployment is performed automatically by this repository.

## Verification and remaining work

Local tests use synthetic temporary fixtures to verify patient/hash isolation, deterministic preprocessing, real TensorFlow optimization and Keras save/load, upload handling, artifact verification, app startup, and metric/report logic. Fixture results are never presented as medical model results. GitHub Actions runs lint and tests on Python 3.12.

To complete the real-data experiment: obtain dataset access and required source permissions, download the exact images, run preprocessing and EDA, train on a Colab GPU, evaluate once on the untouched test set, document actual results, and then explicitly publish permitted code/model artifacts and deploy. Patient metadata, images, secrets, notebooks with outputs, trained weights, and generated reports belong outside Git.
