# Model Card

## Overview

This project uses a custom convolutional neural network to classify
chest X-ray images as COVID-positive or COVID-negative.

The model was built using TensorFlow/Keras and trained from random
initialization. No pretrained architecture or weights were used.

## Intended Use

Educational demonstrations and research experiments.

This model is not clinically validated and must not be used for
diagnosis, screening, treatment, or other medical decisions.

## Dataset

Source: https://www.kaggle.com/datasets/andyczhao/covidx-cxr2

Selected release: Kaggle Version 7, CXR-3 update.

Original labeled images:

| Split | Images |
| --- | ---: |
| Official training | 29,986 |
| Official test | 400 |
| Total | 30,386 |

All labeled image paths existed and all images were successfully
decoded during the audit.

Nine byte-identical duplicate groups were identified within the
official training split. Ten redundant copies were excluded from
derived manifests; original files were preserved.

## Data Splitting

Patient identities were represented by (source, patient_id).
Identities linked by byte-identical training files were grouped
together before splitting.

The first fold of StratifiedGroupKFold with five folds,
shuffle=True, and random_state=42 defined the validation set.

| Split | Negative | Positive | Total |
| --- | ---: | ---: | ---: |
| Training | 11,188 | 12,685 | 23,873 |
| Validation | 2,803 | 3,300 | 6,103 |
| Official test | 200 | 200 | 400 |

Training and validation shared no constructed patient groups,
filenames, or exact-file hashes. No byte-identical files were
shared between the official training and test splits.

These checks do not rule out near duplicates or unidentified
patient relationships.

## Preprocessing

- Apply EXIF orientation metadata.
- Composite transparent images onto a black background.
- Convert to single-channel grayscale.
- Resize with aspect-ratio preservation and black padding.
- Target dimensions: 224 × 224 × 1.
- Normalize pixels to [0, 1] using float32.

Training and application inference use the shared preprocessing
function in src/covid_xray/data.py.

## Architecture

Three convolution blocks with 32, 64, and 128 filters.

Each block contains convolution, batch normalization, ReLU
activation, and max pooling.

The classification head uses global average pooling, a 64-unit
dense layer, dropout with rate 0.4, and a single sigmoid output.

Total parameters: 101,665.

## Training

- Optimizer: Adam.
- Initial learning rate: 0.001.
- Loss: binary cross-entropy.
- Batch size: 32.
- Maximum epochs: 30.
- Completed epochs: 9.
- Best checkpoint: epoch 4, selected by minimum validation loss.
- Early stopping patience: 5.
- Learning rate reduced on validation-loss plateaus.
- Class weights calculated from the training subset.
- Training-only augmentation: small rotations and translations.
- Recorded training duration: approximately 55 minutes.
- Training used one Tesla T4 GPU on Kaggle.

Validation and test images were not augmented.

## Evaluation

Class mapping: negative = 0, positive = 1.

Decision rule: positive when model score > 0.5.
The threshold was fixed before official test evaluation.

| Metric | Validation | Official test |
| --- | ---: | ---: |
| Accuracy | 89.91% | 78.50% |
| Positive precision | 92.39% | 88.00% |
| Positive recall | 88.64% | 66.00% |
| Positive F1 | 90.47% | 75.43% |
| ROC-AUC | 0.9572 | 0.9055 |

Official test confusion counts:

| Outcome | Count |
| --- | ---: |
| True negatives | 182 |
| False positives | 18 |
| False negatives | 68 |
| True positives | 132 |

The model missed 68 of 200 positive test images, giving a
false-negative rate of 34%.

Validation was used for checkpoint selection and is not an
independent final performance estimate.

## Limitations

- Official test performance was lower than validation performance.
  The cause of this gap has not been established.
- The test set contains only 400 images and is class-balanced;
  its accuracy and precision may not represent other populations.
- Patient-level uncertainty intervals and subgroup evaluations
  have not been calculated.
- Model scores have not been calibrated as medical probabilities.
- The application does not detect or reject non-X-ray images.
- Source-specific artifacts, acquisition differences, and other
  confounding factors have not been ruled out.
- An exploratory single-seed validation comparison found that a
  baseline without augmentation or class weights produced 111
  false negatives, compared with 375 for the original model.
  This does not support a false-negative reduction claim.
- Both checkpoints were selected using validation loss. This
  comparison is not an independent test-set result and does not
  isolate augmentation from class-weight effects.
- Baseline official test performance remains unknown. The application
  still uses the original evaluated checkpoint.
- No prospective or clinical validation has been performed.

## Software and Reproducibility

Kaggle training and evaluation used TensorFlow 2.20.0.

Local development used TensorFlow 2.17.1. The saved checkpoint
loaded successfully and local inference executed, but numerical
prediction equivalence across environments has not been verified.

The repository contains source code, executed notebooks,
evaluation reports, and software tests. Private split manifests
and model binaries are not tracked in Git.

The original evaluated checkpoint is available as a GitHub Release asset:
https://github.com/umang-kumar03/Chest-X-Ray-Diagnosis-using-Deep-Learning-/releases/tag/model-v1

The application verifies the checkpoint's SHA-256 before loading it.
The later baseline model is not the checkpoint used by the application.

Dataset and source-image licenses govern data usage.
This model card does not grant permission to redistribute them.