# Dataset Setup and Audit

## Source and Version

Dataset: COVIDx, Kaggle Version 7 — CXR-3 update.

Source:
https://www.kaggle.com/datasets/andyczhao/covidx-cxr2

The current dataset page may display a newer release title.
Select and pin Version 7 explicitly.

## Original Layout

```text
dataset_root/
├── train/
├── test/
├── train.txt
└── test.txt
```
## Label Format

Each non-empty label row contains four fields:

```text
patient_id filename class source
```

Class labels are positive and negative.

## Verified Counts

| Official split | Negative | Positive | Total |
| --- | ---: | ---: | ---: |
| Train | 13,992 | 15,994 | 29,986 |
| Test | 200 | 200 | 400 |
| Total | 14,192 | 16,194 | 30,386 |

These counts were verified from the selected Version 7 label files.

## Label Checksums

SHA-256 checksums of the original label-file bytes:

```text
train.txt
52af9a4fdc769396bc5ce530872cf0d51bce8fd7e25579b333007d4ebee6a6ba

test.txt
7c92ae0bef49652005e1541a1f2b774e313be4431a891cb0aa5539d37ca2a71e
```

Verify these checksums before reusing saved split manifests.

## Training Data Sources

Source names and counts recorded from the original training labels:

| Source label | Images |
| --- | ---: |
| stonybrook | 13,636 |
| rsna | 13,588 |
| sirm | 943 |
| ricord | 896 |
| cohen | 567 |
| bimcv | 200 |
| actmed | 132 |
| fig1 | 24 |
| Total | 29,986 |

These are counts before exact-file deduplication.

## Audit Findings

- All 30,386 labeled image files existed and were successfully decoded.
- No duplicate label rows or repeated filenames were found within
  either official split.
- No patient ID strings or filenames were shared between the
  official training and test labels.
- Training images included L, P, RGB, and RGBA modes.
- All test images used grayscale mode L.

SHA-256 file hashing identified nine duplicate groups containing
19 files, all within the official training split.

No byte-identical files crossed the official train/test boundary.
No duplicate group had conflicting class labels.
One duplicate group contained two different patient ID strings.

These checks do not rule out near duplicates, identical images
with different encodings, or unidentified patient relationships.

## Derived Training Pool

Original image files and label files were preserved.

Training rows were sorted by SHA-256, source, patient ID, and
filename. One representative per exact-file hash was retained.

Ten redundant copies were excluded, leaving 29,976 images:
13,991 negative and 15,985 positive.

Patient identities were represented by (source, patient_id).
Identities connected by identical training files were linked
before splitting, including links from excluded copies.

This produced 16,647 linked groups from 16,648 source-specific
patient identities.

## Training and Validation Split

The first fold of StratifiedGroupKFold defined validation:

```python
StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
```

Preserve the deduplicated manifest ordering used in the exploration
notebook, because input ordering affects split reproduction.

| Derived split | Negative | Positive | Total |
| --- | ---: | ---: | ---: |
| Training | 11,188 | 12,685 | 23,873 |
| Validation | 2,803 | 3,300 | 6,103 |

Training and validation shared no filenames, exact-file hashes,
or constructed patient groups.

The official test split remained unchanged at 400 images and
was reserved for final evaluation.

## Reproducing the Kaggle Workflow

1. Pin the original COVIDx dataset to Version 7.
2. Run 01_data_exploration.ipynb to generate audit reports and manifests.
3. Run 02_preprocessing.ipynb to verify preprocessing and batching.
4. Run 03_cnn_training.ipynb to train and save the model.
5. Run 04_model_evaluation.ipynb to evaluate the saved checkpoint.

The executed notebooks reference attached Kaggle datasets containing
the original images, private audit manifests, a project-source
snapshot, and a trained-model backup.

Private inputs are not automatically available to other users.
Regenerate manifests or supply an authorized copy as needed.

Exact split reproduction depends on compatible package versions
and matching input ordering.

The scikit-learn version used for the original split was not recorded.
To reproduce the exact split, compare regenerated manifests with the
privately retained original manifests; matching split counts alone
does not prove identical membership.

Fresh-account end-to-end reproduction has not yet been verified.

## Local Data

Optional local dataset location:

```text
data/raw/covidx_cxr3a_v7/
```

Raw images and processed datasets are excluded from Git.

## Usage Terms

Review the Kaggle data card and applicable source-image licenses
before downloading, using, or redistributing data.

Do not commit credentials or dataset archives. Confirm redistribution
permissions before publishing patient-level manifests or images.

This repository does not grant rights to the original images.