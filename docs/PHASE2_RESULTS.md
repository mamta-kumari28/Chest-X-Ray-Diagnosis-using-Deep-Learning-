# Verified Phase 2 dataset results

These are observed data-quality results from the repository's Phase 2 workflow,
not model-training or evaluation metrics. The repository owner ran the workflow
in Google Colab on 2026-10-16 using the Kaggle COVIDx CXR-3 version-7 image
archive and the pinned three-class CXR-3A label files documented in
`docs/DATASET.md`.

The archive's bundled `train.txt` and `test.txt` use binary labels and were not
used as three-class metadata. The workflow verified the pinned CXR-3A label-file
checksums before matching their 30,034 records to the archive.

## Observed cohort

| Official partition | Class | Records | Readable images | Unique upstream patient IDs |
| --- | --- | ---: | ---: | ---: |
| Train | COVID-19 | 15,994 | 15,994 | 2,808 |
| Train | Normal | 8,085 | 8,085 | 8,085 |
| Train | Pneumonia | 5,555 | 5,555 | 5,531 |
| Test | COVID-19 | 200 | 200 | 178 |
| Test | Normal | 100 | 100 | 100 |
| Test | Pneumonia | 100 | 100 | 100 |
| **Total** |  | **30,034** | **30,034** | **16,802** |

The per-class patient counts and 16,802 total match the independently recomputed
pinned-label statistics in `docs/DATASET.md`. The audit found no patient ID shared
across the official train/test boundary. These remain upstream identifiers and
were not independently clinically verified.

## Data-quality observations

| Check | Observed result |
| --- | ---: |
| Unique metadata records | 30,034 |
| Unique accessible image files | 30,034 |
| Readable images | 30,034 |
| Missing files | 0 |
| Corrupt files | 0 |
| Unsupported files | 0 |
| Ambiguous filename records | 0 |
| Unsafe paths | 0 |
| Duplicate metadata records | 0 |
| Duplicate image-path records | 0 |
| Patient IDs crossing official train/test | 0 |
| Image-path label conflicts | 0 |
| Exact-content label conflicts | 0 |

Pillow decoded every image. Exact duplicate detection hashes decoded RGB pixels
and dimensions; it found **8 groups covering 17 records**. The 64-bit perceptual
hash produced **242 matching groups covering 581 records**. A perceptual-hash
match is a review candidate and does not prove that two images are duplicates.

## Phase 3 gate

Before creating a train/validation split or training a model:

1. Inspect all exact duplicate groups and determine whether any group crosses the
   official train/test boundary. The held-out test records must not be moved.
2. Review perceptual-hash candidates for resized, recompressed, cropped, mirrored,
   or otherwise related images. Record the review rule and outcome.
3. Keep each patient and accepted duplicate group wholly within one derived
   train/validation partition. Abort for review if any accepted group connects
   the official training pool to the held-out test set.
4. Record any source exclusions required by image-use terms and rerun this audit
   on the resulting modified cohort before training.

Generated reports and plots stay under ignored output directories. The aggregate
JSON/CSV supplied for this review contained no image paths or patient IDs. The
sample-image montage, patient-level metadata, validated manifest, raw archive,
and credentials were not committed.
