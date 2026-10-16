# Phase 2 audit

This phase extends the existing package instead of replacing it. The audit was
performed against Phase 1 on `main` before creating `phase2/dataset-eda`.

## Reused implementation

- `src/covid_xray/config.py` already defines the canonical three classes and
  project paths.
- `src/covid_xray/data.py` already contains image indexing, patient-aware split
  safeguards, decoded-pixel hashes, and perceptual hashes.
- `src/covid_xray/eda.py` already creates plots from a completed manifest.
- `scripts/import_covidx_metadata.py` already verifies the pinned CXR-3A label
  filenames, checksums, labels, patient IDs, and official partitions.
- Existing tests, CI, Streamlit applications, training modules, and deployment
  files remain in place.

## Phase 2 gaps addressed

- The strict importer previously stopped at the first missing image. It now has
  a reporting mode that retains missing or ambiguous records, while preserving
  the strict option.
- Metadata validation now reports metadata records and accessible, readable
  image files separately. It checks paths, formats, decoding, dimensions,
  labels, duplicate rows, pixel duplicates, perceptual-hash candidates, label
  conflicts, and official-partition patient overlap.
- The EDA workflow now works before a final train/validation split and writes
  plots only from images that were actually decoded.
- The focused notebooks now cover Phase 2 metadata preparation and EDA. They do
  not create final splits or train a model.
- Synthetic pytest fixtures cover parser compatibility, labels, valid images,
  missing files, corrupt files, unsupported formats, duplicates, and reports.
- `covid_xray.review` now produces an ignored, patient-level duplicate review CSV
  and optional contact sheets. It distinguishes confirmed exact matches from
  perceptual candidates and validates explicit human decisions.
- Manifest creation now links patients, exact duplicates, and only human-accepted
  perceptual groups. It stops for exact, accepted, or unresolved candidate links
  across the official train/test boundary.

## Environment finding

The two pinned upstream label files were reachable and their documented SHA-256
checksums and row counts were verified on 2026-10-16. Direct access to the
Kaggle version-7 archive was blocked by the development environment's proxy.
The repository owner subsequently ran the supplied workflow in an authorized
Google Colab session against the complete archive. Its aggregate findings are
recorded in `docs/PHASE2_RESULTS.md`. Medical images, patient-level CSV files,
and the sample-image montage remain outside Git.

The prior aggregate report did not include group-level partition membership.
Consequently, the 8 exact groups and 242 perceptual candidates still require the
private workflow and human review described in `docs/PHASE2_FINAL_CHECKLIST.md`.
No such human decision is marked complete in the repository.

