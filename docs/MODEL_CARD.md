# Model card: COVID-19 Chest X-Ray Detection

**Status: not trained on real chest X-rays.** No performance figures, clinical validation, model download URL, or deployment are currently available. Complete this card using the actual training and evaluation artifacts; do not replace blanks with example numbers.

| Item | Current specification |
| --- | --- |
| Intended use | Educational ML portfolio and reproducible research |
| Excluded use | Diagnosis, triage, treatment decisions, or ruling out disease |
| Inputs | De-identified PNG/JPEG, RGB, 224 × 224, pixels 0–255 |
| Architecture | Custom three-block CNN, embedded normalization/augmentation |
| Outputs | COVID-19, Normal, Pneumonia softmax scores in that order |
| Data plan | COVIDx CXR-3A; source-specific access and reuse terms apply |
| Split plan | Preserve official test; patient/duplicate-group validation split |
| Checkpoint selection | Lowest validation loss; test used separately |
| Actual model hash / run date | Pending real training |
| Actual dataset/version/filtering | Pending licensed download and indexing |
| Actual accuracy / per-class recall / F1 / ROC-AUC | Pending held-out evaluation |
| Error review / subgroup results | Pending real evaluation |

These mutually exclusive dataset classes simplify overlapping clinical concepts: COVID-19 may cause pneumonia. Dataset labels reflect source annotation, not a definitive patient diagnosis. Chest X-rays alone do not establish COVID-19 infection.

Report real image and patient counts per split and per source, exclusions, duplicate-review decisions, label uncertainty, best epoch, software/hardware versions, model and manifest hashes, and test results from `metrics.json`. Distinguish undefined metrics from zero. Inspect COVID-19 false negatives and false positives; avoid tuning to the test set. Quantify uncertainty (for example, patient-level bootstrap confidence intervals in a later statistical analysis) before making stronger claims.

Important limitations include source/hospital confounding, scanner and projection artifacts, age and demographic imbalance, imperfect patient metadata, unseen sites, domain shift, label noise, exact-hash limitations for transformed duplicates, and softmax miscalibration. No prospective study, independent external cohort, anatomy-quality gate, fairness study, or regulatory review has been performed. The official test cohort is small relative to the training pool and not evidence of clinical readiness.

Never publish patient identifiers or source images merely because they can be downloaded publicly. Review component licenses and model redistribution terms before releasing weights. This repository's software license does not override any dataset license.
