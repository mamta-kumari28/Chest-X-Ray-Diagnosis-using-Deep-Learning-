# Phase 2 final validation checklist

Phase 2 is **code-complete but not fully human-verified**. The prior authorized
Colab run established that 30,034 files decoded successfully. The private report
needed to classify duplicate groups and their official-partition relationships
has not been supplied, and component-source permissions remain owner decisions.

## Completion criteria and evidence

| Criterion | Status | Evidence or action required |
| --- | --- | --- |
| Pinned CXR-3A labels verified | Complete | Checksums and 30,034 rows in `docs/DATASET.md` |
| All referenced files readable | Previously observed complete | Authorized Colab report summarized in `docs/PHASE2_RESULTS.md`; rerun after exclusions |
| Missing/corrupt/unsupported/path checks | Previously observed complete | All zero in the supplied aggregate report |
| Official patient-ID overlap check | Previously observed complete | Zero in the supplied aggregate report |
| Exact group partition audit | Pending private run | Generate `duplicate_review.csv`; cross-partition exact count must be zero or have an approved exclusion policy |
| Perceptual candidate review | Pending human review | Mark every candidate group `accepted` or `rejected`; no `unresolved` cross-partition group may remain |
| Group-aware split guard | Code-complete | Synthetic tests cover exact, accepted, rejected, and unresolved behavior |
| Source permissions for intended training | Pending owner decision | Record source-specific training exclusions/permissions |
| Public model-weight sharing permission | Pending owner decision | Obtain a source-by-source decision before publishing weights |
| Final cohort statistics after exclusions | Pending if exclusions occur | Filter by `source`, rerun validation, and report the modified cohort |

## Exact Colab commands

Run from the latest repository checkout after mounting Drive. These commands do
not train a model or create a final train/validation split.

```bash
%cd /content/covid-xray-project

!PYTHONPATH=src python -m covid_xray.data duplicates \
  --validated-metadata "/content/drive/MyDrive/covidx-cxr3a-v7/processed/validated_metadata.csv" \
  --output "/content/drive/MyDrive/covidx-cxr3a-v7/eda/private/duplicate_review.csv" \
  --summary "/content/drive/MyDrive/covidx-cxr3a-v7/eda/duplicate_review_summary.json" \
  --contact-sheets "/content/drive/MyDrive/covidx-cxr3a-v7/eda/private/contact_sheets"
```

Open the private CSV and contact sheets only in an authorized environment. For
each `perceptual:*` group, set one consistent `review_status` to `accepted`,
`rejected`, or `unresolved`, and add a short `review_notes` explanation. Exact
pixel groups remain leakage-linked even if a reviewer disputes their clinical
identity. Save decisions as a private CSV, then validate them:

```bash
!PYTHONPATH=src python -m covid_xray.data duplicates \
  --validated-metadata "/content/drive/MyDrive/covidx-cxr3a-v7/processed/validated_metadata.csv" \
  --decisions "/content/drive/MyDrive/covidx-cxr3a-v7/eda/private/duplicate_decisions.csv" \
  --output "/content/drive/MyDrive/covidx-cxr3a-v7/eda/private/reviewed_duplicates.csv" \
  --summary "/content/drive/MyDrive/covidx-cxr3a-v7/eda/reviewed_duplicate_summary.json"
```

Inspect the summary. Phase 3 may proceed only when:

- `cross_partition_exact_groups` is zero;
- `unresolved_cross_partition_perceptual_groups` is zero;
- `accepted_cross_partition_perceptual_groups` is zero;
- source permissions and exclusions are recorded; and
- any modified cohort has been re-audited.

When Phase 3 later creates a manifest, pass the private decisions file with
`--duplicate-review`. The splitter preserves the official test set and stops if
the decision state is unsafe:

```bash
!PYTHONPATH=src python -m covid_xray.data split \
  --index "/content/drive/MyDrive/covidx-cxr3a-v7/processed/index.csv" \
  --duplicate-review "/content/drive/MyDrive/covidx-cxr3a-v7/eda/private/duplicate_decisions.csv" \
  --output "/content/drive/MyDrive/covidx-cxr3a-v7/processed/manifest.csv"
```

The last command belongs to Phase 3 and is shown only to document the guard. Do
not run it as part of Phase 2.
