# Dataset collection, provenance, and leakage prevention

## Selected plan: COVIDx CXR-3A, version 7

Use the **three-class COVIDx CXR-3A** release, documented by the original
[COVID-Net authors](https://github.com/lindawangg/COVID-Net/blob/9645505646161c7eae5fc1a8ecbfa34a77551f2f/docs/COVIDx.md).
It provides **30,034 image records** before local integrity checks, with all three
requested classes and upstream patient identifiers. This is an image count, not
a patient count. The authors' similarly named CXR-3B release is binary and must
not be substituted. More recent versions on the same Kaggle page may have
different images, labels, licenses, and class definitions.

| Official partition | Normal | Non-COVID pneumonia | COVID-19 | Total images | Unique upstream patient IDs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training pool | 8,085 | 5,555 | 15,994 | 29,634 | 16,424 |
| Held-out test | 100 | 100 | 200 | 400 | 378 |
| Total | 8,185 | 5,655 | 16,194 | **30,034** | **16,802** |

These counts were recomputed from the actual author-maintained label files on
**2026-10-16** and match the authors' documentation. They are not this project's
training or evaluation results. Counts may decrease after rights-based source
exclusions, corrupt-image checks, or duplicate review. Validation is selected
from the training pool; its size depends on independent patient groups.

The classes are mutually exclusive benchmark labels: `Pneumonia` means the
benchmark's **non-COVID pneumonia**, while `COVID-19` may itself include patients
with pneumonia. They do not represent a complete differential diagnosis.

## Verified links and limits of verification

| Resource | Link | Check on 2026-10-16 |
| --- | --- | --- |
| Author's pinned dataset instructions | [COVIDx.md](https://raw.githubusercontent.com/lindawangg/COVID-Net/9645505646161c7eae5fc1a8ecbfa34a77551f2f/docs/COVIDx.md) | HTTPS 200; class counts and release links inspected |
| Official three-class training labels | [train_COVIDx_CXR-3A.txt](https://raw.githubusercontent.com/lindawangg/COVID-Net/9645505646161c7eae5fc1a8ecbfa34a77551f2f/labels/train_COVIDx_CXR-3A.txt) | HTTPS 200; 29,634 rows, four columns each |
| Official three-class test labels | [test_COVIDx_CXR-3A.txt](https://raw.githubusercontent.com/lindawangg/COVID-Net/9645505646161c7eae5fc1a8ecbfa34a77551f2f/labels/test_COVIDx_CXR-3A.txt) | HTTPS 200; 400 rows, four columns each |
| Image archive release linked by the authors | [Kaggle: COVIDx CXR, version 7](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2/versions/7) | Release link verified in author instructions; direct Kaggle HTTPS blocked by the environment's proxy (`CONNECT 403`), so archive contents, current availability, and wrapper license were not inspected |
| COVID-Net repository license | [LICENSE.md](https://raw.githubusercontent.com/lindawangg/COVID-Net/9645505646161c7eae5fc1a8ecbfa34a77551f2f/LICENSE.md) | HTTPS 200; GNU AGPL 3.0 for the upstream repository; **not blanket permission for source images** |

Pinned metadata file SHA-256 values:

```text
a7e7ef10b00df1bdeb05c5028b04fbe99fe7011593382e2adff8a2550705835a  train_COVIDx_CXR-3A.txt
048629233e9426f1b42ce1d86e898bdb1691ff8544d81fa44b1d6a3657d88871  test_COVIDx_CXR-3A.txt
```

The importer enforces these hashes and exact filenames. If a file changes, fetch
the pinned version; do not replace a checksum merely to silence a failure.
The hashes above verify labels, not downloaded medical images.

## Licenses: inspect the original sources before using images

COVIDx aggregates multiple collections. Public availability and a Kaggle listing
do not establish a common image license. Keep the archive's license notices and
review every included source's terms, attribution, and permitted use. The
repository's code license does not override those terms. This project does not
redistribute images or the upstream label files.

| Source or material | Verified statement / required action |
| --- | --- |
| COVID-Net code, documentation, and distributed metadata | The upstream [repository license](https://github.com/lindawangg/COVID-Net/blob/9645505646161c7eae5fc1a8ecbfa34a77551f2f/LICENSE.md) states GNU AGPL 3.0. Review obligations before redistributing its metadata or incorporating its code; this implementation does not copy upstream model code. |
| Cohen `cohen` collection | Its [README](https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/README.md), HTTPS 200, says each image's license is in `metadata.csv` and may be Apache 2.0, CC BY-NC-SA 4.0, or CC BY 4.0. Metadata, scripts, and documents are CC BY-NC-SA 4.0. Its [schema](https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/SCHEMA.md), HTTPS 200, says a blank image license means unknown. Check the exact image record, source URL, license, and attribution; exclude unknown or incompatible permissions. |
| BIMCV `bimcv` collection | The actual [dataset Research Use Agreement](https://raw.githubusercontent.com/BIMCV-CSUSP/BIMCV-COVID-19/master/BIMCV-COVID19%2B/LICENSE.md), HTTPS 200, permits research use, prohibits sale and redistribution without written permission, and requires case-by-case approval for nonacademic research. Its separate root software license is not the dataset license. A portfolio owner must obtain applicable permission or exclude this source. |
| Figure 1 `fig1` collection | The original [README](https://raw.githubusercontent.com/agchung/Figure1-COVID-chestxray-dataset/master/README.md), HTTPS 200, states research purposes only; an explicit standalone image license was not established in this session. Obtain applicable permission or exclude. |
| Actualmed `actmed` collection | The original [README](https://raw.githubusercontent.com/agchung/Actualmed-COVID-chestxray-dataset/master/README.md), HTTPS 200, states research purposes only; an explicit standalone image license was not established in this session. Obtain applicable permission or exclude. |
| `rsna`, `ricord`, `stonybrook`, and `sirm` collections | These source tokens appear in the inspected labels. The author instructions link the [RSNA challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge), [RICORD source](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230281), [Stony Brook source](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=89096912), and [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database). Current collection-specific image license pages were not reachable/verified here. Review and record the exact releases' terms before training; do not assume the Kaggle wrapper or upstream software license grants image rights. |

For a public portfolio, publish the code, reproducible instructions, and your
permitted aggregate findings. Use only images and derived model artifacts whose
terms permit your intended use. Do not publish patient-level manifests, image
samples, error-analysis montages, or downloaded images without explicit rights.
Keep local EDA and evaluation outputs private until their contents are reviewed.

If sources are excluded, create a new local patient CSV by filtering its `source`
column, retain `original_split`, document the filter and resulting counts, and
rerun indexing and splitting. Do not edit the pinned label files. This becomes a
**modified cohort**, and must not be reported as the full COVIDx CXR-3A benchmark.
The metadata importer intentionally does not claim to grant or check image rights.

## Download and prepare the real data

1. Review the version-7 page and original source terms above. Accept any required
   agreements using the dataset host. Use existing Kaggle authentication if
   available. If needed, configure it securely using Kaggle/Colab secret settings;
   never paste credentials into a notebook, source file, Git, or chat.
2. Download **version 7** from the browser link, rather than the current default
   release. Extract one copy of the image files under `data/raw/covidx/`. The
   versioned archive can contain train and test directories. Plan sufficient
   local disk space for both the archive and extracted images; the archive size
   was not verified here.
3. Optionally use the current official Kaggle CLI. Its dataset identifier supports
   a third numeric version segment, as verified in the [official CLI implementation](https://github.com/Kaggle/kaggle-api/blob/main/src/kaggle/api/kaggle_api_extended.py):

   ```bash
   python -m pip install kaggle
   kaggle datasets download andyczhao/covidx-cxr2/7 --path data/raw/covidx --unzip
   ```

   This command downloads a substantial external archive and was not run in this
   environment. If your CLI rejects version identifiers, use the version-7 browser
   download or upgrade the official CLI. Do not silently download another release.
4. Fetch the small official label files with normal HTTPS certificate checks:

   ```bash
   mkdir -p data/metadata
   curl --fail --location --proto '=https' --tlsv1.2 \
     https://raw.githubusercontent.com/lindawangg/COVID-Net/9645505646161c7eae5fc1a8ecbfa34a77551f2f/labels/train_COVIDx_CXR-3A.txt \
     --output data/metadata/train_COVIDx_CXR-3A.txt
   curl --fail --location --proto '=https' --tlsv1.2 \
     https://raw.githubusercontent.com/lindawangg/COVID-Net/9645505646161c7eae5fc1a8ecbfa34a77551f2f/labels/test_COVIDx_CXR-3A.txt \
     --output data/metadata/test_COVIDx_CXR-3A.txt
   ```

5. Convert labels into the project's metadata schema:

   ```bash
   python scripts/import_covidx_metadata.py \
     --labels data/metadata/train_COVIDx_CXR-3A.txt data/metadata/test_COVIDx_CXR-3A.txt \
     --data-dir data/raw/covidx \
     --output data/metadata/patients.csv
   ```

   The importer rejects missing images, duplicate filenames, changed labels,
   missing IDs, source ID collisions, and official train/test patient overlap.
   It writes paths relative to `--data-dir` and never overwrites an existing CSV.
   It does not infer a patient ID from an image name or create placeholder IDs.
6. After required source exclusions/permissions, build and split the image index:

   ```bash
   python -m covid_xray.data index --data-dir data/raw/covidx \
     --patients data/metadata/patients.csv --output data/processed/index.csv
   python -m covid_xray.data split --index data/processed/index.csv \
     --output data/processed/manifest.csv --val-size 0.15
   ```

   Install the project as described in the main README before these module
   commands. See the Colab notebook for GPU training and local EDA commands.

All downloaded data, generated patient metadata, and model outputs belong in
ignored directories. A legitimate download still does not authorize publishing
the data in GitHub. Keep secrets outside the checkout.

## How the metadata prevents leakage

The official label format is four whitespace-separated columns:

```text
patient_id image_filename benchmark_label source_dataset
```

`scripts/import_covidx_metadata.py` produces:

```text
path,patient_id,source,label,original_split
```

`path` is relative to the extracted image root; canonical labels are `COVID-19`,
`Normal`, and `Pneumonia`. `original_split` is `train` or `test`. Raw upstream
patient IDs remain consistent across images; the inspected release contains no
ID shared between different source tokens and no train/test patient overlap.
Patient IDs are taken from the authors' metadata, not independently clinically
verified identities.

Indexing computes SHA-256 over decoded RGB pixels and image dimensions, so
identical pixel arrays saved with different metadata are detected. Splitting
links all images belonging to the same patient or decoded-image hash, including
transitive links. Matching 64-bit perceptual hashes are also grouped conservatively
to flag many resized/recompressed copies; false matches may reduce split flexibility
and must be reviewed rather than ignored. Different perceptual hashes are not proof
of unrelated images. The official test set stays held out; validation comes from
the official training pool. If patient/duplicate links cross that official
boundary, splitting fails for review rather than silently leaking or moving the
test images. Training and evaluation recheck the manifest's integrity.

Exact pixel matching does **not** prove that cropped, resized, recompressed,
mirrored, or lightly edited versions are absent. Before publishing clinical
claims, audit near duplicates with a documented perceptual-similarity process
and manual review, and reconcile any patient overlap across original collections.
Upstream patient IDs may not identify the same person across institutions. These
limitations prevent a guarantee that all real-world patient overlap is eliminated.
Report them alongside any measured performance.

Apply augmentation only to training images. Keep validation/test preprocessing
deterministic; select all tuning choices and stopping criteria without consulting
the test results. Source/country/view differences can cause shortcut learning,
so inspect class-by-source distributions and consider an additional independent
external cohort before interpreting the model's generalization.

## Other public candidates

- [COVID-QU-Ex on Kaggle](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu)
  is a larger candidate often described as 33,920 COVID/Non-COVID/Normal images.
  Direct access was blocked by `CONNECT 403` here; its counts, current license,
  exact Non-COVID disease definition, and genuine patient metadata were not
  independently verified. It is not the project's default. Never rename
  `Non-COVID` as `Pneumonia` without checking that release's documented labels.
- The [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
  is another public candidate linked by the COVID-Net authors. Its exact current
  license and patient mapping were not verified here. `Lung_Opacity` is not a
  synonym for non-COVID pneumonia. An image-only folder layout is insufficient
  for this project's patient-safe split requirement.

If substituting a dataset, record the exact version, source terms, label
definitions, metadata provenance, and patient mapping before indexing. A smaller
auditable cohort is preferable to an unsupported 30,000-image claim.

## What has and has not happened

The HTTPS author instructions, pinned label files, patient/count checks, and
specific license evidence above were inspected. No full image dataset was
downloaded, no real-data EDA or GPU training was performed during dataset research,
and no evaluation scores are asserted. Real sample plots, split class counts,
training curves, and test metrics must be generated from an authorized local
dataset by the supplied project workflow.

For dataset attribution, cite the COVID-Net authors and every used source
collection. The original COVID-Net paper is Wang, Lin, and Wong (2020),
*COVID-Net: a tailored deep convolutional neural network design for detection of
COVID-19 cases from chest X-ray images*, Scientific Reports 10, 19549,
[doi:10.1038/s41598-020-76550-z](https://doi.org/10.1038/s41598-020-76550-z).
The DOI is supplied in the verified upstream README; the publisher endpoint was
not independently reachable in this environment.
