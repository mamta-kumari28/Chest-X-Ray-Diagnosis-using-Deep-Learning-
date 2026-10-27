# Application Setup and Deployment

## Current Status

The Streamlit application has been tested locally on Windows.
The public showcase is deployed at:
https://umang-chest-xray-showcase.streamlit.app/

Public access and recorded-result displays were checked in an
incognito browser. Interactive inference is available in the public application.
On first prediction, the app downloads the original evaluated
checkpoint from GitHub Releases and verifies its SHA-256 before loading.

Model weights are not tracked in Git; they are distributed as
the model-v1 release asset.

Verified locally:
- Trained checkpoint loading.
- Image upload and preview.
- Image preprocessing and prediction.
- Display of saved official test metrics and confusion matrix.
- Upload limit displayed as 10 MB.

## Requirements

- Python 3.12.
- Project source code and dependencies.
- The trained best_model.keras checkpoint.

Model weights are excluded from Git. Obtain the checkpoint from
your authorized training backup and place it at:

```text
artifacts/models/best_model.keras
```

Do not retrain the model merely to launch the application.

## Install Locally

Run commands from the repository root.

Create a virtual environment:

```powershell
py -3.12 -m venv .venv
```

Activate it in Windows Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

Install dependencies and the project:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip check
```

If the virtual environment already exists, activate it and reuse it.

## Checkpoint Integrity

The application verifies the model file's SHA-256 before loading it.

Expected checkpoint checksum:

```text
806d01759ea3f389d1b052b5f2a5ae5bb6e9bd0d3a4931cb328726c0227530c9
```

A different file is rejected with a checksum mismatch error.
Do not replace the expected checksum merely to accept an
unverified checkpoint.

For local source development, use an editable installation:

```bash
python -m pip install -e .
```

Restart Streamlit after changing model-loading code.

## Run Locally

From the repository root, with the virtual environment active:

```powershell
python -m streamlit run app.py
```

Open the Local URL printed in the terminal.

To stop the application, press Ctrl+C in its terminal.

## Verify the Application

1. Expand Official test results.
2. Confirm accuracy is 78.50%, positive recall is 66.00%,
   and positive F1 is 75.43%.
3. Confirm the saved confusion matrix renders.
4. Upload an authorized, anonymized PNG or JPEG chest X-ray.
5. Confirm the preview appears and Classify image returns a score
   and label without an application error.

A successful prediction request verifies the software flow.
It does not establish that the classification is correct.

The decision rule is positive when score > 0.5.
Scores are not calibrated medical probabilities.

## Configuration

The upload limit is configured in .streamlit/config.toml:

```toml
[server]
maxUploadSize = 10
```

The application also checks uploaded file size.

Streamlit 1.45.1 is pinned in requirements.txt. In the tested local
environment, changing to this version resolved an image-serving
failure observed with the previously installed version.

## Public Deployment Requirements

The public showcase is deployed. Public interactive inference
remains pending review of applicable source terms.

Before deployment:

- Choose a hosting provider with suitable Python and TensorFlow support.
- Install the project package so covid_xray imports resolve.
- Supply the trained checkpoint at the expected model path.
- Confirm permission to distribute or host trained weights.
- Verify dependency compatibility in the hosting environment.
- Keep credentials and patient-level manifests out of public files.
- Test startup, image upload, prediction, and saved metric displays.

Private Kaggle inputs and local model files are not automatically
available to a deployed application. Do not assume that connecting
the GitHub repository alone supplies the checkpoint.

## Limitations

The application does not verify that an upload is a chest X-ray.
Non-X-ray images can also receive predictions.

The model missed 68 of 200 positive official test images.
It is not validated for clinical diagnosis and must not guide
medical decisions.

## Public Deployment and Usage Terms

The public application supports interactive image classification.
Checkpoint download, checksum verification, and prediction were
functionally tested on Streamlit Community Cloud.

The project owner confirmed that applicable terms were clarified
before enabling public inference. This confirmation has not been
independently verified here.

Related clarification request:
https://github.com/BIMCV-CSUSP/BIMCV-COVID-19/issues/39

Dataset attribution and applicable source terms remain relevant.
Deployment does not establish clinical suitability.