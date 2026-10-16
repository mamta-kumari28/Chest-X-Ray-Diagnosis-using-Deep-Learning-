# Run and deploy the Streamlit application

The application is an educational demonstration. Its predictions are experimental and are **not medical diagnoses**. Deployment does not establish clinical validity. A model has not been trained or published merely because these application files exist.

## Run locally

Use Python 3.12 and a CPU environment for serving. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
streamlit run app.py
```

The page launches before a trained model exists. It shows model readiness and validates image uploads, but it does not produce a disease classification without a verified artifact bundle. The server normally listens on port 8501; for a cloud workspace, forward that port using the platform's preview feature. For a container that needs an external bind address, use:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## Install your trained model

Follow the Colab training instructions and evaluate the held-out test split first. Keep the three exported files together:

```text
artifacts/
├── model.keras
├── class_names.json
└── model_metadata.json
```

`class_names.json` is a JSON list in this exact order:

```json
["COVID-19", "Normal", "Pneumonia"]
```

The training pipeline writes `model_metadata.json` with at least these fields:

| Field | Required value |
| --- | --- |
| `trained` | `true`, written only after real training |
| `class_names` | The ordered class list above |
| `input_size` | `[224, 224]` |
| `input_range` | `[0, 255]` |
| `sha256` | The actual lowercase SHA-256 of the `.keras` file |

Metadata may also include actual training and dataset provenance. Do not handcraft metadata to present an untrained model as trained. The application rejects missing metadata, incompatible shapes or class order, checksum mismatches, and invalid probability outputs. Loading uses Keras `safe_mode=True` and `compile=False`; only use model files from a source you trust. A checksum verifies integrity against the supplied metadata, not the trustworthiness of an unknown publisher.

For a different location, set the model file path. The two JSON companions must be in the same directory:

```bash
MODEL_PATH=/absolute/path/to/model.keras streamlit run app.py
```

The model accepts RGB float32 pixels in **0–255**. Normalization is embedded in the trained model. Do not normalize the same input a second time in serving code.

## Download large model artifacts

Keep model files and datasets out of Git. Publish your own evaluated artifact bundle to a versioned GitHub Release or another suitable HTTPS artifact host only when you choose to publish. Confirm that your dataset and model redistribution terms permit publication. Record the download location and a separately obtained trusted SHA-256 for each file. Use the repository's checksum-verifying helper; see its exact options with:

```bash
python scripts/download_model.py --help
```

The helper downloads only when you explicitly run it. The default `app.py` performs no network downloads. Transfer all three files and restart the app, or explicitly configure the optional hosted entry point below. A missing file leaves predictions disabled. Never put secret URL tokens in source files, command history, public documentation, or Git commits.

For example, after you publish a reviewed public release containing the three files, replace the repository, tag, and checksum placeholders below:

```bash
python scripts/download_model.py \
  --base-url https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY/releases/download/YOUR_MODEL_TAG \
  --sha256 YOUR_TRUSTED_64_CHARACTER_LOWERCASE_SHA256 \
  --output artifacts
streamlit run app.py
```

The downloader refuses to overwrite an existing bundle. Use a new output directory and update `MODEL_PATH` when installing another model version.

## Deploy to Streamlit Community Cloud

Deployment requires your own GitHub access, a Streamlit account, and the final manual deployment action. No live deployment is claimed by this repository.

1. Review the repository, run its tests, train and evaluate the model, and publish the code to your GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository and branch. Set the entry point to **`deploy_app.py`** and select Python 3.12 in advanced settings.
3. Publish your permitted, evaluated model bundle to a versioned public HTTPS release. Place `model.keras`, `class_names.json`, and `model_metadata.json` under the same release base URL. Obtain the model checksum directly from your trusted training export, not from an untrusted download. No such public release exists yet for this project.
4. Add this configuration through Community Cloud's **Secrets** editor, replacing both placeholders with your actual public release and model hash. Never commit a local `secrets.toml`:

   ```toml
   [model]
   base_url = "https://YOUR-HOST/VERSION"
   sha256 = "YOUR_ACTUAL_64_CHARACTER_LOWERCASE_MODEL_SHA256"
   ```

   `deploy_app.py` explicitly downloads the three files, checks the trusted hash and metadata, caches the bundle outside the checkout, sets `MODEL_PATH`, and renders the same interface. Interrupted or incompatible bundles stop inference with an actionable readiness message. Without configuration it starts as a model-free preview. This startup path uses public release URLs only; private credential injection requires a separate supported hosting integration. On a VM, `MODEL_BASE_URL` and `MODEL_SHA256` environment variables provide the same configuration.
5. No API key is required by this application itself. If artifacts are already provisioned on a container/VM filesystem, use the simpler `app.py` entry point with `MODEL_PATH` instead. Keep all datasets and model files outside the code repository.
6. Select **Deploy** yourself. Confirm the startup log, the model-readiness indicator, and one valid PNG/JPEG prediction. Verify that a corrupt upload is rejected. Readiness should stay unavailable when any companion file is missing or a checksum is wrong.

The hosted downloader was tested with mocked transport and local temporary bundles; no real trained release was downloaded or deployed. Verify the actual release's network access and provider resource limits before expecting public inference. The app remains a useful interface preview without model files.

A minimal container/VM startup after provisioning the environment is:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
# Provision the verified model bundle into /models before starting this process.
MODEL_PATH=/models/model.keras streamlit run app.py --server.address 0.0.0.0
```

Choose a machine with enough memory for TensorFlow, the loaded model, and concurrent requests. Begin with low concurrency and measure real usage; serving normally does not require a GPU. Restrict public access as appropriate for a portfolio demonstration.

## Uploads, limitations, and operational checks

- PNG and JPEG content is decoded and verified; the application rejects empty, corrupt, animated, oversized, and unsupported images. The app limit is 10 MB and 25 megapixels. Grayscale and color images become RGB at 224 × 224.
- A valid image file is not proof that the image is a chest X-ray. There is no automatic anatomy or acquisition-quality check.
- Uploads are processed in memory by the application, and application code does not write them to disk. Remove identifying information before use and review your hosting provider's own data-handling behavior.
- Confidence bars show uncalibrated model class scores. They do not estimate a person's probability of illness. A Normal prediction cannot rule out disease.
- Run `pytest` before publishing. Review the actual evaluation artifacts; never replace missing training results with illustrative performance claims.
- Missing-model, metadata-validation, dependency, or model-shape errors are shown as readiness messages. Check TensorFlow compatibility and the original training export instead of disabling integrity verification.
