# Completion checklist

The codebase is complete and pushed to GitHub. Local validation passed with 90 tests. No real-data training or medical performance result is claimed yet.

Only the following account-owned or externally restricted actions remain.

## 1. Publish the cloud environment

In the cloud environment settings, review the saved draft, select **Save**, and then **Publish**. The draft contains the tested Python installation/startup instructions and Kaggle network domains. This action cannot be performed from repository code.

## 2. Accept the dataset terms

Sign in to Kaggle and open [COVIDx CXR-3A, version 7](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2/versions/7). Review and accept the dataset terms plus the original component-source licenses documented in [DATASET.md](DATASET.md). Some component sources restrict nonacademic use or redistribution. Exclude sources for which you do not have permission and document the resulting modified cohort.

## 3. Configure Kaggle access in Colab

Create a Kaggle API token from **Kaggle → Settings → API**. In Google Colab, add `KAGGLE_USERNAME` and `KAGGLE_KEY` through the Secrets panel and enable notebook access. Never paste either value into this repository, notebook cells, chat, screenshots, or GitHub Actions.

## 4. Run the GPU notebook

Open the [training notebook in Colab](https://colab.research.google.com/github/mamta-kumari28/Chest-X-Ray-Diagnosis-using-Deep-Learning-/blob/main/notebooks/train_colab.ipynb), select **Runtime → Change runtime type → T4 GPU**, and run the cells in order. The notebook performs:

1. environment installation and GPU verification;
2. pinned metadata verification;
3. patient metadata import and image indexing;
4. patient/duplicate-safe train-validation-test preparation;
5. real-data EDA;
6. custom CNN training with callbacks;
7. held-out test evaluation;
8. private artifact export.

If the dataset download command needs credentials, expose the two Colab secrets to environment variables only for that runtime. Do not save notebook outputs containing paths, patient IDs, or restricted images back into Git.

## 5. Preserve and review the output

Download or copy the generated private artifact bundle to Google Drive before the Colab runtime disconnects. It contains the best `.keras` model, class order, checksum-bound metadata, training curves, confusion matrix, ROC curves, metrics, and error analysis.

Review actual results before publishing anything. Replace the pending fields in [MODEL_CARD.md](MODEL_CARD.md) only with values produced by the held-out evaluation. Check component licenses before publishing model weights, sample images, or error montages.

## 6. Deploy after evaluation

Publish the permitted three-file model bundle to a versioned HTTPS release:

- `model.keras`
- `class_names.json`
- `model_metadata.json`

Then follow [DEPLOYMENT.md](DEPLOYMENT.md). For Streamlit Community Cloud, choose `deploy_app.py` as the entry point and configure the public model release URL and trusted SHA-256 in its Secrets editor. Select the final **Deploy** action from your Streamlit account and verify one valid image plus one invalid upload.

Until steps 2–5 are completed, the app intentionally shows a model-unavailable state and does not fabricate predictions.
