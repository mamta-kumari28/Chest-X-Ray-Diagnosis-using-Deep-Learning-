# Phase 1 repository audit

Audit scope: preserve the existing implementation and organize the repository for
separate portfolio phases. No dataset was downloaded, no model was trained, and no
metric was generated during this work.

## Existing implementation retained

| Requirement | Existing implementation | Decision |
| --- | --- | --- |
| Dataset metadata, hashing and patient-safe splitting | `src/covid_xray/data.py` | Retained as the single implementation |
| EDA tables and figures | `src/covid_xray/eda.py` | Retained; notebook calls `create_report` |
| Custom CNN and embedded preprocessing | `src/covid_xray/model.py` | Retained unchanged |
| Training and callbacks | `src/covid_xray/train.py` | Retained; default output moved to `artifacts/models` |
| Evaluation and error analysis | `src/covid_xray/evaluate.py` | Retained; defaults moved to organized artifact folders |
| Upload and model-bundle validation | `src/covid_xray/inference.py` | Retained; default model location updated |
| End-to-end GPU workflow | `notebooks/train_colab.ipynb` | Retained and aligned with artifact folders |
| Dataset/model helper scripts | `scripts/` | Retained; downloader default aligned |
| Streamlit interfaces | `app.py`, `deploy_app.py` | Retained; model guidance aligned |
| Automated verification | `tests/`, `.github/workflows/tests.yml` | Retained |
| Dataset, model and deployment documentation | `docs/` | Retained |

## Missing target structure added

- `notebooks/01_data_exploration.ipynb`
- `notebooks/02_preprocessing.ipynb`
- `notebooks/03_cnn_training.ipynb`
- `notebooks/04_model_evaluation.ipynb`
- tracked documentation placeholders for `data/raw`, `data/processed`,
  `artifacts/models`, `artifacts/results`, and `sample_images`

The new notebooks are thin orchestration layers. They import or invoke the existing
package, contain no alternate CNN/data/evaluation logic, and have no saved outputs.
The EDA notebook depends on a validated manifest; the preprocessing notebook must
therefore be run once before first EDA. Numeric filenames match the requested
portfolio structure, while code dependencies remain explicit and fail with clear
messages when prerequisites are absent.

## Dependency and path impact

No package dependency was added or removed. Editable installation still exposes
`src/covid_xray`, CI still installs `requirements-lock.txt`, and all notebook code
runs from the repository root. Runtime outputs now use:

```text
data/raw/                 licensed source files (ignored)
data/processed/           patient metadata, indexes, manifests (ignored)
artifacts/models/         model bundle and training logs (ignored)
artifacts/results/        EDA and held-out evaluation output (ignored)
sample_images/            optional licensed demos (ignored)
```

`.gitignore` preserves only the explanatory README files in these directories and
also excludes common medical image formats, credentials, checkpoints, archives,
and generated artifacts.
