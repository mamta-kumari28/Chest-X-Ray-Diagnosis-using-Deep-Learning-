# Kaggle run files

Upload the four `.ipynb` files in this folder to Kaggle. First upload
`covid_xray_project_source.zip` as a private Kaggle Dataset and attach it as an
Input. This avoids GitHub authentication for the private repository. Also attach
dataset `andyczhao/covidx-cxr2`, select version 7, and use this order:

1. `02_preprocessing_kaggle.ipynb` (Internet on, accelerator off)
2. `01_data_exploration_kaggle.ipynb` (accelerator off)
3. Complete the private duplicate review described at the end of notebook 01.
4. `03_cnn_training_kaggle.ipynb` (GPU on)
5. `04_model_evaluation_kaggle.ipynb` (GPU on)

After each notebook finishes, select **Save Version → Save & Run All**. Attach
the preceding notebook's saved output to the next notebook. Notebook 03 also
needs the completed `duplicate_decisions.csv` as a private Kaggle input.

The notebooks prefer the attached project-source ZIP and only try GitHub clone
when that ZIP is absent. Generated medical-data metadata, review sheets, models,
and large artifacts must not be committed.

No notebook contains fabricated outputs. Only outputs produced by the attached
version-7 dataset and the actual trained model should be retained.
