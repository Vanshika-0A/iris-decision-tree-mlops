# Kaggle Iris classification — complete MLOps case study

[![CI](https://github.com/Vanshika-0A/iris-decision-tree-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Vanshika-0A/iris-decision-tree-mlops/actions/workflows/ci.yml)

Predict three Iris flower species from sepal length, sepal width, petal length and petal width (centimetres). The model is a decision tree (entropy criterion, maximum depth 3) trained in an executed Jupyter notebook on the actual Kaggle CSV.

## Dataset and results

- **Exact Kaggle dataset:** [Iris Species — UCI Machine Learning (`uciml/iris`)](https://www.kaggle.com/datasets/uciml/iris).
- **Dataset uploaded in this repository:** [data/Iris.csv](data/Iris.csv). See [dataset source and column details](data/README.md).
- Download: `https://www.kaggle.com/api/v1/datasets/download/uciml/iris`.
- 150 rows, 4 measurements, 3 balanced classes. The identifier is excluded.
- Stratified split: 120 training rows, 30 held-out test rows, random seed 42.
- Test accuracy: **96.67% (29/30)**. Macro F1: **0.9666**.
- Confusion matrix, per-class precision/recall/F1, split IDs, versions and CSV checksum are in `reports/` and the notebook outputs.

This is a small educational case study. Results on 30 samples do not establish production reliability. No model selection uses the test set. The CSV is included for reproducible offline reruns; original data attribution and reuse terms remain with the source dataset.

## 1. Train and test in Jupyter (outside VS Code)

Use Python 3.12. In a terminal opened in this project folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-notebook.txt
jupyter notebook notebooks/01_iris_training.ipynb
```

Use **Run All** in Jupyter. Alternatively, execute through Jupyter's own kernel using:

```powershell
python -m nbconvert --to notebook --execute --inplace notebooks/01_iris_training.ipynb
```

The committed notebook already contains the executed training/test outputs. It saves `models/model.joblib`, `models/parameters.json`, `reports/metrics.json` and `reports/training_provenance.json`. The joblib bundle contains the estimator and its parameters, feature order, classes, metrics and source checksum. Load joblib artifacts only from trusted sources.

## 2. Test the API in VS Code

Open this folder in VS Code, select the Python environment, then use its terminal:

```powershell
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m uvicorn main:app --host 127.0.0.1 --port 8020
```

Open [interactive API documentation](http://127.0.0.1:8020/docs). The model loads from the application directory, independently of the shell's current folder. Endpoints: `GET /`, `GET /health`, `GET /model-info`, `POST /predict`.

Example request:

```json
{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}
```

Expected class: `Iris-setosa`. Response includes all three class probabilities, the largest class probability as `confidence`, and the model type. Confidence is a model output, not a calibrated certainty measure. Missing fields, extra fields, non-numeric/non-finite values and values outside `(0, 30]` receive HTTP 422. The broad upper bound prevents unreasonable inputs; it is not a biological distribution detector.

## 3. Run through Docker

Start Docker Desktop and stop any native server using port 8020, then:

```powershell
docker compose up --build -d
python scripts/smoke_test.py http://127.0.0.1:8020
docker compose logs
docker compose down
```

Docker serves the same [Swagger UI](http://127.0.0.1:8020/docs). It runs as an unprivileged user and publishes only on the laptop's loopback address. The smoke test makes real HTTP calls to health, model metadata, OpenAPI and prediction, and checks a rejected request.

## 4. GitHub Actions

`.github/workflows/ci.yml` runs on pushes, pull requests and manual dispatch. It:

1. Installs the pinned direct dependencies on Python 3.12.
2. Tests the committed model and API.
3. Executes the Jupyter notebook to regenerate training evidence.
4. Tests the regenerated model and API.
5. Builds and runs the Docker image.
6. Calls the container's endpoints and verifies valid/invalid predictions.
7. Uploads the executed notebook, model and evaluation reports as a workflow artifact.

See `reports/VERIFICATION.md` for the observed completion status. Direct dependencies are pinned; transitive dependencies are resolved at installation time.

## Version differences

This version adapts the earlier [Iris MLOps example](https://github.com/24f2003970/iris-kaggle-mlops) with a Decision Tree instead of a random forest, entropy splits, a maximum tree depth of 3, and prediction confidence/model metadata. It was retrained and tested independently. The same Kaggle Iris CSV and held-out split are retained for a comparable educational exercise.
