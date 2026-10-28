# Verification — 28 October 2026

- Dataset: Kaggle **Iris Species**, `uciml/iris`; CSV included under `data/`.
- Jupyter notebook executed with a Python kernel outside VS Code.
- Model: DecisionTreeClassifier, criterion entropy, max_depth 3, random_state 42.
- Training/test split: stratified 120/30, seed 42.
- Held-out accuracy: **29/30 = 96.67%**; macro F1 **0.966583**.
- Saved model and parameters reloaded and verified.
- All five API test groups passed, including confidence scores and invalid inputs.
- Docker Compose API exposed at `http://127.0.0.1:8020`; smoke evidence in `docker-smoke.json`.
- Project opened in VS Code; tests were executed through Python tools.
- GitHub publishing and CI: pending account publication.

Small educational dataset; confidence probabilities are not calibrated real-world certainty.
