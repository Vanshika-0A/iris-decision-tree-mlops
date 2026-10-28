# Dataset used: Iris Species (Kaggle)

- **Kaggle dataset title:** Iris Species
- **Dataset identifier:** `uciml/iris`
- **Publisher:** UCI Machine Learning
- **Original Kaggle page:** https://www.kaggle.com/datasets/uciml/iris
- **CSV used for training:** [Iris.csv](Iris.csv), included in this GitHub repository.
- **Direct download endpoint:** https://www.kaggle.com/api/v1/datasets/download/uciml/iris
- **Downloaded:** 28 October 2026.
- **CSV SHA-256:** `600ac44f23c2e6e0ae37daac8ceb2baba4df963efa580eb31b3b576b28e34c55`

The data contains 150 flowers: 50 each of Iris-setosa, Iris-versicolor and Iris-virginica.

| Column | Use |
| --- | --- |
| Id | Row identifier; excluded from training |
| SepalLengthCm | Input measurement in centimetres |
| SepalWidthCm | Input measurement in centimetres |
| PetalLengthCm | Input measurement in centimetres |
| PetalWidthCm | Input measurement in centimetres |
| Species | Target class |

The executed [Jupyter notebook](../notebooks/01_iris_training.ipynb) reads this CSV directly. It uses 120 training rows and 30 held-out test rows, with a stratified split and random seed 42. The executed notebook and reports contain this version's measured test accuracy. The exact split IDs and file checksum are saved in [training_provenance.json](../reports/training_provenance.json).

This is the Kaggle CSV, not scikit-learn's bundled `load_iris()` data. Refer to the Kaggle page for the source's attribution and reuse terms.
