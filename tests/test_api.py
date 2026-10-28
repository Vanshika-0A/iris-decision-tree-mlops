import unittest

import joblib
from fastapi.testclient import TestClient

from main import app, artifact


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "healthy", "model_loaded": True})

    def test_known_species(self):
        examples = [([5.1, 3.5, 1.4, 0.2], "Iris-setosa"),
                    ([6.0, 2.9, 4.5, 1.5], "Iris-versicolor"),
                    ([6.5, 3.0, 5.8, 2.2], "Iris-virginica")]
        for measurements, expected in examples:
            with self.subTest(expected=expected):
                response = self.client.post("/predict", json=dict(zip(
                    ["sepal_length", "sepal_width", "petal_length", "petal_width"], measurements)))
                self.assertEqual(response.status_code, 200)
                result = response.json()
                self.assertEqual(result["class_name"], expected)
                self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0)
                self.assertEqual(result["confidence"], max(result["probabilities"].values()))
                self.assertEqual(result["model_type"], "DecisionTreeClassifier")

    def test_invalid_requests(self):
        valid = dict(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2)
        for payload in [{}, {**valid, "sepal_length": 0}, {**valid, "petal_width": -1},
                        {**valid, "sepal_width": "bad"}, {**valid, "petal_length": 100},
                        {**valid, "unexpected": 1}]:
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/predict", json=payload).status_code, 422)
        response = self.client.post("/predict", content='{"sepal_length": NaN, "sepal_width": 3, "petal_length": 1, "petal_width": 1}', headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 422)

    def test_metadata(self):
        response = self.client.get("/model-info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["features"]), 4)
        self.assertGreaterEqual(artifact["metrics"]["accuracy"], 0.8)

    def test_saved_parameters(self):
        self.assertEqual(artifact["model"].get_params(), artifact["parameters"])
        self.assertEqual(len(artifact["classes"]), 3)


if __name__ == "__main__":
    unittest.main()
