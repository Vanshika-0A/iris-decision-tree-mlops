"""Exercise the running API over real HTTP, including a rejected request."""
import json
import http.client
import sys
import time
import urllib.error
import urllib.request

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8018"
for attempt in range(60):
    try:
        with urllib.request.urlopen(base + "/health", timeout=2) as response:
            health = json.load(response)
        break
    except (urllib.error.URLError, OSError, http.client.HTTPException):
        time.sleep(1)
else:
    raise RuntimeError("API did not become healthy")
assert health == {"status": "healthy", "model_loaded": True}
with urllib.request.urlopen(base + "/model-info") as response:
    assert len(json.load(response)["classes"]) == 3
with urllib.request.urlopen(base + "/openapi.json") as response:
    assert "/predict" in json.load(response)["paths"]
payload = dict(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2)
request = urllib.request.Request(base + "/predict", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request) as response:
    prediction = json.load(response)
assert prediction["class_name"] == "Iris-setosa", prediction
request = urllib.request.Request(base + "/predict", data=b'{}', headers={"Content-Type": "application/json"})
try:
    urllib.request.urlopen(request)
    raise AssertionError("Invalid request was accepted")
except urllib.error.HTTPError as error:
    assert error.code == 422
print(json.dumps({"health": health, "prediction": prediction, "invalid_request_status": 422, "openapi": "passed"}, indent=2))
