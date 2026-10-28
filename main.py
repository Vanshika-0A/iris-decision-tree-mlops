from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

artifact = joblib.load(Path(__file__).parent / "models" / "model.joblib")
model = artifact["model"]
app = FastAPI(title="Iris Decision Tree Prediction API", version="1.0.0")


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    # Do not echo non-finite input values, which cannot be serialized as JSON.
    return JSONResponse(status_code=422, content={"detail": [
        {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]})


class IrisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    sepal_length: float = Field(gt=0, le=30, description="Centimetres")
    sepal_width: float = Field(gt=0, le=30, description="Centimetres")
    petal_length: float = Field(gt=0, le=30, description="Centimetres")
    petal_width: float = Field(gt=0, le=30, description="Centimetres")


@app.get("/")
def index():
    return {"message": "Iris prediction API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.get("/model-info")
def model_info():
    return {key: artifact[key] for key in ("features", "classes", "parameters", "metrics", "dataset_sha256")}


@app.post("/predict")
def predict(data: IrisRequest):
    features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    probabilities = model.predict_proba(features)[0]
    prediction = str(model.predict(features)[0])
    return {
        "class_name": prediction,
        "probabilities": dict(zip(model.classes_.tolist(), probabilities.tolist())),
        "confidence": float(max(probabilities)),
        "model_type": "DecisionTreeClassifier",
    }
