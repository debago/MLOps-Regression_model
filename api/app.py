import os
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from ruamel import yaml

# 👉 Create FastAPI app
app = FastAPI(title="ML Model API", version="1.0")


# 👉 Env-based MLflow URI (important for Docker/K8s)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# 👉 Model URI (can switch stage easily)
MODEL_URI = os.getenv("MODEL_URI", "models:/iris-model@latest")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# 👉 Load model safely
model = None
model_loaded = False

try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    model_loaded = True
except Exception as e:
    print(f"❌ Model load failed: {e}")

# 👉 Input schema (VERY IMPORTANT 🔥)
class IrisInput(BaseModel):
    sepal_length_cm: float
    sepal_width_cm: float
    petal_length_cm: float
    petal_width_cm: float


@app.get("/")
def home():
    return {"message": "🚀 ML Model API is running"}

# ✅ BASIC HEALTH CHECK
@app.get("/health")
def health():
    return {"status": "ok"}

# 🔥 DEEP HEALTH CHECK (Recommended)
@app.get("/healthz")
def healthz():
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded
    }


@app.post("/predict")
def predict(input_data: IrisInput):

    if not model_loaded:
        return {"error": "Model not loaded"}

    # 👉 Convert input to DataFrame
    df = pd.DataFrame([{
        "sepal length (cm)": input_data.sepal_length_cm,
        "sepal width (cm)": input_data.sepal_width_cm,
        "petal length (cm)": input_data.petal_length_cm,
        "petal width (cm)": input_data.petal_width_cm,
    }])

    prediction = model.predict(df)

    return {
        "prediction": int(prediction[0])
    }