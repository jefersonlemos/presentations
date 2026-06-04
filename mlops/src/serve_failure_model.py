import os
import time
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response


DEFAULT_MODEL_PATH = "/models/model.joblib"
MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))

REQUEST_COUNT = Counter(
    "failure_model_prediction_requests_total",
    "Total prediction requests received by the failure model API.",
)
ERROR_COUNT = Counter(
    "failure_model_prediction_errors_total",
    "Total prediction requests that failed.",
)
PREDICTION_COUNT = Counter(
    "failure_model_predictions_total",
    "Total predictions by predicted failure type.",
    ["failure_type"],
)
REQUEST_LATENCY = Histogram(
    "failure_model_prediction_latency_seconds",
    "Prediction request latency in seconds.",
)


class KubernetesObservation(BaseModel):
    restart_count: int = Field(ge=0)
    cpu_usage_pct: float = Field(ge=0, le=100)
    memory_usage_pct: float = Field(ge=0, le=100)
    pod_ready: int = Field(ge=0, le=1)
    last_exit_code: int = Field(ge=0)
    waiting_reason: str
    oom_killed_count: int = Field(ge=0)
    image_pull_errors: int = Field(ge=0)
    failed_scheduling_events: int = Field(ge=0)
    readiness_probe_failures: int = Field(ge=0)


app = FastAPI(title="Kubernetes Failure Prediction API")
model = None


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(observation: KubernetesObservation):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    REQUEST_COUNT.inc()
    start_time = time.time()

    try:
        input_data = pd.DataFrame([observation.model_dump()])
        predicted_failure_type = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        class_probabilities: Dict[str, float] = {
            label: float(probability)
            for label, probability in zip(model.classes_, probabilities)
        }
        healthy_probability = class_probabilities.get("healthy", 0.0)
        risk_score = 1.0 - healthy_probability

        PREDICTION_COUNT.labels(failure_type=predicted_failure_type).inc()

        return {
            "predicted_failure_type": predicted_failure_type,
            "risk_score": round(risk_score, 4),
            "class_probabilities": {
                label: round(probability, 4)
                for label, probability in class_probabilities.items()
            },
        }
    except Exception as exc:
        ERROR_COUNT.inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        REQUEST_LATENCY.observe(time.time() - start_time)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
