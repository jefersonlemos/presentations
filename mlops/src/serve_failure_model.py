import os
import time
from typing import Dict, List

import httpx
from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response


DEFAULT_KSERVE_INFER_URL = (
    "http://failure-model-predictor.mlops-poc.svc.cluster.local"
    "/v2/models/failure-model/infer"
)
KSERVE_INFER_URL = os.getenv("KSERVE_INFER_URL", DEFAULT_KSERVE_INFER_URL)
KSERVE_READY_URL = os.getenv(
    "KSERVE_READY_URL",
    KSERVE_INFER_URL.rsplit("/infer", 1)[0] + "/ready",
)
KSERVE_TIMEOUT_SECONDS = float(os.getenv("KSERVE_TIMEOUT_SECONDS", "5"))
MODEL_CLASSES = [
    label.strip()
    for label in os.getenv(
        "MODEL_CLASSES",
        "crash_loop,healthy,image_pull_error,oom_killed,probe_failure,scheduling_failure",
    ).split(",")
    if label.strip()
]

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


def build_kserve_request(observation: KubernetesObservation):
    values = observation.model_dump()
    return {
        "parameters": {"content_type": "pd"},
        "inputs": [
            {
                "name": "restart_count",
                "datatype": "INT64",
                "shape": [1],
                "data": [values["restart_count"]],
            },
            {
                "name": "cpu_usage_pct",
                "datatype": "FP64",
                "shape": [1],
                "data": [values["cpu_usage_pct"]],
            },
            {
                "name": "memory_usage_pct",
                "datatype": "FP64",
                "shape": [1],
                "data": [values["memory_usage_pct"]],
            },
            {
                "name": "pod_ready",
                "datatype": "INT64",
                "shape": [1],
                "data": [values["pod_ready"]],
            },
            {
                "name": "last_exit_code",
                "datatype": "INT64",
                "shape": [1],
                "data": [values["last_exit_code"]],
            },
            {
                "name": "waiting_reason",
                "datatype": "BYTES",
                "parameters": {"content_type": "str"},
                "shape": [1],
                "data": [values["waiting_reason"]],
            },
            {
                "name": "oom_killed_count",
                "datatype": "INT64",
                "shape": [1],
                "data": [values["oom_killed_count"]],
            },
            {
                "name": "image_pull_errors",
                "datatype": "INT64",
                "shape": [1],
                "data": [values["image_pull_errors"]],
            },
            {
                "name": "failed_scheduling_events",
                "datatype": "INT64",
                "shape": [1],
                "data": [values["failed_scheduling_events"]],
            },
            {
                "name": "readiness_probe_failures",
                "datatype": "INT64",
                "shape": [1],
                "data": [values["readiness_probe_failures"]],
            },
        ],
    }


def output_data(response_body: dict, name: str) -> List:
    for output in response_body.get("outputs", []):
        if output.get("name") == name:
            return output.get("data", [])
    return []


def first_output_data(response_body: dict) -> List:
    outputs = response_body.get("outputs", [])
    if not outputs:
        return []
    return outputs[0].get("data", [])


def class_probabilities(probabilities: List[float]) -> Dict[str, float]:
    return {
        label: float(probability)
        for label, probability in zip(MODEL_CLASSES, probabilities)
    }


@app.get("/health")
async def health():
    kserve_ready = False
    kserve_error = None
    try:
        async with httpx.AsyncClient(timeout=KSERVE_TIMEOUT_SECONDS) as client:
            response = await client.get(KSERVE_READY_URL)
            kserve_ready = response.status_code < 500
    except httpx.HTTPError as exc:
        kserve_error = str(exc)

    return {
        "status": "ok",
        "kserve_infer_url": KSERVE_INFER_URL,
        "kserve_ready": kserve_ready,
        "kserve_error": kserve_error,
    }


@app.post("/predict")
async def predict(observation: KubernetesObservation):
    REQUEST_COUNT.inc()
    start_time = time.time()

    try:
        payload = build_kserve_request(observation)
        async with httpx.AsyncClient(timeout=KSERVE_TIMEOUT_SECONDS) as client:
            response = await client.post(KSERVE_INFER_URL, json=payload)
        response.raise_for_status()
        inference = response.json()

        predicted_values = output_data(inference, "predict") or first_output_data(inference)
        if not predicted_values:
            raise ValueError("KServe response did not include prediction output data.")

        predicted_failure_type = str(predicted_values[0])
        probabilities = class_probabilities(output_data(inference, "predict_proba"))
        if probabilities:
            healthy_probability = probabilities.get("healthy", 0.0)
            risk_score = 1.0 - healthy_probability
        else:
            risk_score = 0.0 if predicted_failure_type == "healthy" else 1.0

        PREDICTION_COUNT.labels(failure_type=predicted_failure_type).inc()

        return {
            "predicted_failure_type": predicted_failure_type,
            "risk_score": round(risk_score, 4),
            "class_probabilities": {
                label: round(probability, 4)
                for label, probability in probabilities.items()
            },
            "served_by": "kserve",
        }
    except httpx.HTTPStatusError as exc:
        ERROR_COUNT.inc()
        raise HTTPException(
            status_code=502,
            detail={
                "message": "KServe inference request failed.",
                "status_code": exc.response.status_code,
                "response": exc.response.text,
            },
        ) from exc
    except Exception as exc:
        ERROR_COUNT.inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        REQUEST_LATENCY.observe(time.time() - start_time)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
