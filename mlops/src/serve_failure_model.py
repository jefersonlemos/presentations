import os
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
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
APP_PREDICTION_COUNT = Counter(
    "failure_model_app_predictions_total",
    "Total predictions by application and predicted failure type.",
    ["app", "failure_type"],
)
APP_RISK_SCORE = Gauge(
    "failure_model_app_risk_score",
    "Latest predicted risk score by application.",
    ["app"],
)
APP_TRAFFIC_COUNT = Counter(
    "failure_model_app_traffic_total",
    "Total prediction traffic by application.",
    ["app"],
)
APP_TRAFFIC_SHARE = Gauge(
    "failure_model_app_traffic_share",
    "Latest observed traffic share by application.",
    ["app"],
)
APP_CRITICALITY_SCORE = Gauge(
    "failure_model_app_criticality_score",
    "Latest business criticality score by application.",
    ["app"],
)
APP_CRITICALITY = Gauge(
    "failure_model_app_criticality",
    "Current business criticality label by application (one-hot).",
    ["app", "criticality"],
)
APP_BUSINESS_RISK_SCORE = Gauge(
    "failure_model_app_business_risk_score",
    "Latest business risk score by application.",
    ["app"],
)
APP_CLASS_PROBABILITY = Gauge(
    "failure_model_app_class_probability",
    "Latest predicted class probability by application and failure type.",
    ["app", "failure_type"],
)
APP_PREDICTED_FAILURE = Gauge(
    "failure_model_app_predicted_failure",
    "Current predicted failure class for an application (1 for the active class).",
    ["app", "failure_type"],
)
APP_SIGNAL = Gauge(
    "failure_model_app_signal",
    "Latest normalized or raw operational signal used to explain application risk.",
    ["app", "signal"],
)
LABELED_PREDICTION_COUNT = Counter(
    "failure_model_labeled_predictions_total",
    "Total predictions with delayed production ground truth.",
)
CORRECT_PREDICTION_COUNT = Counter(
    "failure_model_correct_predictions_total",
    "Total predictions matching delayed production ground truth.",
)
CONFUSION_COUNT = Counter(
    "failure_model_prediction_confusion_total",
    "Labeled production predictions by expected and predicted failure type.",
    ["expected_failure_type", "predicted_failure_type"],
)
APP_PREDICTION_CORRECT = Gauge(
    "failure_model_app_prediction_correct",
    "Whether the latest labeled prediction for an application was correct.",
    ["app"],
)
APP_EXPECTED_FAILURE = Gauge(
    "failure_model_app_expected_failure",
    "Current delayed ground-truth failure class for an application.",
    ["app", "failure_type"],
)
REQUEST_LATENCY = Histogram(
    "failure_model_prediction_latency_seconds",
    "Prediction request latency in seconds.",
)

APP_CRITICALITY_PROFILES = {
    "checkout": ("supporting", 0.25),
    "catalog": ("important", 0.60),
    "recommendations": ("important", 0.75),
    "payments": ("core", 1.0),
}
DEFAULT_APP_CRITICALITY = ("supporting", 0.5)
APP_REQUEST_COUNTS = defaultdict(int)
APP_REQUEST_COUNTS_LOCK = threading.Lock()


class KubernetesObservation(BaseModel):
    app_name: str = Field(default="unknown", min_length=1)
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
    expected_failure_type: Optional[str] = None


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


def criticality_profile(app_name: str) -> tuple[str, float]:
    return APP_CRITICALITY_PROFILES.get(app_name, DEFAULT_APP_CRITICALITY)


def business_risk_level(score: float) -> str:
    if score >= 0.85:
        return "critical"
    if score >= 0.65:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def risk_level_from_score(score: float) -> str:
    return business_risk_level(score)


FAILURE_SEVERITY = {
    "healthy": 0.05,
    "image_pull_error": 0.65,
    "probe_failure": 0.70,
    "scheduling_failure": 0.75,
    "crash_loop": 0.90,
    "oom_killed": 0.95,
}


def operational_risk(observation: KubernetesObservation, failure_type: str):
    factors = {
        "model_severity": FAILURE_SEVERITY.get(failure_type, 0.5),
        "not_ready": 0.20 if observation.pod_ready == 0 else 0.0,
        "memory_pressure": max(0.0, observation.memory_usage_pct - 70.0) / 30.0 * 0.25,
        "restart_pressure": min(observation.restart_count / 10.0, 1.0) * 0.15,
        "event_pressure": min(
            observation.oom_killed_count * 0.15
            + observation.image_pull_errors * 0.12
            + observation.failed_scheduling_events * 0.12
            + observation.readiness_probe_failures * 0.08,
            0.35,
        ),
    }
    symptom_risk = min(
        1.0,
        factors["not_ready"]
        + factors["memory_pressure"]
        + factors["restart_pressure"]
        + factors["event_pressure"],
    )
    risk_score = 1.0 - (1.0 - factors["model_severity"]) * (1.0 - symptom_risk)
    factors["symptom_risk"] = symptom_risk
    return min(risk_score, 1.0), factors


def update_traffic_share(app_name: str) -> float:
    with APP_REQUEST_COUNTS_LOCK:
        APP_REQUEST_COUNTS[app_name] += 1
        total_requests = sum(APP_REQUEST_COUNTS.values())
        if total_requests <= 0:
            return 0.0
        return APP_REQUEST_COUNTS[app_name] / total_requests


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
        risk_score, risk_factors = operational_risk(observation, predicted_failure_type)

        PREDICTION_COUNT.labels(failure_type=predicted_failure_type).inc()
        APP_PREDICTION_COUNT.labels(
            app=observation.app_name,
            failure_type=predicted_failure_type,
        ).inc()
        APP_TRAFFIC_COUNT.labels(app=observation.app_name).inc()
        APP_RISK_SCORE.labels(app=observation.app_name).set(risk_score)
        criticality_label, criticality_score = criticality_profile(observation.app_name)
        traffic_share = update_traffic_share(observation.app_name)
        business_risk_score = min(
            1.0, risk_score * criticality_score * traffic_share
        )
        APP_TRAFFIC_SHARE.labels(app=observation.app_name).set(traffic_share)
        APP_CRITICALITY_SCORE.labels(app=observation.app_name).set(criticality_score)
        APP_BUSINESS_RISK_SCORE.labels(app=observation.app_name).set(
            business_risk_score
        )
        for label in {criticality_label, *[value[0] for value in APP_CRITICALITY_PROFILES.values()]}:
            APP_CRITICALITY.labels(app=observation.app_name, criticality=label).set(
                float(label == criticality_label)
            )
        for label in MODEL_CLASSES:
            APP_PREDICTED_FAILURE.labels(
                app=observation.app_name,
                failure_type=label,
            ).set(float(label == predicted_failure_type))
        signals = {
            "cpu_usage_pct": observation.cpu_usage_pct,
            "memory_usage_pct": observation.memory_usage_pct,
            "restart_count": observation.restart_count,
            "pod_ready": observation.pod_ready,
            "oom_killed_count": observation.oom_killed_count,
            "image_pull_errors": observation.image_pull_errors,
            "failed_scheduling_events": observation.failed_scheduling_events,
            "readiness_probe_failures": observation.readiness_probe_failures,
            **risk_factors,
        }
        for signal, value in signals.items():
            APP_SIGNAL.labels(app=observation.app_name, signal=signal).set(value)
        for label, probability in probabilities.items():
            APP_CLASS_PROBABILITY.labels(
                app=observation.app_name,
                failure_type=label,
            ).set(probability)

        quality_match = None
        if observation.expected_failure_type is not None:
            if observation.expected_failure_type not in MODEL_CLASSES:
                raise ValueError(
                    "expected_failure_type must be one of: " + ", ".join(MODEL_CLASSES)
                )
            quality_match = observation.expected_failure_type == predicted_failure_type
            LABELED_PREDICTION_COUNT.inc()
            if quality_match:
                CORRECT_PREDICTION_COUNT.inc()
            CONFUSION_COUNT.labels(
                expected_failure_type=observation.expected_failure_type,
                predicted_failure_type=predicted_failure_type,
            ).inc()
            APP_PREDICTION_CORRECT.labels(app=observation.app_name).set(
                float(quality_match)
            )
            for label in MODEL_CLASSES:
                APP_EXPECTED_FAILURE.labels(
                    app=observation.app_name,
                    failure_type=label,
                ).set(float(label == observation.expected_failure_type))

        return {
            "app_name": observation.app_name,
            "predicted_failure_type": predicted_failure_type,
            "risk_score": round(risk_score, 4),
            "risk_level": (
                risk_level_from_score(risk_score)
            ),
            "criticality_label": criticality_label,
            "criticality_score": round(criticality_score, 4),
            "traffic_share": round(traffic_share, 4),
            "business_risk_score": round(business_risk_score, 4),
            "business_risk_level": business_risk_level(business_risk_score),
            "risk_factors": {
                name: round(value, 4)
                for name, value in risk_factors.items()
            },
            "class_probabilities": {
                label: round(probability, 4)
                for label, probability in probabilities.items()
            },
            "expected_failure_type": observation.expected_failure_type,
            "quality_match": quality_match,
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
