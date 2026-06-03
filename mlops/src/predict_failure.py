import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


DEFAULT_MODEL_PATH = "artifacts/failure_model_events/model.joblib"


SAMPLE_OBSERVATION = {
    "restart_count": 3,
    "cpu_usage_pct": 58,
    "memory_usage_pct": 97,
    "pod_ready": 0,
    "last_exit_code": 137,
    "waiting_reason": "OOMKilled",
    "oom_killed_count": 2,
    "image_pull_errors": 0,
    "failed_scheduling_events": 0,
    "readiness_probe_failures": 2,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a local model prediction.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def predict(model_path, observation):
    model = joblib.load(model_path)
    input_data = pd.DataFrame([observation])
    predicted_failure_type = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    class_probabilities = {
        label: float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }
    healthy_probability = class_probabilities.get("healthy", 0.0)

    return {
        "predicted_failure_type": predicted_failure_type,
        "risk_score": round(1.0 - healthy_probability, 4),
        "class_probabilities": {
            label: round(probability, 4)
            for label, probability in class_probabilities.items()
        },
    }


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(json.dumps(predict(model_path, SAMPLE_OBSERVATION), indent=2))


if __name__ == "__main__":
    main()
