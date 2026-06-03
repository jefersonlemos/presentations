import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURE_COLUMNS = [
    "restart_count",
    "cpu_usage_pct",
    "memory_usage_pct",
    "pod_ready",
    "last_exit_code",
    "waiting_reason",
    "oom_killed_count",
    "image_pull_errors",
    "failed_scheduling_events",
    "readiness_probe_failures",
]
TARGET_COLUMN = "label"

BASIC_NUMERIC_FEATURES = [
    "restart_count",
    "cpu_usage_pct",
    "memory_usage_pct",
    "pod_ready",
    "last_exit_code",
]
EVENT_NUMERIC_FEATURES = [
    "oom_killed_count",
    "image_pull_errors",
    "failed_scheduling_events",
    "readiness_probe_failures",
]
EVENT_CATEGORICAL_FEATURES = ["waiting_reason"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Kubernetes failure prediction model.")
    parser.add_argument("--data", default="data/kubernetes_failures.csv", help="Training dataset path.")
    parser.add_argument("--output-dir", default="artifacts/failure_model", help="Artifact output directory.")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db", help="MLflow tracking URI.")
    parser.add_argument("--experiment-name", default="Kubernetes Failure Prediction", help="MLflow experiment name.")
    parser.add_argument("--run-name", default="random-forest-failure-model", help="MLflow run name.")
    parser.add_argument(
        "--feature-set",
        choices=["basic", "events"],
        default="events",
        help="Use basic pod metrics only, or include Kubernetes event features.",
    )
    parser.add_argument("--test-size", type=float, default=0.3, help="Fraction of rows used for testing.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees.")
    return parser.parse_args()


def load_dataset(path):
    dataset = pd.read_csv(path)
    missing_columns = sorted(set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(dataset.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")
    return dataset


def select_features(feature_set):
    if feature_set == "basic":
        return BASIC_NUMERIC_FEATURES, []

    return BASIC_NUMERIC_FEATURES + EVENT_NUMERIC_FEATURES, EVENT_CATEGORICAL_FEATURES


def build_pipeline(numeric_features, categorical_features, n_estimators, random_state):
    transformers = []
    if categorical_features:
        transformers.append(
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        )
    transformers.append(("numeric", "passthrough", numeric_features))

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_model(
    dataset,
    feature_columns,
    numeric_features,
    categorical_features,
    test_size,
    random_state,
    n_estimators,
):
    x = dataset[feature_columns]
    y = dataset[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = build_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1_weighted": f1_score(y_test, predictions, average="weighted"),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }

    return model, metrics


def save_artifacts(
    model,
    metrics,
    output_dir,
    feature_columns,
    numeric_features,
    categorical_features,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "model.joblib"
    metrics_path = output_path / "metrics.json"
    schema_path = output_path / "feature_schema.json"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    schema_path.write_text(
        json.dumps(
            {
                "features": FEATURE_COLUMNS,
                "selected_features": feature_columns,
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "target": TARGET_COLUMN,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return model_path, metrics_path, schema_path


def log_to_mlflow(
    args,
    dataset,
    model,
    metrics,
    model_path,
    metrics_path,
    schema_path,
    feature_columns,
):
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "data": args.data,
                "dataset_rows": len(dataset),
                "test_size": args.test_size,
                "random_state": args.random_state,
                "n_estimators": args.n_estimators,
                "feature_set": args.feature_set,
                "features": ",".join(feature_columns),
                "target": TARGET_COLUMN,
            }
        )
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("f1_weighted", metrics["f1_weighted"])

        mlflow.log_artifact(args.data, artifact_path="data")
        mlflow.log_artifact(metrics_path, artifact_path="reports")
        mlflow.log_artifact(schema_path, artifact_path="schema")
        mlflow.log_artifact(model_path, artifact_path="joblib")
        mlflow.sklearn.log_model(model, artifact_path="model")

        return run.info.run_id


def main():
    args = parse_args()
    dataset = load_dataset(args.data)
    numeric_features, categorical_features = select_features(args.feature_set)
    feature_columns = numeric_features + categorical_features
    model, metrics = train_model(
        dataset=dataset,
        feature_columns=feature_columns,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )
    model_path, metrics_path, schema_path = save_artifacts(
        model,
        metrics,
        args.output_dir,
        feature_columns,
        numeric_features,
        categorical_features,
    )
    run_id = log_to_mlflow(
        args,
        dataset,
        model,
        metrics,
        model_path,
        metrics_path,
        schema_path,
        feature_columns,
    )

    print(f"model_artifact: {model_path}")
    print(f"metrics_artifact: {metrics_path}")
    print(f"schema_artifact: {schema_path}")
    print(f"mlflow_run_id: {run_id}")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"f1_weighted: {metrics['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()
