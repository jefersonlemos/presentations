from kfp import compiler, dsl


TRAINING_IMAGE = "ttl.sh/failure-model-training-REPLACE:24h"


@dsl.container_component
def prepare_data(
    source_data_path: str,
    prepared_data: dsl.Output[dsl.Dataset],
):
    return dsl.ContainerSpec(
        image=TRAINING_IMAGE,
        command=["sh", "-c"],
        args=[
            "mkdir -p $(dirname \"$1\") && cp \"$0\" \"$1\"",
            source_data_path,
            prepared_data.path,
        ],
    )


@dsl.container_component
def train_model(
    prepared_data: dsl.Input[dsl.Dataset],
    feature_set: str,
    run_name: str,
    tracking_uri: str,
    model_artifacts: dsl.Output[dsl.Model],
):
    return dsl.ContainerSpec(
        image=TRAINING_IMAGE,
        command=["python3"],
        args=[
            "src/train_failure_model.py",
            "--data",
            prepared_data.path,
            "--output-dir",
            model_artifacts.path,
            "--feature-set",
            feature_set,
            "--run-name",
            run_name,
            "--tracking-uri",
            tracking_uri,
        ],
    )


@dsl.container_component
def evaluate_model(
    model_artifacts: dsl.Input[dsl.Model],
    metrics: dsl.Output[dsl.Metrics],
):
    return dsl.ContainerSpec(
        image=TRAINING_IMAGE,
        command=["sh", "-c"],
        args=[
            "cp \"$0/metrics.json\" \"$1\"",
            model_artifacts.path,
            metrics.path,
        ],
    )


@dsl.container_component
def select_model(
    metrics: dsl.Input[dsl.Metrics],
    model_artifacts: dsl.Input[dsl.Model],
    minimum_f1_weighted: float,
    decision: dsl.Output[dsl.Artifact],
):
    return dsl.ContainerSpec(
        image=TRAINING_IMAGE,
        command=["python3", "-c"],
        args=[
            (
                "import json,sys;"
                "metrics=json.load(open(sys.argv[1]));"
                "f1=metrics['f1_weighted'];"
                "decision={'selected': f1 >= float(sys.argv[3]), "
                "'f1_weighted': f1, 'model_dir': sys.argv[2]};"
                "json.dump(decision, open(sys.argv[4], 'w'), indent=2)"
            ),
            metrics.path,
            model_artifacts.path,
            str(minimum_f1_weighted),
            decision.path,
        ],
    )


@dsl.container_component
def prepare_deployment(
    model_artifacts: dsl.Input[dsl.Model],
    decision: dsl.Input[dsl.Artifact],
    deployment_manifest: dsl.Output[dsl.Artifact],
):
    return dsl.ContainerSpec(
        image=TRAINING_IMAGE,
        command=["python3", "-c"],
        args=[
            (
                "import json,sys;"
                "decision=json.load(open(sys.argv[2]));"
                "manifest='MODEL_PATH=' + sys.argv[1] + '/model.joblib\\n'"
                "+ 'SELECTED=' + str(decision['selected']).lower() + '\\n';"
                "open(sys.argv[3], 'w').write(manifest)"
            ),
            model_artifacts.path,
            decision.path,
            deployment_manifest.path,
        ],
    )


@dsl.pipeline(name="kubernetes-failure-prediction")
def failure_prediction_pipeline(
    source_data_path: str = "data/kubernetes_failures.csv",
    feature_set: str = "events",
    run_name: str = "kubeflow-random-forest-events",
    tracking_uri: str = "http://mlflow.mlops-poc.svc.cluster.local:5000",
    minimum_f1_weighted: float = 0.8,
):
    prepared_data = prepare_data(source_data_path=source_data_path)
    trained_model = train_model(
        prepared_data=prepared_data.outputs["prepared_data"],
        feature_set=feature_set,
        run_name=run_name,
        tracking_uri=tracking_uri,
    )
    evaluation = evaluate_model(model_artifacts=trained_model.outputs["model_artifacts"])
    decision = select_model(
        metrics=evaluation.outputs["metrics"],
        model_artifacts=trained_model.outputs["model_artifacts"],
        minimum_f1_weighted=minimum_f1_weighted,
    )
    prepare_deployment(
        model_artifacts=trained_model.outputs["model_artifacts"],
        decision=decision.outputs["decision"],
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=failure_prediction_pipeline,
        package_path="pipelines/kubernetes_failure_prediction.yaml",
    )
