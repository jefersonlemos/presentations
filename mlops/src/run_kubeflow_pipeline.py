"""Submit the compiled failure-prediction pipeline to Kubeflow Pipelines."""

import argparse

import kfp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://kubeflow-pipelines.mlops.local")
    parser.add_argument(
        "--pipeline",
        default="pipelines/kubernetes_failure_prediction.yaml",
    )
    parser.add_argument(
        "--experiment",
        default="Kubernetes Failure Prediction POC",
    )
    parser.add_argument("--namespace", default="kubeflow")
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args()

    client = kfp.Client(host=args.host)
    experiment = client.create_experiment(
        name=args.experiment,
        namespace=args.namespace,
    )
    run = client.run_pipeline(
        experiment_id=experiment.experiment_id,
        job_name=args.job_name,
        pipeline_package_path=args.pipeline,
        params={
            "run_name": args.run_name,
            "n_estimators": args.n_estimators,
        },
        enable_caching=False,
    )
    print(f"experiment_id={experiment.experiment_id}")
    print(f"run_id={run.run_id}")


if __name__ == "__main__":
    main()
