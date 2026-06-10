import os

from kfp import compiler, dsl


TRAINING_IMAGE = os.getenv(
    "TRAINING_IMAGE",
    "ttl.sh/failure-model-training-post-upgrade-20260609:24h",
)


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
    n_estimators: int,
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
            "--n-estimators",
            str(n_estimators),
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
    bucket: str,
    model_object_prefix: str,
    deployment_manifest: dsl.Output[dsl.Artifact],
):
    return dsl.ContainerSpec(
        image=TRAINING_IMAGE,
        command=["python3", "-c"],
        args=[
            (
                "import json,sys;"
                "decision=json.load(open(sys.argv[2]));"
                "storage_uri='s3://' + sys.argv[3].strip('/') + '/' + sys.argv[4].strip('/') + '/';"
                "manifest='SELECTED=' + str(decision['selected']).lower() + '\\n'"
                "+ 'MODEL_PATH=' + sys.argv[1] + '/model.joblib\\n'"
                "+ 'STORAGE_URI=' + storage_uri + '\\n';"
                "open(sys.argv[5], 'w').write(manifest)"
            ),
            model_artifacts.path,
            decision.path,
            bucket,
            model_object_prefix,
            deployment_manifest.path,
        ],
    )


@dsl.container_component
def publish_selected_model(
    model_artifacts: dsl.Input[dsl.Model],
    decision: dsl.Input[dsl.Artifact],
    bucket: str,
    model_object_prefix: str,
    endpoint_url: str,
    access_key_id: str,
    secret_access_key: str,
    publish_result: dsl.Output[dsl.Artifact],
):
    return dsl.ContainerSpec(
        image=TRAINING_IMAGE,
        command=["python3", "-c"],
        args=[
            (
                "import json,subprocess,sys;"
                "decision=json.load(open(sys.argv[2]));"
                "bucket=sys.argv[3];prefix=sys.argv[4].strip('/');"
                "storage_uri='s3://' + bucket + '/' + prefix + '/';"
                "result={'selected': decision['selected'], 'published': False, "
                "'storage_uri': storage_uri, 'f1_weighted': decision['f1_weighted']};"
                "\nif decision['selected']:\n"
                "    subprocess.check_call(["
                "'python3','src/publish_model_to_minio.py',"
                "'--model-path',sys.argv[1] + '/model.joblib',"
                "'--bucket',bucket,"
                "'--key',prefix + '/model.joblib',"
                "'--endpoint-url',sys.argv[5],"
                "'--access-key-id',sys.argv[6],"
                "'--secret-access-key',sys.argv[7]]);"
                "result['published']=True\n"
                "json.dump(result, open(sys.argv[8], 'w'), indent=2)"
            ),
            model_artifacts.path,
            decision.path,
            bucket,
            model_object_prefix,
            endpoint_url,
            access_key_id,
            secret_access_key,
            publish_result.path,
        ],
    )


@dsl.container_component
def deploy_inferenceservice(
    publish_result: dsl.Input[dsl.Artifact],
    namespace: str,
    inferenceservice_name: str,
    deployment_result: dsl.Output[dsl.Artifact],
):
    return dsl.ContainerSpec(
        image=TRAINING_IMAGE,
        command=["python3", "-c"],
        args=[
            (
                "import datetime,json,ssl,sys,time,urllib.request;"
                "result=json.load(open(sys.argv[1]));"
                "namespace=sys.argv[2];name=sys.argv[3];"
                "out={"
                "'selected': result['selected'],"
                "'published': result['published'],"
                "'storage_uri': result['storage_uri'],"
                "'deployed': False};"
                "\nif result['published']:\n"
                "    token=open('/var/run/secrets/kubernetes.io/serviceaccount/token').read();"
                "ca='/var/run/secrets/kubernetes.io/serviceaccount/ca.crt';"
                "url='https://kubernetes.default.svc/apis/serving.kserve.io/v1beta1/namespaces/'"
                "+namespace+'/inferenceservices/'+name;"
                "patch={'metadata':{'annotations':{'mlops-poc/model-published-at':"
                "datetime.datetime.now(datetime.timezone.utc).isoformat()}},"
                "'spec':{'predictor':{'model':{'storageUri':result['storage_uri']}}}};"
                "headers={'Authorization':'Bearer '+token,'Content-Type':'application/merge-patch+json'};"
                "context=ssl.create_default_context(cafile=ca);"
                "request=urllib.request.Request(url,data=json.dumps(patch).encode(),"
                "headers=headers,method='PATCH');"
                "response=json.load(urllib.request.urlopen(request,context=context,timeout=30));"
                "generation=response['metadata']['generation'];"
                "deadline=time.time()+300;"
                "\n    while time.time() < deadline:\n"
                "        get_request=urllib.request.Request(url,headers={'Authorization':'Bearer '+token});"
                "current=json.load(urllib.request.urlopen(get_request,context=context,timeout=30));"
                "ready=next((c for c in current.get('status',{}).get('conditions',[]) "
                "if c.get('type')=='Ready'),{});"
                "\n        if current.get('status',{}).get('observedGeneration',0) >= generation "
                "and ready.get('status')=='True':\n"
                "            out.update({'deployed':True,'generation':generation});break\n"
                "        time.sleep(5)\n"
                "    if not out['deployed']: raise TimeoutError('InferenceService did not become ready')\n"
                "json.dump(out,open(sys.argv[4],'w'),indent=2)"
            ),
            publish_result.path,
            namespace,
            inferenceservice_name,
            deployment_result.path,
        ],
    )


@dsl.container_component
def smoke_test_serving(
    publish_result: dsl.Input[dsl.Artifact],
    predict_url: str,
    smoke_test_result: dsl.Output[dsl.Artifact],
):
    return dsl.ContainerSpec(
        image=TRAINING_IMAGE,
        command=["python3", "-c"],
        args=[
            (
                "import json,sys,urllib.request;"
                "result=json.load(open(sys.argv[1]));"
                "out={'selected': result['selected'], 'published': result['published'], "
                "'tested': False};"
                "\nif result['published']:\n"
                "    payload=json.dumps({"
                "'app_name':'kubeflow-pipeline-smoke-test',"
                "'restart_count':3,'cpu_usage_pct':58.0,'memory_usage_pct':97.0,"
                "'pod_ready':0,'last_exit_code':137,'waiting_reason':'OOMKilled',"
                "'oom_killed_count':2,'image_pull_errors':0,"
                "'failed_scheduling_events':0,'readiness_probe_failures':2"
                "}).encode();"
                "req=urllib.request.Request(sys.argv[2], data=payload, "
                "headers={'Content-Type':'application/json'});"
                "resp=urllib.request.urlopen(req, timeout=20);"
                "out.update({'tested': True, 'status': resp.status, "
                "'response': resp.read().decode('utf-8')[:1000]})\n"
                "json.dump(out, open(sys.argv[3], 'w'), indent=2)"
            ),
            publish_result.path,
            predict_url,
            smoke_test_result.path,
        ],
    )


@dsl.pipeline(name="kubernetes-failure-prediction")
def failure_prediction_pipeline(
    source_data_path: str = "data/kubernetes_failures.csv",
    feature_set: str = "events",
    run_name: str = "kubeflow-random-forest-events",
    tracking_uri: str = "http://mlflow.mlops-poc.svc.cluster.local:5000",
    n_estimators: int = 200,
    minimum_f1_weighted: float = 0.8,
    model_bucket: str = "mlops-models",
    model_object_prefix: str = "failure-model/events",
    s3_endpoint_url: str = "http://minio.mlops-poc.svc.cluster.local:9000",
    s3_access_key_id: str = "minioadmin",
    s3_secret_access_key: str = "minioadmin",
    predict_url: str = "http://failure-model-api.mlops-poc.svc.cluster.local/predict",
    serving_namespace: str = "mlops-poc",
    inferenceservice_name: str = "failure-model",
):
    prepared_data = prepare_data(source_data_path=source_data_path)
    trained_model = train_model(
        prepared_data=prepared_data.outputs["prepared_data"],
        feature_set=feature_set,
        run_name=run_name,
        tracking_uri=tracking_uri,
        n_estimators=n_estimators,
    )
    evaluation = evaluate_model(model_artifacts=trained_model.outputs["model_artifacts"])
    decision = select_model(
        metrics=evaluation.outputs["metrics"],
        model_artifacts=trained_model.outputs["model_artifacts"],
        minimum_f1_weighted=minimum_f1_weighted,
    )
    deployment = prepare_deployment(
        model_artifacts=trained_model.outputs["model_artifacts"],
        decision=decision.outputs["decision"],
        bucket=model_bucket,
        model_object_prefix=model_object_prefix,
    )
    published_model = publish_selected_model(
        model_artifacts=trained_model.outputs["model_artifacts"],
        decision=decision.outputs["decision"],
        bucket=model_bucket,
        model_object_prefix=model_object_prefix,
        endpoint_url=s3_endpoint_url,
        access_key_id=s3_access_key_id,
        secret_access_key=s3_secret_access_key,
    )
    deployed_model = deploy_inferenceservice(
        publish_result=published_model.outputs["publish_result"],
        namespace=serving_namespace,
        inferenceservice_name=inferenceservice_name,
    ).after(deployment)
    smoke_test_serving(
        publish_result=published_model.outputs["publish_result"],
        predict_url=predict_url,
    ).after(deployed_model)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=failure_prediction_pipeline,
        package_path="pipelines/kubernetes_failure_prediction.yaml",
    )
