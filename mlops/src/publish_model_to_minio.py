import argparse
from pathlib import Path

import boto3
from botocore.client import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Publish a trained model artifact to MinIO.")
    parser.add_argument(
        "--model-path",
        default="artifacts/failure_model_events/model.joblib",
        help="Local model artifact to upload.",
    )
    parser.add_argument("--bucket", default="mlops-models", help="Target S3 bucket.")
    parser.add_argument(
        "--key",
        default="failure-model/events/model.joblib",
        help="Target object key. KServe storageUri should point to the parent prefix.",
    )
    parser.add_argument(
        "--endpoint-url",
        default="http://127.0.0.1:9000",
        help="MinIO S3 endpoint.",
    )
    parser.add_argument("--access-key-id", default="minioadmin")
    parser.add_argument("--secret-access-key", default="minioadmin")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    client = boto3.client(
        "s3",
        endpoint_url=args.endpoint_url,
        aws_access_key_id=args.access_key_id,
        aws_secret_access_key=args.secret_access_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

    existing_buckets = {
        bucket["Name"]
        for bucket in client.list_buckets().get("Buckets", [])
    }
    if args.bucket not in existing_buckets:
        client.create_bucket(Bucket=args.bucket)

    client.upload_file(str(model_path), args.bucket, args.key)
    print(f"model_published: s3://{args.bucket}/{args.key}")


if __name__ == "__main__":
    main()
