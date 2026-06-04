import argparse
import shutil
from pathlib import Path

import mlflow.artifacts


def parse_args():
    parser = argparse.ArgumentParser(description="Download a selected model artifact for serving.")
    parser.add_argument("--model-uri", required=True, help="MLflow artifact URI, for example runs:/<run_id>/joblib/model.joblib.")
    parser.add_argument("--output-path", default="/models/model.joblib", help="Path where the serving API expects the model.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded_path = Path(mlflow.artifacts.download_artifacts(artifact_uri=args.model_uri))
    if downloaded_path.is_dir():
        downloaded_path = downloaded_path / "model.joblib"

    if not downloaded_path.exists():
        raise FileNotFoundError(f"Downloaded model artifact was not found: {downloaded_path}")

    shutil.copyfile(downloaded_path, output_path)
    print(f"model_downloaded: {output_path}")


if __name__ == "__main__":
    main()
