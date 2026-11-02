"""Train the production EBM pipeline and log it (params, data hash, model) to MLflow."""

import hashlib
import json
import os
import subprocess

import mlflow
import pandas as pd
from mlflow.models import infer_signature

from src import tracking
from src.config import ROOT, load_params
from src.data import model_inputs
from src.model import build_pipeline, save

MODEL_FILE = ROOT / "models" / "model.joblib"


def git_sha() -> str:
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def file_md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    params = load_params()
    m = params["model"]
    train_path = ROOT / params["data"]["processed_dir"] / "train.csv"
    df = pd.read_csv(train_path)
    X, y = model_inputs(df), df["y"]

    if not tracking.configure().startswith("http"):
        print("WARNING: DAGSHUB_OWNER / DAGSHUB_TOKEN not set, logging to a local MLflow store.")
    mlflow.set_experiment(m["experiment"])

    with mlflow.start_run(run_name="ebm-pipeline") as run:
        mlflow.set_tags({"git_sha": git_sha(), "source": "dvc-pipeline", "algorithm": "EBM"})
        mlflow.log_params({**m["ebm"], "train_rows": len(df)})
        mlflow.log_params(
            {
                "train_positive_rate": round(float(y.mean()), 4),
                "train_data_md5": file_md5(train_path),
            }
        )

        model = build_pipeline(m["ebm"]).fit(X, y)
        save(model, MODEL_FILE)

        sample = X.head(200)
        # cloudpickle keeps the EBM loadable by mlflow.sklearn.load_model (the newer default,
        # skops, refuses the interpret classes unless they are explicitly trusted).
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=infer_signature(sample, model.predict_proba(sample)[:, 1]),
            input_example=X.head(3),
        )
        run_info = {"run_id": run.info.run_id, "model_uri": info.model_uri}
        (ROOT / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
        print(f"Logged run {run.info.run_id} -> {info.model_uri}")


if __name__ == "__main__":
    main()
