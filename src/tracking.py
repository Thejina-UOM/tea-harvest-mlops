"""Point MLflow at my DagsHub repository when DAGSHUB_OWNER and DAGSHUB_TOKEN are set.

Codespaces secrets and GitHub Actions both provide these values, so every script calls
configure() before talking to MLflow. Without them, MLflow logs to a local ./mlruns store.
"""

import os

DEFAULT_REPO = "tea-harvest-mlops"


def configure() -> str:
    owner, token = os.getenv("DAGSHUB_OWNER"), os.getenv("DAGSHUB_TOKEN")
    repo = os.getenv("DAGSHUB_REPO") or DEFAULT_REPO
    if owner and token and not os.getenv("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{owner}/{repo}.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"] = owner
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    return os.getenv("MLFLOW_TRACKING_URI", "")
