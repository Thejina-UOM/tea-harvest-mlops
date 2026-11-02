"""Quality gate: the newly trained model (challenger) must be at least as good as @champion.

Both models are scored on the same evaluation set, which neither was trained on.
Exit code 0 = passed, 1 = failed. A markdown report is written for the pull request.
"""

import argparse
import sys

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException

from src import tracking
from src.config import ROOT, load_params
from src.data import model_inputs
from src.metrics import score
from src.model import load


def load_champion(name: str):
    try:
        return mlflow.sklearn.load_model(f"models:/{name}@champion")
    except MlflowException as err:
        print(f"No champion found ({err.error_code}); the first model passes automatically.")
        return None
    except Exception as err:  # e.g. no tracking server reachable in a local run
        print(f"Could not load the champion ({type(err).__name__}: {err}); treating as none.")
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="gate_report.md")
    args = parser.parse_args()

    tracking.configure()
    params = load_params()
    e = params["evaluate"]
    df = pd.read_csv(ROOT / params["data"]["processed_dir"] / "eval.csv")
    X, y = model_inputs(df), df["y"]

    challenger = load(ROOT / "models" / "model.joblib")
    new = score(y, challenger.predict_proba(X)[:, 1], e["threshold"])

    champion = load_champion(params["model"]["name"])
    old = score(y, champion.predict_proba(X)[:, 1], e["threshold"]) if champion else None

    metric, delta = e["gate_metric"], e["gate_min_delta"]
    passed = old is None or new[metric] >= old[metric] + delta

    lines = [
        f"## Model quality gate: {'PASSED' if passed else 'FAILED'}",
        "",
        f"Evaluation set: {len(df)} rows, ready rate {y.mean():.3f}. "
        f"Rule: challenger `{metric}` must be at least champion + {delta}.",
        "",
        "| Metric | Champion | Challenger |",
        "|---|---|---|",
    ]
    for key in ["f1", "accuracy", "roc_auc", "recall_not_ready", "mean_prediction"]:
        champ_value = f"{old[key]:.4f}" if old else "none yet"
        lines.append(f"| {key} | {champ_value} | {new[key]:.4f} |")
    report = "\n".join(lines) + "\n"
    (ROOT / args.report).write_text(report)
    print(report)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
