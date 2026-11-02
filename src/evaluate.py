"""Score the trained model on the evaluation set, plot its EBM term importances,
and log both to MLflow (on the same run that trained the model)."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from src import tracking  # noqa: E402
from src.config import ROOT, load_params  # noqa: E402
from src.data import model_inputs  # noqa: E402
from src.metrics import score  # noqa: E402
from src.model import load  # noqa: E402


def main() -> None:
    tracking.configure()
    params = load_params()
    df = pd.read_csv(ROOT / params["data"]["processed_dir"] / "eval.csv")
    X, y = model_inputs(df), df["y"]

    model = load(ROOT / "models" / "model.joblib")
    metrics = score(y, model.predict_proba(X)[:, 1], params["evaluate"]["threshold"])
    metrics["eval_rows"] = len(df)
    (ROOT / "eval_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))

    # EBM is a glass-box model: its term importances are exact, no SHAP approximation needed.
    ebm = model.named_steps["model"]
    imp = pd.Series(ebm.term_importances(), index=ebm.term_names_).sort_values().tail(12)
    plot_path = ROOT / "plots" / "feature_importance.png"
    plot_path.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(
        imp.index, imp.values, color=["#D4A017" if " & " in t else "#2D6A4F" for t in imp.index]
    )
    ax.set_title("EBM term importance (mean |log-odds contribution|)")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    run_info = ROOT / "run_info.json"
    if run_info.exists():
        run_id = json.loads(run_info.read_text())["run_id"]
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({f"eval_{name}": value for name, value in metrics.items()})
            mlflow.log_artifact(str(plot_path))


if __name__ == "__main__":
    main()
