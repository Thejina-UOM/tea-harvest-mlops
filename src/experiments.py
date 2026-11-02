"""Compare several model families on the same data, every run tracked in MLflow (DagsHub).

This is exploration, separate from the production pipeline in dvc.yaml. The full,
tuned comparison (with hyperparameter searches) is in notebooks/Tea_Harvest_Readiness_EBM.ipynb;
here I log the tuned configurations so the comparison is also visible on DagsHub.
"""

import time

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from src import tracking
from src.config import ROOT, load_params
from src.data import model_inputs
from src.metrics import score
from src.model import build_pipeline


def cyclical(X: pd.DataFrame) -> pd.DataFrame:
    """Linear models need month on a circle (Dec next to Jan) and log rainfall."""
    X = X.copy()
    X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
    X["rainfall_mm"] = np.log1p(X["rainfall_mm"].clip(lower=0))
    return X.drop(columns="month")


def build_models(params: dict) -> dict:
    seed = params["data"]["seed"]
    return {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median").set_output(transform="pandas"),
            FunctionTransformer(cyclical),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=3000),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_depth=14, min_samples_leaf=2, random_state=seed, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=seed,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            random_state=seed,
            verbose=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_seed=seed,
            verbose=0,
            allow_writing_files=False,
        ),
        "ebm": build_pipeline(params["model"]["ebm"]),
    }


def main() -> None:
    tracking.configure()
    params = load_params()
    data_dir = ROOT / params["data"]["processed_dir"]
    train, val = pd.read_csv(data_dir / "train.csv"), pd.read_csv(data_dir / "eval.csv")
    X_tr, y_tr = model_inputs(train), train["y"]
    X_val, y_val = model_inputs(val), val["y"]

    mlflow.set_experiment(params["model"]["experiment"] + "-experiments")
    results = []
    for name, model in build_models(params).items():
        with mlflow.start_run(run_name=name):
            start = time.perf_counter()
            model.fit(X_tr, y_tr)
            fit_seconds = time.perf_counter() - start
            metrics = score(y_val, model.predict_proba(X_val)[:, 1])
            mlflow.log_param("model_family", name)
            mlflow.log_metrics({**metrics, "fit_seconds": fit_seconds})
        results.append({"model": name, **metrics, "fit_s": fit_seconds})

    table = pd.DataFrame(results).set_index("model").round(4)
    print(table[["f1", "accuracy", "roc_auc", "fit_s"]].sort_values("f1", ascending=False))


if __name__ == "__main__":
    main()
