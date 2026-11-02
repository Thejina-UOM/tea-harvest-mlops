"""Build, save and load the production model (imputer + Explainable Boosting Machine).

I save the whole scikit-learn Pipeline with joblib, so the API and the Streamlit app
apply exactly the same preprocessing that was used in training.
"""

from pathlib import Path

import joblib
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_pipeline(ebm_params: dict) -> Pipeline:
    """Median imputer (for blank inputs at serving time) followed by the EBM."""
    imputer = SimpleImputer(strategy="median").set_output(transform="pandas")
    return Pipeline([("impute", imputer), ("model", ExplainableBoostingClassifier(**ebm_params))])


def save(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)


def load(path) -> Pipeline:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Run `make repro` or `dvc pull`.")
    return joblib.load(path)
