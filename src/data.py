"""Validate the raw tea data against a contract, then build the training / evaluation /
"future" sets used by the DVC pipeline.

How I split the data for the MLOps pipeline (different from the notebook on purpose):
  * history = 2001-2021  -> randomly split (stratified) into train and eval
  * future  = 2022-2024  -> never trained on; the replayer sends these rows to the live
                            API to simulate "next years" of production traffic and drift.
"""

import pandas as pd
import pandera.pandas as pa

from src.config import ROOT, load_params
from src.features import MODEL_FEATURES, REGIONS, TARGET, VALID_RANGES

# Fields a client sends to the API (elevation is looked up from the region).
REQUEST_FIELDS = [
    "region",
    "year",
    "month",
    "rainfall_mm",
    "temperature_c",
    "humidity_percent",
    "soil_moisture_percent",
]
# Every column of the processed data except the label: the request fields + derived elevation.
FEATURES = REQUEST_FIELDS + ["elevation_m"]


def _range(col: str) -> pa.Check:
    lo, hi = VALID_RANGES[col]
    return pa.Check.in_range(lo, hi)


# The data contract: every column, its type and allowed values. Bad data fails HERE,
# before training, instead of silently producing a bad model.
FEATURE_SCHEMA = pa.DataFrameSchema(
    {
        "region": pa.Column(str, pa.Check.isin(list(REGIONS))),
        "year": pa.Column(int, _range("year")),
        "month": pa.Column(int, _range("month")),
        "rainfall_mm": pa.Column(float, _range("rainfall_mm")),
        "temperature_c": pa.Column(float, _range("temperature_c")),
        "humidity_percent": pa.Column(float, _range("humidity_percent")),
        "soil_moisture_percent": pa.Column(float, _range("soil_moisture_percent")),
        "elevation_m": pa.Column(int, _range("elevation_m")),
    },
    coerce=True,
)


def load_raw(path) -> pd.DataFrame:
    """Read the raw CSV, check the target and rename it to `y` (the name the pipeline uses)."""
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"target column '{TARGET}' not found in {path}")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError(f"target column '{TARGET}' must only contain 0 or 1")
    df = df.rename(columns={TARGET: "y"})
    return df.sort_values(["year", "month", "region"]).reset_index(drop=True)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check features against the contract. Raises pandera.errors.SchemaError on bad data."""
    df = df.copy()
    if "elevation_m" not in df.columns:
        df["elevation_m"] = df["region"].map(lambda r: REGIONS.get(r, {}).get("elevation_m"))
    checked = FEATURE_SCHEMA.validate(df[FEATURES])
    # Elevation must match the region table (catches a mislabelled region upstream).
    expected = checked["region"].map(lambda r: REGIONS[r]["elevation_m"])
    if not (expected == checked["elevation_m"]).all():
        raise ValueError("elevation_m does not match the region lookup table")
    return checked.assign(y=df["y"].astype(int).to_numpy())


def split_history(df: pd.DataFrame, history_until: int, val_frac: float, seed: int):
    """Oldest years = history (train + eval); newest years = 'future' traffic for the replayer."""
    history, future = df[df.year <= history_until], df[df.year > history_until]
    val = history.groupby("y", group_keys=False).sample(frac=val_frac, random_state=seed)
    train = history.drop(val.index)
    return history, train, val, future


def build_sets(raw: pd.DataFrame, production: pd.DataFrame, p: dict):
    history, train, val, future = split_history(
        raw, p["history_until_year"], p["val_frac"], p["seed"]
    )
    if len(production) == 0:
        # First model: train on history, judge it on the held-out validation rows.
        return train, val, future
    # Retraining: the newest labelled production rows become the evaluation set and
    # everything older (all history + older production rows) is used for training.
    production = production.reset_index(drop=True)
    cut = int(len(production) * (1 - p["holdout_frac"]))
    older, holdout = production.iloc[:cut], production.iloc[cut:]
    return pd.concat([history, older], ignore_index=True), holdout, future


def model_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """The six columns the model is trained on (same as the notebook)."""
    return df[MODEL_FEATURES]


def main() -> None:
    p = load_params()["data"]
    raw = validate(load_raw(ROOT / p["raw"]))
    production_path = ROOT / p["production"]
    production = pd.read_csv(production_path) if production_path.exists() else pd.DataFrame()
    if len(production):
        production = validate(production.sort_values("timestamp"))

    train, evaluation, future = build_sets(raw, production, p)
    out = ROOT / p["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    evaluation.to_csv(out / "eval.csv", index=False)
    future.to_csv(out / "future.csv", index=False)
    print(
        f"train={len(train)}  eval={len(evaluation)}  future={len(future)}  "
        f"production rows used={len(production)}"
    )
    print(f"ready rate: train={train.y.mean():.3f}  eval={evaluation.y.mean():.3f}")


if __name__ == "__main__":
    main()
