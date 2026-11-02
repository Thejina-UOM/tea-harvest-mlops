"""Domain knowledge and feature engineering shared by the notebook, the DVC pipeline,
the FastAPI service and the Streamlit app.

I keep this in ONE place so that the model always sees exactly the same features in
training and in production (training/serving skew is a classic MLOps failure).

All transformations here are *stateless* (they do not learn anything from the data),
so applying them before the train/test split cannot leak information. Anything that
learns from data (imputation, scaling) lives inside the scikit-learn Pipeline instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "harvest_ready"

# ---------------------------------------------------------------------------
# Domain knowledge: the 11 tea-growing regions in the dataset.
# Elevation and zone come from the dataset itself (they are fixed per region).
# Latitude/longitude are approximate town centres, used only for the map in the app.
# ---------------------------------------------------------------------------
REGIONS: dict[str, dict] = {
    "Nuwara Eliya": {"elevation_m": 1868, "zone": "high-grown", "lat": 6.9497, "lon": 80.7891},
    "Haputale": {"elevation_m": 1431, "zone": "high-grown", "lat": 6.7656, "lon": 80.9580},
    "Dimbula": {"elevation_m": 1271, "zone": "high-grown", "lat": 6.9370, "lon": 80.6590},
    "Ella": {"elevation_m": 1041, "zone": "mid-grown", "lat": 6.8667, "lon": 81.0466},
    "Uda Pussellawa": {"elevation_m": 900, "zone": "mid-grown", "lat": 6.9500, "lon": 80.8600},
    "Badulla": {"elevation_m": 678, "zone": "mid-grown", "lat": 6.9934, "lon": 81.0550},
    "Kandy": {"elevation_m": 500, "zone": "mid-grown", "lat": 7.2906, "lon": 80.6337},
    "Kegalle": {"elevation_m": 116, "zone": "low-grown", "lat": 7.2513, "lon": 80.3464},
    "Ratnapura": {"elevation_m": 18, "zone": "low-grown", "lat": 6.6828, "lon": 80.3992},
    "Galle": {"elevation_m": 14, "zone": "low-grown", "lat": 6.0535, "lon": 80.2210},
    "Matara": {"elevation_m": 11, "zone": "low-grown", "lat": 5.9549, "lon": 80.5550},
}

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Maha = North-East monsoon cultivation season (Oct-Apr); Yala = South-West monsoon (May-Sep).
MAHA_MONTHS = {10, 11, 12, 1, 2, 3, 4}

# NDVI in this file is a lookup by zone and season (only 6 distinct values), so I treat it
# as derived information and do NOT give it to the model. Kept here for the app display.
NDVI_LOOKUP = {
    ("high-grown", "Maha"): 0.64,
    ("high-grown", "Yala"): 0.71,
    ("mid-grown", "Maha"): 0.57,
    ("mid-grown", "Yala"): 0.63,
    ("low-grown", "Maha"): 0.48,
    ("low-grown", "Yala"): 0.54,
}

# The six inputs the final model uses (see notebook section 5 for why I dropped the rest).
MODEL_FEATURES = [
    "elevation_m",
    "month",
    "rainfall_mm",
    "temperature_c",
    "humidity_percent",
    "soil_moisture_percent",
]

# Human-friendly labels for plots and the app.
FEATURE_LABELS = {
    "elevation_m": "Elevation (m)",
    "month": "Month",
    "rainfall_mm": "Monthly rainfall (mm)",
    "temperature_c": "Mean temperature (°C)",
    "humidity_percent": "Relative humidity (%)",
    "soil_moisture_percent": "Soil moisture (%)",
}

# Physically sensible input ranges. They are slightly wider than what the dataset
# contains, so a real but unusual month is accepted, while typing errors are rejected.
VALID_RANGES = {
    "elevation_m": (0, 2600),
    "month": (1, 12),
    "year": (1990, 2100),
    "rainfall_mm": (0.0, 1200.0),
    "temperature_c": (5.0, 40.0),
    "humidity_percent": (20.0, 100.0),
    "soil_moisture_percent": (0.0, 100.0),
}

# Range actually seen in training. Inputs outside this are allowed but flagged in the app
# because the model is extrapolating there.
TRAINING_RANGES = {
    "elevation_m": (11, 1868),
    "month": (1, 12),
    "rainfall_mm": (1.1, 738.4),
    "temperature_c": (12.8, 30.2),
    "humidity_percent": (61.5, 93.5),
    "soil_moisture_percent": (47.0, 100.0),
}

# Columns that are present in the raw file but deliberately not used as model inputs.
DROPPED_COLUMNS = {
    "year": "time index only - using it would tie the model to past calendar years",
    "region": "1-to-1 with elevation_m; elevation generalises to an unseen estate",
    "zone": "derived from elevation_m (3 elevation bands)",
    "temperature_c_raw": "unadjusted grid value; temperature_c already corrects it for elevation",
    "ndvi_index": "lookup table by zone and season (only 6 distinct values)",
    "season": "derived from month (Maha = Oct-Apr, Yala = May-Sep)",
}


def season_of(month: int) -> str:
    """Return the Sri Lankan cultivation season for a calendar month."""
    return "Maha" if int(month) in MAHA_MONTHS else "Yala"


def region_info(region: str) -> dict:
    """Look up elevation/zone for a region name, with a helpful error for typos."""
    if region not in REGIONS:
        raise ValueError(f"Unknown region '{region}'. Choose one of: {', '.join(REGIONS)}")
    return REGIONS[region]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw rows (dataset rows, API requests or app inputs) into model inputs.

    - If `elevation_m` is missing but `region` is present, I fill it from the region table.
    - Values are coerced to numbers; anything unparsable becomes NaN so that the
      pipeline's SimpleImputer can deal with it instead of the app crashing.
    """
    if df is None or len(df) == 0:
        raise ValueError("No rows were given to build_features().")
    out = df.copy()
    if "elevation_m" not in out.columns and "region" in out.columns:
        out["elevation_m"] = out["region"].map(lambda r: REGIONS.get(r, {}).get("elevation_m"))
    missing = [c for c in MODEL_FEATURES if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")
    X = out[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce").astype(float)
    return X


def validate_inputs(row: dict) -> list[str]:
    """Return a list of human-readable problems with one input row (empty list = OK)."""
    problems = []
    for col, (lo, hi) in VALID_RANGES.items():
        if col not in row or row[col] is None:
            continue
        try:
            value = float(row[col])
        except (TypeError, ValueError):
            problems.append(f"{FEATURE_LABELS.get(col, col)} must be a number.")
            continue
        if np.isnan(value):
            continue
        if not lo <= value <= hi:
            problems.append(f"{FEATURE_LABELS.get(col, col)} = {value} is outside {lo}-{hi}.")
    return problems


def out_of_training_range(row: dict) -> list[str]:
    """Features whose value lies outside what the model saw during training."""
    flagged = []
    for col, (lo, hi) in TRAINING_RANGES.items():
        try:
            value = float(row.get(col))
        except (TypeError, ValueError):
            continue
        if not lo <= value <= hi:
            flagged.append(FEATURE_LABELS.get(col, col))
    return flagged


def add_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add readable month name and season columns (for EDA and the app only)."""
    out = df.copy()
    out["month_name"] = out["month"].map(lambda m: MONTH_NAMES[int(m) - 1])
    out["season"] = out["month"].map(season_of)
    return out
