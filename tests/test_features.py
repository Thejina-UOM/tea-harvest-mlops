import numpy as np
import pandas as pd
import pytest

from src.features import (
    MODEL_FEATURES,
    REGIONS,
    build_features,
    out_of_training_range,
    season_of,
    validate_inputs,
)


def test_build_features_fills_elevation_from_region():
    X = build_features(
        pd.DataFrame(
            [
                {
                    "region": "Galle",
                    "month": 4,
                    "rainfall_mm": 200,
                    "temperature_c": 27,
                    "humidity_percent": 85,
                    "soil_moisture_percent": 70,
                }
            ]
        )
    )
    assert list(X.columns) == MODEL_FEATURES
    assert X.loc[0, "elevation_m"] == REGIONS["Galle"]["elevation_m"]


def test_build_features_turns_bad_text_into_nan():
    X = build_features(
        pd.DataFrame(
            [
                {
                    "elevation_m": 500,
                    "month": 3,
                    "rainfall_mm": "abc",
                    "temperature_c": 25,
                    "humidity_percent": 80,
                    "soil_moisture_percent": 60,
                }
            ]
        )
    )
    assert np.isnan(X.loc[0, "rainfall_mm"])


def test_build_features_rejects_missing_columns():
    with pytest.raises(ValueError):
        build_features(pd.DataFrame([{"month": 3}]))


def test_seasons():
    assert season_of(1) == "Maha" and season_of(10) == "Maha" and season_of(7) == "Yala"


def test_validate_inputs_flags_out_of_range_values():
    assert validate_inputs({"humidity_percent": 0.82}) != []
    assert validate_inputs({"humidity_percent": 82, "month": 5}) == []


def test_out_of_training_range():
    assert out_of_training_range({"rainfall_mm": 900}) == ["Monthly rainfall (mm)"]
