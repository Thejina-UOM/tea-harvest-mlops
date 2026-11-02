import pandas as pd
import pandera.errors
import pytest

from src.data import FEATURES, build_sets, validate
from src.features import MODEL_FEATURES


def test_contract_accepts_valid_rows(rows):
    assert len(validate(rows)) == len(rows)


def test_contract_rejects_humidity_sent_as_fraction(rows):
    rows.loc[0, "humidity_percent"] = 0.82  # the "sensor bug" from my incident drill
    with pytest.raises(pandera.errors.SchemaError):
        validate(rows)


def test_contract_rejects_unknown_region(rows):
    rows.loc[0, "region"] = "Jaffna"
    with pytest.raises(pandera.errors.SchemaError):
        validate(rows)


def test_contract_rejects_wrong_elevation_for_region(rows):
    rows.loc[0, "elevation_m"] = rows.loc[0, "elevation_m"] + 100
    with pytest.raises(ValueError):
        validate(rows)


def test_target_never_used_as_feature():
    assert "y" not in FEATURES and "harvest_ready" not in MODEL_FEATURES


def test_first_model_uses_history_only(rows):
    p = {"history_until_year": 2018, "val_frac": 0.2, "holdout_frac": 0.3, "seed": 1}
    train, evaluation, future = build_sets(rows, pd.DataFrame(), p)
    assert (future.year > 2018).all()
    assert (train.year <= 2018).all() and (evaluation.year <= 2018).all()
    assert set(train.index).isdisjoint(evaluation.index)


def test_retraining_evaluates_on_newest_production_rows(rows):
    p = {"history_until_year": 2018, "val_frac": 0.2, "holdout_frac": 0.3, "seed": 1}
    production = validate(rows.head(100))
    train, evaluation, _ = build_sets(rows, production, p)
    assert len(evaluation) == 30  # newest 30% of production rows
    history = (rows.year <= 2018).sum()
    assert len(train) == history + 70  # all history plus the older production rows
