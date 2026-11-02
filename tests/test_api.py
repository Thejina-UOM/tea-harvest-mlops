import importlib
import os

import pytest
from fastapi.testclient import TestClient

from src.features import MODEL_FEATURES
from src.model import build_pipeline, save
from tests.conftest import make_rows

PAYLOAD = {
    "region": "Nuwara Eliya",
    "year": 2024,
    "month": 4,
    "rainfall_mm": 165.0,
    "temperature_c": 17.1,
    "humidity_percent": 82.0,
    "soil_moisture_percent": 70.0,
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Train a tiny, fast EBM on synthetic data so the API can be tested without the real model.
    df = make_rows(300)
    model = build_pipeline(
        {"outer_bags": 1, "interactions": 0, "max_rounds": 200, "random_state": 0}
    ).fit(df[MODEL_FEATURES], df["y"])
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    save(model, path)
    os.environ["MODEL_PATH"] = str(path)

    import app.main

    importlib.reload(app.main)  # pick up MODEL_PATH
    with TestClient(app.main.app) as test_client:
        yield test_client


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy" and body["model_loaded"] is True


def test_predict(client):
    body = client.post("/predict", json=PAYLOAD).json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("ready", "not ready")
    assert body["request_id"]


def test_explain_returns_contributions(client):
    body = client.post("/explain", json=PAYLOAD).json()
    assert len(body["contributions"]) >= 6
    assert "intercept" in body


def test_contract_rejects_humidity_fraction(client):
    assert client.post("/predict", json={**PAYLOAD, "humidity_percent": 0.82}).status_code == 422


def test_contract_rejects_unknown_region(client):
    assert client.post("/predict", json={**PAYLOAD, "region": "Jaffna"}).status_code == 422


def test_feedback(client):
    request_id = client.post("/predict", json=PAYLOAD).json()["request_id"]
    response = client.post("/feedback", json={"request_id": request_id, "actual_outcome": 1})
    assert response.status_code == 200
