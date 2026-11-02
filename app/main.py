"""Tea harvest readiness prediction service.

Every prediction and every piece of feedback is printed as one JSON line; Azure Container
Apps ships these lines to Log Analytics, where my monitoring job reads them for drift checks.
"""

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import joblib
import pandas as pd

# Send request metrics to Application Insights when running in Azure.
if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    from azure.monitor.opentelemetry import configure_azure_monitor

    configure_azure_monitor()

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from app.schemas import (  # noqa: E402
    Contribution,
    ExplanationResponse,
    FeedbackRequest,
    PredictionRequest,
    PredictionResponse,
)
from src.features import REGIONS, build_features  # noqa: E402

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")
THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
STATE: dict = {}

# A dedicated logger writes one complete JSON line per event.
events = logging.getLogger("events")
events.setLevel(logging.INFO)
events.propagate = False
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
events.addHandler(_handler)


def log_event(event: dict) -> None:
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    events.info(json.dumps(event))


@asynccontextmanager
async def lifespan(_: FastAPI):
    STATE["model"] = joblib.load(MODEL_PATH)
    yield
    STATE.clear()


app = FastAPI(
    title="Sri Lanka Tea Harvest Readiness API",
    description="Explainable Boosting Machine that predicts whether a tea region is harvest-ready.",
    version="1.0.0",
    lifespan=lifespan,
)
# The Streamlit front end may call this API from another origin.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _features(req: PredictionRequest) -> tuple[dict, pd.DataFrame]:
    raw = req.model_dump()
    raw["elevation_m"] = REGIONS[req.region]["elevation_m"]
    return raw, build_features(pd.DataFrame([raw]))


def _predict(req: PredictionRequest) -> tuple[dict, pd.DataFrame, PredictionResponse, float]:
    if "model" not in STATE:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    start = time.perf_counter()
    raw, X = _features(req)
    probability = float(STATE["model"].predict_proba(X)[0, 1])
    prediction = int(probability >= THRESHOLD)
    response = PredictionResponse(
        request_id=str(uuid.uuid4()),
        probability=round(probability, 5),
        prediction=prediction,
        label="ready" if prediction else "not ready",
        model_version=MODEL_VERSION,
    )
    return raw, X, response, (time.perf_counter() - start) * 1000


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "model_version": MODEL_VERSION, "model_loaded": "model" in STATE}


@app.get("/version")
def version() -> dict:
    return {"model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    raw, _, response, latency = _predict(req)
    log_event(
        {
            "event": "prediction",
            **response.model_dump(),
            "latency_ms": round(latency, 2),
            "features": raw,
        }
    )
    return response


@app.post("/explain", response_model=ExplanationResponse)
def explain(req: PredictionRequest) -> ExplanationResponse:
    """Prediction plus the EBM's exact per-term contributions (log-odds)."""
    raw, X, response, latency = _predict(req)
    pipeline = STATE["model"]
    ebm = pipeline.named_steps["model"]
    local = ebm.explain_local(pipeline.named_steps["impute"].transform(X)).data(0)
    contributions = [
        Contribution(term=name, contribution=round(float(score), 5))
        for name, score in sorted(
            zip(local["names"], local["scores"]), key=lambda t: abs(t[1]), reverse=True
        )
    ]
    log_event(
        {
            "event": "prediction",
            **response.model_dump(),
            "latency_ms": round(latency, 2),
            "features": raw,
        }
    )
    return ExplanationResponse(
        **response.model_dump(), intercept=float(ebm.intercept_[0]), contributions=contributions
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    log_event({"event": "feedback", **req.model_dump()})
    return {"status": "recorded"}
