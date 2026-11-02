"""Request and response models. The request is the API's data contract: it mirrors the
Pandera schema used in training (src/data.py), so bad inputs are rejected with HTTP 422."""

from typing import Literal

from pydantic import BaseModel, Field

Region = Literal[
    "Nuwara Eliya",
    "Haputale",
    "Dimbula",
    "Ella",
    "Uda Pussellawa",
    "Badulla",
    "Kandy",
    "Kegalle",
    "Ratnapura",
    "Galle",
    "Matara",
]


class PredictionRequest(BaseModel):
    region: Region = Field(description="Tea-growing region (elevation is looked up from it)")
    year: int = Field(
        ge=1990, le=2100, description="Calendar year (for logging; not a model input)"
    )
    month: int = Field(ge=1, le=12)
    rainfall_mm: float = Field(ge=0, le=1200, description="Total monthly rainfall")
    temperature_c: float = Field(
        ge=5, le=40, description="Mean monthly temperature (elevation-adjusted)"
    )
    humidity_percent: float = Field(ge=20, le=100)
    soil_moisture_percent: float = Field(ge=0, le=100)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "region": "Nuwara Eliya",
                    "year": 2024,
                    "month": 4,
                    "rainfall_mm": 165.0,
                    "temperature_c": 17.1,
                    "humidity_percent": 82.0,
                    "soil_moisture_percent": 70.0,
                }
            ]
        }
    }


class Contribution(BaseModel):
    term: str
    contribution: float


class PredictionResponse(BaseModel):
    request_id: str
    probability: float
    prediction: int
    label: str
    model_version: str


class ExplanationResponse(PredictionResponse):
    intercept: float
    contributions: list[Contribution]


class FeedbackRequest(BaseModel):
    request_id: str
    actual_outcome: int = Field(ge=0, le=1)
