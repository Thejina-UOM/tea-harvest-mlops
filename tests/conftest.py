"""Shared test fixtures. Tests use small synthetic data so they run without DVC or MLflow."""

import random

import pandas as pd
import pytest

from src.features import REGIONS


def make_rows(n: int, seed: int = 0) -> pd.DataFrame:
    """Synthetic but realistic rows; readiness follows a simple weather rule."""
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        region = rnd.choice(list(REGIONS))
        rain = round(rnd.uniform(5, 600), 2)
        temp = round(rnd.uniform(13, 30), 2)
        hum = round(rnd.uniform(62, 93), 2)
        rows.append(
            {
                "region": region,
                "year": 2001 + i % 24,
                "month": 1 + i % 12,
                "rainfall_mm": rain,
                "temperature_c": temp,
                "humidity_percent": hum,
                "soil_moisture_percent": float(rnd.randint(47, 100)),
                "elevation_m": REGIONS[region]["elevation_m"],
                "y": int(rain > 80 and hum < 90 and temp > 15),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def rows() -> pd.DataFrame:
    return make_rows(240)
