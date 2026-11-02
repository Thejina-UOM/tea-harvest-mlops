"""Simulate production traffic against the live API, then send the real outcomes as feedback.

Modes:
  normal      random rows from the training period (2001-2021): what "no drift" looks like
  future      the newest rows (2022-2024), in date order: real-world drift over time
  sensor-bug  like future, but an upstream weather feed now sends humidity as a fraction
              (0.82 instead of 82 %) - the API's data contract must reject these (HTTP 422)
  climate     future rows with +2.5 degC and -35 % rainfall: a simulated drought / warming
              scenario that should trigger the drift alarm

Example:
  uv run python replayer/replay.py https://<app-url> --mode future --rows 600
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REQUEST_FIELDS = [
    "region",
    "year",
    "month",
    "rainfall_mm",
    "temperature_c",
    "humidity_percent",
    "soil_moisture_percent",
]


def load_rows(mode: str, rows: int, start: int) -> pd.DataFrame:
    processed = ROOT / "data" / "processed"
    if mode == "normal":
        df = pd.read_csv(processed / "train.csv")
        df = df.sample(min(rows, len(df)), random_state=7)
    else:
        df = pd.read_csv(processed / "future.csv").iloc[start : start + rows]
    if mode == "sensor-bug":
        df = df.assign(humidity_percent=(df["humidity_percent"] / 100).round(3))
    if mode == "climate":
        df = df.assign(
            temperature_c=(df["temperature_c"] + 2.5).round(2),
            rainfall_mm=(df["rainfall_mm"] * 0.65).round(2),
        )
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="base URL of the API, without a trailing slash")
    parser.add_argument(
        "--mode", choices=["normal", "future", "sensor-bug", "climate"], default="future"
    )
    parser.add_argument("--rows", type=int, default=600)
    parser.add_argument("--start", type=int, default=0, help="first future row to send")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--feedback-delay", type=float, default=20)
    parser.add_argument("--no-feedback", action="store_true")
    args = parser.parse_args()

    df = load_rows(args.mode, args.rows, args.start)
    outcomes = df["y"].astype(int).tolist()
    payloads = df[REQUEST_FIELDS].to_dict(orient="records")
    client = httpx.Client(base_url=args.url.rstrip("/"), timeout=30)

    def send(i: int):
        try:
            response = client.post("/predict", json=payloads[i])
        except httpx.HTTPError as err:
            return i, 0, str(err)
        return i, response.status_code, response.json() if response.status_code == 200 else None

    print(f"Sending {len(payloads)} '{args.mode}' requests to {args.url} ...")
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(send, range(len(payloads))))
    ok = [(i, body) for i, status, body in results if status == 200]
    rejected = sum(1 for _, status, _ in results if status == 422)
    failed = len(results) - len(ok) - rejected
    print(f"  {len(ok)} predicted, {rejected} rejected by the data contract (422), {failed} errors")
    if not ok:
        sys.exit(1 if failed else 0)

    if args.no_feedback:
        return
    print(f"Waiting {args.feedback_delay:.0f}s before sending the real outcomes ...")
    time.sleep(args.feedback_delay)

    def feedback(item):
        i, body = item
        client.post(
            "/feedback", json={"request_id": body["request_id"], "actual_outcome": outcomes[i]}
        )

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(feedback, ok))
    correct = sum(int(body["prediction"] == outcomes[i]) for i, body in ok)
    print(f"  feedback sent for {len(ok)} requests; live accuracy = {correct / len(ok):.3f}")


if __name__ == "__main__":
    main()
