"""Download real hourly rainfall and soil moisture for Chennai.

Source: Open-Meteo historical archive (ERA5 reanalysis). Free, no API key.
https://archive-api.open-meteo.com/v1/archive

Emits data/raw/rainfall.csv with one row per hour and columns:
    timestamp, rainfall_mm, soil_moisture, source

ERA5 is a ~25 km reanalysis grid. It represents monsoon structure, multi-day
accumulation and storm shape well, but smooths convective peaks: for the
1-2 Dec 2015 Chennai event it reports 107 mm over two days where the
documented single-day record is 319.6 mm. This limitation is handled in
src/backtest/replay_2015.py, not hidden here.
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config

API = "https://archive-api.open-meteo.com/v1/archive"
TIMEOUT = 180


def fetch(start: str, end: str, retries: int = 4) -> dict:
    params = {
        "latitude": config.RAIN_LAT,
        "longitude": config.RAIN_LON,
        "start_date": start,
        "end_date": end,
        "hourly": "precipitation,soil_moisture_0_to_7cm",
        "timezone": config.RAIN_TZ,
    }
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(API, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            print(f"  attempt {attempt}/{retries} failed: {exc}")
            if attempt == retries:
                raise
            time.sleep(3 * attempt)
    raise RuntimeError("unreachable")


def main() -> None:
    print(f"Downloading ERA5 rainfall {config.RAIN_START} .. {config.RAIN_END}")
    payload = fetch(config.RAIN_START, config.RAIN_END)

    hourly = payload["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "rainfall_mm": pd.to_numeric(hourly["precipitation"], errors="coerce"),
        "soil_moisture": pd.to_numeric(
            hourly["soil_moisture_0_to_7cm"], errors="coerce"
        ),
    })
    df["source"] = "real:open-meteo/era5"

    # ERA5 occasionally returns nulls. Linear interpolation on an hourly grid
    # is the documented fill; anything still null becomes 0 for rainfall and
    # the series mean for soil moisture, and both are counted and reported.
    n_null_before = int(df["rainfall_mm"].isna().sum() + df["soil_moisture"].isna().sum())
    df["rainfall_mm"] = df["rainfall_mm"].interpolate(limit_direction="both").fillna(0.0)
    df["soil_moisture"] = (
        df["soil_moisture"].interpolate(limit_direction="both").ffill().bfill()
    )
    df = df.sort_values("timestamp").reset_index(drop=True)

    out = config.RAW / "rainfall.csv"
    df.to_csv(out, index=False)

    span_days = (df["timestamp"].max() - df["timestamp"].min()).days
    print(f"  rows            {len(df)}")
    print(f"  span            {df['timestamp'].min()} .. {df['timestamp'].max()}")
    print(f"  nulls filled    {n_null_before}")
    print(f"  mean annual mm  {df['rainfall_mm'].sum() / (span_days / 365.25):.1f}")
    print(f"  max hourly mm   {df['rainfall_mm'].max():.1f}")
    print(f"  wrote           {out}")


if __name__ == "__main__":
    main()
