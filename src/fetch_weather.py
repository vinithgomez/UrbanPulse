"""
Fetch hourly historical weather for Delhi from the Open-Meteo Archive API.

No API key required. Run this locally (not inside Claude's sandboxed
container, which has a restricted network allowlist).

Usage:
    python fetch_weather.py

Output:
    data/raw/weather_raw.json
    data/processed/weather_hourly.csv
"""

import json
import requests
import pandas as pd

# Representative Delhi coordinate (approx. ITO / central Delhi).
# AQI's weather sensitivity is driven by broad meteorology (wind, humidity,
# temperature inversion) rather than hyper-local conditions, so a single
# city-level weather series applied across all three AQI stations is a
# reasonable, defensible simplification -- same logic as the traffic proxy.
# Stated explicitly here so it's easy to cite in the report's methodology
# section.
LATITUDE = 28.6289
LONGITUDE = 77.2405

START_DATE = "2022-07-01"
END_DATE = "2023-12-31"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "surface_pressure",
]


def fetch():
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Asia/Kolkata",
    }

    print(f"Requesting Open-Meteo archive for {START_DATE} to {END_DATE}...")
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    with open("data/raw/weather_raw.json", "w") as f:
        json.dump(data, f)
    print("Saved raw response: data/raw/weather_raw.json")

    return data


def to_dataframe(data: dict) -> pd.DataFrame:
    hourly = data["hourly"]
    df = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "temperature_c": hourly["temperature_2m"],
        "humidity_pct": hourly["relative_humidity_2m"],
        "wind_speed_kmh": hourly["wind_speed_10m"],
        "precipitation_mm": hourly["precipitation"],
        "pressure_hpa": hourly["surface_pressure"],
    })
    return df


def main():
    data = fetch()
    df = to_dataframe(data)

    print(f"\nRows: {len(df)}")
    print(f"Range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Missing values per column:\n{df.isna().sum()}")

    out_path = "data/processed/weather_hourly.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
