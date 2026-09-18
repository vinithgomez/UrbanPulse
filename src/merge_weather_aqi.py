"""
Merge the cleaned, windowed AQI data (all 3 stations) with hourly Delhi
weather into a single Phase-1-complete dataset ready for Phase 2 (EDA/
correlation) and Phase 3 (feature engineering).

Requires:
    data/processed/aqi_window_cleaned.csv   (from clean_window.py)
    data/processed/weather_hourly.csv        (from fetch_weather.py)

Output:
    data/processed/urbanpulse_master.csv
"""

import pandas as pd

aqi = pd.read_csv("data/processed/aqi_window_cleaned.csv", parse_dates=["datetime"])
weather = pd.read_csv("data/processed/weather_hourly.csv", parse_dates=["datetime"])

# Weather is one city-wide series -> join on datetime alone, applied to every station
merged = aqi.merge(weather, on="datetime", how="left")

# Sanity checks
n_weather_missing = merged["temperature_c"].isna().sum()
print(f"Rows: {len(merged)}")
print(f"Stations: {merged['station'].unique().tolist()}")
print(f"Date range: {merged['datetime'].min()} to {merged['datetime'].max()}")
print(f"Rows with missing weather after join: {n_weather_missing}")

if n_weather_missing > 0:
    print("WARNING: some AQI timestamps had no matching weather row - "
          "check for date range mismatch between the two sources.")

out_path = "data/processed/urbanpulse_master.csv"
merged.to_csv(out_path, index=False)
print(f"\nSaved master dataset: {out_path}")
print(f"\nColumns: {list(merged.columns)}")
print(merged.head())
