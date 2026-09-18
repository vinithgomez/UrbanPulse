"""
Tests for src/merge_weather_aqi.py's AQI-weather join.

merge_weather_aqi.py is a straight-line script (module-level code, not a
function), so these tests load it via importlib with pandas.read_csv and
DataFrame.to_csv monkeypatched -- this runs the ACTUAL merge line the
script executes (`aqi.merge(weather, on="datetime", how="left")`) against
small synthetic inputs, rather than reimplementing that logic separately
in the test (which would only prove the test's own copy of the logic is
correct, not the script's).

Correctness claims under test:
1. The left join must not duplicate or drop any AQI rows -- the output
   row count must equal the input AQI row count exactly.
2. A deliberately-introduced datetime present in the AQI data but absent
   from the weather data must produce a row with missing (NaN) weather
   columns -- not get silently matched to some other timestamp's weather,
   and not disappear from the output.
"""

import importlib.util
from pathlib import Path

import pandas as pd

SRC_PATH = Path(__file__).resolve().parent.parent / "src" / "merge_weather_aqi.py"


def _run_merge_script(aqi_df, weather_df, monkeypatch):
    """Execute merge_weather_aqi.py's real top-level code with its two
    pd.read_csv() calls intercepted to return the given synthetic
    dataframes, and its to_csv() write suppressed. Returns the executed
    module so its module-level variables (`merged`, `n_weather_missing`)
    can be inspected directly.
    """
    real_read_csv = pd.read_csv

    def fake_read_csv(path, *args, **kwargs):
        path_str = str(path)
        if "aqi_window_cleaned" in path_str:
            return aqi_df.copy()
        if "weather_hourly" in path_str:
            return weather_df.copy()
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda self, *a, **k: None)

    spec = importlib.util.spec_from_file_location("merge_weather_aqi_under_test", SRC_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # runs the script's real merge logic
    return module


def _make_synthetic_inputs():
    timestamps = pd.to_datetime([
        "2023-01-01 00:00:00",
        "2023-01-01 01:00:00",
        "2023-01-01 02:00:00",  # deliberately missing from weather below
        "2023-01-01 03:00:00",
    ])

    aqi_rows = []
    for station, base in [("A", 100.0), ("B", 200.0)]:
        for i, ts in enumerate(timestamps):
            aqi_rows.append({"station": station, "datetime": ts, "aqi_clean": base + 10 * i})
    aqi_df = pd.DataFrame(aqi_rows)

    # Weather covers 00:00, 01:00, 03:00 -- 02:00 is intentionally absent.
    weather_df = pd.DataFrame({
        "datetime": timestamps[[0, 1, 3]],
        "temperature_c": [20.0, 21.0, 23.0],
    })

    return aqi_df, weather_df, timestamps


def test_merge_preserves_row_count_with_no_duplication(monkeypatch):
    """8 AQI rows (2 stations x 4 hours) in -> 8 rows out. If the join
    silently fanned out (e.g. weather had duplicate datetimes) or dropped
    unmatched rows (e.g. an inner join instead of left), this would fail.
    """
    aqi_df, weather_df, _ = _make_synthetic_inputs()
    module = _run_merge_script(aqi_df, weather_df, monkeypatch)

    assert len(module.merged) == len(aqi_df)
    assert module.merged.duplicated(subset=["station", "datetime"]).sum() == 0


def test_datetime_mismatch_produces_missing_weather_not_wrong_match(monkeypatch):
    """The 02:00 timestamp exists in AQI but not weather. Every row at
    that timestamp must have NaN temperature_c (caught, not silently
    matched to a neighboring hour's weather), while every other row must
    have the exact correct weather value for its own timestamp.
    """
    aqi_df, weather_df, timestamps = _make_synthetic_inputs()
    module = _run_merge_script(aqi_df, weather_df, monkeypatch)
    merged = module.merged

    mismatched_ts = timestamps[2]
    mismatched_rows = merged[merged["datetime"] == mismatched_ts]
    assert len(mismatched_rows) == 2  # both stations' rows at that hour
    assert mismatched_rows["temperature_c"].isna().all()

    # The script's own diagnostic counter must have caught exactly these
    # 2 rows -- this is the actual detection mechanism the script relies
    # on to warn about a date-range mismatch, not just a side effect.
    assert module.n_weather_missing == 2

    # Spot-check correct (not merely non-null) matching for an unaffected timestamp.
    ok_rows = merged[merged["datetime"] == timestamps[0]]
    assert (ok_rows["temperature_c"] == 20.0).all()
    ok_rows_1 = merged[merged["datetime"] == timestamps[1]]
    assert (ok_rows_1["temperature_c"] == 21.0).all()
