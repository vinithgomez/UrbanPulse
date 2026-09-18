"""
Tests for src/feature_engineering.py's leakage-prevention logic.

This is the highest-value test in the suite: the whole point of
add_lag_and_rolling_features() is that a row's lag/rolling-window features
must be computable from a real forecaster's information at that hour --
i.e. strictly from *past* observations, never the row's own value, and
never another station's data. Every downstream model (Phases 4/5) assumes
this holds. Until now it was verified only by eyeballing the first two
rows of each station's output (see README > Data decisions); these tests
make that check automatic and push it harder with a deliberate outlier.
"""

import numpy as np
import pandas as pd
import pytest

from feature_engineering import add_lag_and_rolling_features


def _hourly_index(n, start="2023-01-01"):
    return pd.date_range(start, periods=n, freq="h")


def test_rolling_stats_exclude_the_row_s_own_value():
    """Construct a constant series for one station, except a single
    extreme outlier at row 30. If the rolling window leaked the row's
    own value, row 30's own aqi_rolling_mean_24h/std_24h would be wildly
    skewed by its own 10000.0. Since the 24h window preceding row 30
    (rows 6-29) is all untouched constant values, row 30's own rolling
    stats must show no trace of the outlier: mean exactly 10.0, std
    exactly 0.0.
    """
    n = 40
    values = np.full(n, 10.0)
    OUTLIER_IDX = 30
    values[OUTLIER_IDX] = 10_000.0

    df = pd.DataFrame({
        "station": "A",
        "datetime": _hourly_index(n),
        "aqi_clean": values,
    })

    out = add_lag_and_rolling_features(df)
    row30 = out.iloc[OUTLIER_IDX]

    assert row30["aqi_rolling_mean_24h"] == 10.0
    assert row30["aqi_rolling_std_24h"] == 0.0
    # Same logic for the lag feature: row 30's own lag_1h must be the
    # *previous* row's value (10.0), not its own (10000.0).
    assert row30["aqi_lag_1h"] == 10.0


def test_rolling_stats_do_pick_up_the_outlier_on_later_rows():
    """The flip side of the previous test: once the outlier row has
    become *past* data (for rows after it), it correctly appears in
    their rolling windows. This confirms the exclusion above is a real
    t-1-cutoff effect, not a bug that zeroes out the outlier everywhere.
    """
    n = 40
    values = np.full(n, 10.0)
    OUTLIER_IDX = 30
    values[OUTLIER_IDX] = 10_000.0

    df = pd.DataFrame({
        "station": "A",
        "datetime": _hourly_index(n),
        "aqi_clean": values,
    })

    out = add_lag_and_rolling_features(df)
    row31 = out.iloc[OUTLIER_IDX + 1]

    # Row 31's 24h window covers rows 7-30 inclusive (23 values of 10.0 + the outlier).
    expected_mean = (23 * 10.0 + 10_000.0) / 24
    assert row31["aqi_rolling_mean_24h"] == pytest.approx(expected_mean)
    assert row31["aqi_rolling_std_24h"] > 0  # no longer constant
    assert row31["aqi_lag_1h"] == 10_000.0  # directly picks up the outlier as t-1


def test_lag_features_do_not_leak_across_a_station_boundary():
    """Two stations concatenated, sorted by (station, datetime) exactly
    as add_lag_and_rolling_features() does internally. Station A's last
    row is a distinctive extreme value; station B's first rows must not
    pick it up as a lag/rolling input -- each station's history must
    reset independently.
    """
    n_a, n_b = 10, 10
    STATION_A_TAIL_VALUE = -99999.0

    a_values = np.full(n_a, 50.0)
    a_values[-1] = STATION_A_TAIL_VALUE  # A's very last (most recent) hour

    b_values = np.full(n_b, 20.0)

    df_a = pd.DataFrame({
        "station": "A",
        "datetime": _hourly_index(n_a, start="2023-01-01"),
        "aqi_clean": a_values,
    })
    # Station B's clock starts wherever its own data starts -- deliberately
    # made to immediately follow A's last timestamp, the worst case for
    # accidental leakage via a naive (non-grouped) shift/rolling.
    df_b = pd.DataFrame({
        "station": "B",
        "datetime": _hourly_index(n_b, start=df_a["datetime"].iloc[-1] + pd.Timedelta(hours=1)),
        "aqi_clean": b_values,
    })

    combined = pd.concat([df_a, df_b], ignore_index=True)
    out = add_lag_and_rolling_features(combined)

    b_rows = out[out["station"] == "B"].reset_index(drop=True)

    # Station B's first row has no prior history *within its own station*,
    # so every lag/rolling feature must be NaN -- never A's tail value.
    first_b = b_rows.iloc[0]
    for col in ["aqi_lag_1h", "aqi_lag_6h", "aqi_lag_24h",
                "aqi_rolling_mean_6h", "aqi_rolling_mean_24h"]:
        assert pd.isna(first_b[col]), f"{col} should be NaN at B's first row, got {first_b[col]}"

    # None of station B's rows should ever contain A's distinctive tail value,
    # in any lag/rolling column, at any point.
    lag_roll_cols = [c for c in out.columns if c.startswith("aqi_lag") or c.startswith("aqi_rolling")]
    for col in lag_roll_cols:
        assert not (b_rows[col] == STATION_A_TAIL_VALUE).any(), (
            f"station B's {col} contains station A's tail value -- leakage across station boundary"
        )
