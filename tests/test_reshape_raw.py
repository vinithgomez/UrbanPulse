"""
Tests for src/reshape_raw.py's month-block CSV parser.

Correctness claim under test: the raw CPCB month-block files contain a
calendar-day column (1-31) applied uniformly to every month header, so a
31-day month header followed by day "30" or "31" can point at a date that
doesn't exist for a shorter month (e.g. "February-2023,...,30,...").
parse_station_csv() must skip that row rather than either crashing
(datetime() raises ValueError for Feb 30) or silently normalizing it into
some other date (e.g. rolling over into March), which would silently
corrupt the AQI time series with a wrong timestamp.
"""

import textwrap

from reshape_raw import parse_station_csv

SAMPLE_CSV = textwrap.dedent("""\
    Year,2023
    February-2023,00:00:00,01:00:00,02:00:00
    1,100.0,110.0,120.0
    30,900.0,910.0,920.0
    March-2023,00:00:00,01:00:00,02:00:00
    1,50.0,60.0,70.0
""")


def test_invalid_date_row_is_skipped_not_crashed(tmp_path):
    """February 30 doesn't exist; parsing must not raise and must not
    emit any row for it."""
    csv_path = tmp_path / "station_raw.csv"
    csv_path.write_text(SAMPLE_CSV, encoding="utf-8")

    # The key assertion is implicit: this call must not raise ValueError.
    df = parse_station_csv(str(csv_path), "TEST")

    # No row anywhere in the output should carry the invalid day's values
    # (900/910/920) or land on Feb 30 / a rolled-over date derived from it.
    assert not (df["aqi"] == 900.0).any()
    assert not ((df["datetime"].dt.month == 2) & (df["datetime"].dt.day == 30)).any()
    # Guard against silent rollover into March 2 (Feb 30 + nothing / normalization artifact)
    assert not ((df["datetime"].dt.month == 3) & (df["datetime"].dt.day == 2)).any()


def test_valid_rows_around_the_invalid_one_still_parse_correctly(tmp_path):
    """The invalid Feb 30 row sits between two valid rows (Feb 1 and
    Mar 1); both must still be parsed with correct dates and values,
    confirming the parser recovers cleanly rather than derailing the
    rest of the file."""
    csv_path = tmp_path / "station_raw.csv"
    csv_path.write_text(SAMPLE_CSV, encoding="utf-8")

    df = parse_station_csv(str(csv_path), "TEST").sort_values("datetime").reset_index(drop=True)

    # Feb 1, hour 0-2: values 100/110/120
    feb1 = df[(df["datetime"].dt.month == 2) & (df["datetime"].dt.day == 1)].sort_values("datetime")
    assert list(feb1["aqi"]) == [100.0, 110.0, 120.0]

    # Mar 1, hour 0-2: values 50/60/70
    mar1 = df[(df["datetime"].dt.month == 3) & (df["datetime"].dt.day == 1)].sort_values("datetime")
    assert list(mar1["aqi"]) == [50.0, 60.0, 70.0]

    # Total rows = 3 (Feb 1) + 3 (Mar 1); the invalid Feb 30 row contributes zero.
    assert len(df) == 6
    assert (df["station"] == "TEST").all()
