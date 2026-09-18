"""
Phase 3 - Feature Engineering.

Builds per-station lag/rolling AQI features (no leakage across station
boundaries, no lookahead), keeps the existing weather features as-is, and
adds calendar features (holiday flag, stubble-burning season) per
UrbanPulse_Spec.md Phase 3.

Requires:
    data/processed/urbanpulse_master_with_traffic.csv

Output:
    data/processed/urbanpulse_features.csv
"""

import pandas as pd

IN_PATH = 'data/processed/urbanpulse_master_with_traffic.csv'
OUT_PATH = 'data/processed/urbanpulse_features.csv'

LAG_HOURS = [1, 6, 24]
ROLLING_WINDOWS = [6, 24]

# Major gazetted Indian national holidays + widely-observed Delhi festivals
# falling inside the analysis window (2022-07-01 to 2023-12-31), from
# publicly published Indian holiday calendars. Fixed-date holidays
# (Independence Day, Gandhi Jayanti, Christmas, Republic Day) are exact;
# lunar-calendar festival dates (Eid, Diwali, Holi, Janmashtami, etc.)
# follow the commonly published date but can shift by a day depending on
# regional moon sighting -- a disclosed approximation, not treated as
# precise. Diwali in particular is worth having: Delhi's fireworks-driven
# AQI spike around Diwali is well documented, so this flag can capture more
# than just a traffic effect.
HOLIDAYS = [
    '2022-07-10',  # Eid al-Adha (Bakrid)
    '2022-08-15',  # Independence Day
    '2022-08-31',  # Janmashtami
    '2022-10-05',  # Dussehra (Vijayadashami)
    '2022-10-24',  # Diwali
    '2022-10-26',  # Govardhan Puja
    '2022-11-08',  # Guru Nanak Jayanti
    '2022-12-25',  # Christmas
    '2023-01-26',  # Republic Day
    '2023-03-08',  # Holi
    '2023-03-30',  # Ram Navami
    '2023-04-04',  # Mahavir Jayanti
    '2023-04-07',  # Good Friday
    '2023-04-22',  # Eid al-Fitr
    '2023-05-05',  # Buddha Purnima
    '2023-06-29',  # Eid al-Adha (Bakrid)
    '2023-08-15',  # Independence Day
    '2023-08-30',  # Raksha Bandhan
    '2023-09-07',  # Janmashtami
    '2023-09-28',  # Milad-un-Nabi
    '2023-10-02',  # Gandhi Jayanti
    '2023-10-24',  # Dussehra (Vijayadashami)
    '2023-11-12',  # Diwali
    '2023-11-13',  # Govardhan Puja
    '2023-11-27',  # Guru Nanak Jayanti
    '2023-12-25',  # Christmas
]
HOLIDAY_DATES = set(pd.to_datetime(HOLIDAYS).date)

# Delhi stubble-burning season: satellite fire-count and CPCB reporting
# consistently show Punjab/Haryana crop-residue burning peaking mid-Oct to
# mid-Nov each year, driving some of Delhi's worst AQI spikes (flagged in
# UrbanPulse_Spec.md Phase 6 as a likely model-failure window). Modeled as a
# fixed calendar window per year, not an actual fire-count dataset.
STUBBLE_SEASON = [
    ('2022-10-15', '2022-11-15'),
    ('2023-10-15', '2023-11-15'),
]


def add_lag_and_rolling_features(df):
    """Per-station lag and rolling AQI features, computed with no leakage.

    Lag features use pandas GroupBy.shift(), which resets at each station's
    first row -- a station never picks up another station's tail values.

    Rolling features are shifted by 1 hour *before* the rolling window is
    applied, so a row's rolling mean/std covers strictly the preceding
    window (t-window .. t-1) and never includes the row's own aqi_clean
    value. GroupBy.transform() keeps this per-station as well.
    """
    df = df.sort_values(['station', 'datetime']).reset_index(drop=True)
    g = df.groupby('station')['aqi_clean']

    for lag in LAG_HOURS:
        df[f'aqi_lag_{lag}h'] = g.shift(lag)

    for window in ROLLING_WINDOWS:
        df[f'aqi_rolling_mean_{window}h'] = g.transform(
            lambda s, w=window: s.shift(1).rolling(window=w, min_periods=w).mean()
        )
        df[f'aqi_rolling_std_{window}h'] = g.transform(
            lambda s, w=window: s.shift(1).rolling(window=w, min_periods=w).std()
        )

    return df


def add_calendar_features(df):
    dates = df['datetime'].dt.date
    df['is_holiday'] = dates.isin(HOLIDAY_DATES)

    is_stubble = pd.Series(False, index=df.index)
    for start, end in STUBBLE_SEASON:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) + pd.Timedelta(hours=23)  # include the full end day
        is_stubble |= df['datetime'].between(start_ts, end_ts)
    df['is_stubble_season'] = is_stubble

    return df


def main():
    df = pd.read_csv(IN_PATH, parse_dates=['datetime'])
    n_before, cols_before = len(df), set(df.columns)

    df = add_lag_and_rolling_features(df)
    df = add_calendar_features(df)

    new_cols = [c for c in df.columns if c not in cols_before]

    df.to_csv(OUT_PATH, index=False)

    print(f"Rows: {len(df)} (unchanged from input: {n_before})")
    print(f"Columns added ({len(new_cols)}): {new_cols}")

    print("\nNaN count per new column:")
    for col in new_cols:
        n_nan = df[col].isna().sum()
        print(f"  {col:<24} {n_nan:>6} NaN  ({n_nan / len(df):.2%})")

    # These NaN counts are notably higher than the ~24-72 rows a bare
    # per-station warm-up period (1/6/24 hours x 3 stations) would produce.
    # The rest comes from aqi_clean's own pre-existing missingness (Phase 1:
    # DTU 2.0%, RK_Puram 4.3%, ITO 6.8% still NaN after cleaning) propagating
    # into every lag/rolling feature that touches an affected timestamp.
    # This compounds for the rolling stats because min_periods=window
    # requires *every* value in the window to be non-null -- a single
    # missing hour anywhere in the trailing 24h poisons aqi_rolling_*_24h
    # for that row. That's a deliberate strict choice (an "average" should
    # not silently be computed over fewer real hours than its label implies)
    # over the looser alternative of min_periods=1, which would produce a
    # non-NaN rolling stat even from a single real observation.
    # Decision: leave all of this as NaN rather than drop or interpolate,
    # matching this project's existing missing-data philosophy (see
    # README > Data decisions: flag, don't silently paper over). The
    # modeling step (Phase 4/5) can drop or impute per-model as needed --
    # dropping here would discard the same information for every downstream
    # model, including ones that don't need it (e.g. LightGBM handles NaN
    # natively).
    incomplete_rows = df[[c for c in new_cols if c.startswith('aqi_')]].isna().any(axis=1).sum()
    print(f"\nRows with at least one incomplete lag/rolling feature: "
          f"{incomplete_rows} ({incomplete_rows / len(df):.2%}) -- left as NaN, not dropped.")

    print("\nSanity check -- correlation of new AQI lag/rolling features "
          "with aqi_clean itself (lag_1h should correlate strongly):")
    lag_roll_cols = [c for c in new_cols if c.startswith('aqi_')]
    print(df[lag_roll_cols + ['aqi_clean']].corr()['aqi_clean'].drop('aqi_clean')
          .sort_values(ascending=False))

    print(f"\nSaved: {OUT_PATH}")


if __name__ == '__main__':
    main()
