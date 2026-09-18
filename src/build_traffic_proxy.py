"""
Construct a time-of-day/day-of-week traffic proxy for Delhi.

This is a stated, deliberate fallback (see README > Open decisions):
no accessible real historical traffic congestion data for Delhi was found
(TomTom/Google historical APIs are enterprise-gated; no confirmed open
Delhi Traffic Police historical dataset). Rather than a flat binary
rush-hour flag, this builds a continuous 0-1 "congestion likelihood"
score from well-documented Delhi weekday/weekend traffic patterns, so it
carries more signal for correlation/forecasting than a simple dummy.

Requires:
    data/processed/urbanpulse_master.csv

Output:
    data/processed/urbanpulse_master_with_traffic.csv
"""

import pandas as pd

# Hourly congestion weights (0-1), informed by widely reported Delhi
# weekday commute patterns: morning peak ~8-10am, evening peak ~6-9pm,
# a smaller midday dip, and low overnight traffic.
WEEKDAY_HOURLY_WEIGHTS = {
    0: 0.05, 1: 0.03, 2: 0.02, 3: 0.02, 4: 0.05, 5: 0.15,
    6: 0.35, 7: 0.60, 8: 0.90, 9: 1.00, 10: 0.80, 11: 0.65,
    12: 0.60, 13: 0.60, 14: 0.55, 15: 0.55, 16: 0.65, 17: 0.80,
    18: 0.95, 19: 1.00, 20: 0.90, 21: 0.70, 22: 0.40, 23: 0.15,
}

# Weekends: flatter profile, later/softer peaks, no sharp commute spikes
WEEKEND_HOURLY_WEIGHTS = {
    0: 0.10, 1: 0.05, 2: 0.03, 3: 0.03, 4: 0.05, 5: 0.08,
    6: 0.15, 7: 0.25, 8: 0.35, 9: 0.45, 10: 0.55, 11: 0.60,
    12: 0.60, 13: 0.60, 14: 0.55, 15: 0.55, 16: 0.55, 17: 0.60,
    18: 0.65, 19: 0.60, 20: 0.55, 21: 0.45, 22: 0.30, 23: 0.15,
}


def traffic_proxy_score(dt: pd.Timestamp) -> float:
    is_weekend = dt.dayofweek >= 5  # Saturday=5, Sunday=6
    weights = WEEKEND_HOURLY_WEIGHTS if is_weekend else WEEKDAY_HOURLY_WEIGHTS
    return weights[dt.hour]


def main():
    df = pd.read_csv('data/processed/urbanpulse_master.csv', parse_dates=['datetime'])

    df['is_weekend'] = df['datetime'].dt.dayofweek >= 5
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek  # 0=Mon ... 6=Sun
    df['traffic_proxy'] = df['datetime'].apply(traffic_proxy_score)

    # Also flag simple rush-hour windows as a binary feature, in case
    # a simpler feature performs better in some models
    df['is_rush_hour'] = (
        (~df['is_weekend']) &
        (df['hour'].isin([8, 9, 18, 19]))
    )

    out_path = 'data/processed/urbanpulse_master_with_traffic.csv'
    df.to_csv(out_path, index=False)

    print(f"Rows: {len(df)}")
    print(f"Columns added: is_weekend, hour, day_of_week, traffic_proxy, is_rush_hour")
    print(f"\nTraffic proxy summary:")
    print(df['traffic_proxy'].describe())
    print(f"\nSaved: {out_path}")
    print(df[['datetime', 'is_weekend', 'hour', 'traffic_proxy', 'is_rush_hour']].head(12))


if __name__ == '__main__':
    main()
