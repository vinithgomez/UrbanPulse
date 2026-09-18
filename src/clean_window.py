"""
Filter to the chosen analysis window (2022-07-01 to 2023-12-31) and apply
the missing-data policy from the spec:
  - forward-fill SHORT gaps (<= 3 consecutive missing hours)
  - flag LONGER gaps rather than silently interpolate (leave as NaN + flag column)
"""
import pandas as pd

SHORT_GAP_THRESHOLD_HOURS = 3
WINDOW_START = '2022-07-01'
WINDOW_END = '2023-12-31 23:00:00'

df = pd.read_csv('data/processed/aqi_combined_long.csv', parse_dates=['datetime'])

# Filter to window
df = df[(df['datetime'] >= WINDOW_START) & (df['datetime'] <= WINDOW_END)].copy()
df = df.sort_values(['station', 'datetime']).reset_index(drop=True)

results = []
for station, sub in df.groupby('station'):
    sub = sub.sort_values('datetime').reset_index(drop=True)

    # Identify consecutive-missing run lengths
    is_na = sub['aqi'].isna()
    run_id = (is_na != is_na.shift()).cumsum()
    run_lengths = is_na.groupby(run_id).transform('sum')
    # run_lengths is only meaningful where is_na is True; where is_na False, transform sum
    # gives count of NaNs in that (non-na) run which is 0 - fine.

    sub['gap_length_hrs'] = run_lengths.where(is_na, 0)
    sub['is_long_gap'] = is_na & (sub['gap_length_hrs'] > SHORT_GAP_THRESHOLD_HOURS)
    sub['was_imputed'] = is_na & (sub['gap_length_hrs'] <= SHORT_GAP_THRESHOLD_HOURS)

    # Forward-fill only the short gaps
    sub['aqi_clean'] = sub['aqi']
    short_gap_mask = sub['was_imputed']
    sub.loc[short_gap_mask, 'aqi_clean'] = sub['aqi'].ffill()[short_gap_mask]

    results.append(sub)

clean = pd.concat(results, ignore_index=True)

# Summary
print(f"Window: {WINDOW_START} to {WINDOW_END}")
print(f"Total rows: {len(clean)}\n")

summary = clean.groupby('station').agg(
    total_hours=('aqi', 'size'),
    originally_missing=('aqi', lambda x: x.isna().sum()),
    short_gap_imputed=('was_imputed', 'sum'),
    long_gap_flagged=('is_long_gap', 'sum'),
    still_missing_after_clean=('aqi_clean', lambda x: x.isna().sum()),
)
summary['pct_still_missing'] = (summary['still_missing_after_clean'] / summary['total_hours'] * 100).round(2)
print(summary)

out_path = 'data/processed/aqi_window_cleaned.csv'
clean.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
