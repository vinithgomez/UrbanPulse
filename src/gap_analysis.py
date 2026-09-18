import pandas as pd

df = pd.read_csv('data/processed/aqi_combined_long.csv', parse_dates=['datetime'])

print("=== Missingness by station and year ===")
df['year'] = df['datetime'].dt.year
pivot = df.groupby(['station', 'year'])['aqi'].apply(lambda x: x.isna().mean() * 100).unstack(0)
print(pivot.round(2))

print("\n=== Missingness by station and month (all years combined) ===")
df['month'] = df['datetime'].dt.month
pivot2 = df.groupby(['station', 'month'])['aqi'].apply(lambda x: x.isna().mean() * 100).unstack(0)
print(pivot2.round(2))

print("\n=== Best candidate 12-18mo windows: yearly missing% recap ===")
print(pivot.mean(axis=1).round(2))

print("\n=== Gap clustering check: longest consecutive-missing streaks per station ===")
for station in df['station'].unique():
    sub = df[df['station'] == station].sort_values('datetime').reset_index(drop=True)
    is_na = sub['aqi'].isna()
    # find run lengths of consecutive True
    streak = 0
    max_streak = 0
    max_streak_start = None
    current_start = None
    for i, val in enumerate(is_na):
        if val:
            if streak == 0:
                current_start = sub.loc[i, 'datetime']
            streak += 1
            if streak > max_streak:
                max_streak = streak
                max_streak_start = current_start
        else:
            streak = 0
    print(f"{station}: longest gap = {max_streak} hours (~{max_streak/24:.1f} days), starting {max_streak_start}")
