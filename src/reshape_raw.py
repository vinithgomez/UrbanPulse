"""
Reshape CPCB/OpenCity month-block hourly AQI CSVs into a clean long format.

Input format (per file, per station):
    Year,2017
    January-2017,00:00:00,01:00:00,...,23:00:00
    1,281.0,251.0,...,441
    2,434.0,437.0,...,341
    ...
    <blank/separator>
    February-2017,00:00:00,...
    ...

Output: one row per (station, datetime, aqi)
"""

import csv
import re
import calendar
from datetime import datetime, timedelta
import pandas as pd

MONTH_HEADER_RE = re.compile(r'^([A-Za-z]+)-(\d{4})$')
MONTHS = {m: i for i, m in enumerate(calendar.month_name) if m}


def parse_station_csv(path: str, station_name: str) -> pd.DataFrame:
    rows = []
    current_month = None
    current_year = None

    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for line in reader:
            if not line or line[0].strip() == '':
                continue
            first = line[0].strip()

            m = MONTH_HEADER_RE.match(first)
            if m:
                current_month = MONTHS.get(m.group(1))
                current_year = int(m.group(2))
                continue

            if first.lower() == 'year':
                continue

            # Data row: first cell should be a day-of-month integer
            if first.isdigit() and current_month is not None:
                day = int(first)
                try:
                    date_ = datetime(current_year, current_month, day)
                except ValueError:
                    # invalid day for this month (e.g. Feb 30) - skip
                    continue

                hour_values = line[1:25]  # up to 24 hourly values
                for hour, raw_val in enumerate(hour_values):
                    raw_val = raw_val.strip()
                    if raw_val == '':
                        aqi = None
                    else:
                        try:
                            aqi = float(raw_val)
                        except ValueError:
                            aqi = None
                    rows.append({
                        'station': station_name,
                        'datetime': date_ + timedelta(hours=hour),
                        'aqi': aqi
                    })

    df = pd.DataFrame(rows)
    return df


def main():
    files = {
        'ITO': 'data/raw/ito_raw.csv',
        'RK_Puram': 'data/raw/rkpuram_raw.csv',
        'DTU': 'data/raw/dtu_raw.csv',
    }

    all_dfs = []
    print(f"{'Station':<10} {'Rows':>8} {'First':>12} {'Last':>12} {'Missing':>8} {'%Miss':>7}")
    for station, path in files.items():
        df = parse_station_csv(path, station)
        df = df.drop_duplicates(subset=['station', 'datetime']).sort_values('datetime')
        n_missing = df['aqi'].isna().sum()
        pct_missing = 100 * n_missing / len(df) if len(df) else 0
        print(f"{station:<10} {len(df):>8} {str(df['datetime'].min()):>12} "
              f"{str(df['datetime'].max()):>12} {n_missing:>8} {pct_missing:>6.2f}%")
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(['station', 'datetime']).reset_index(drop=True)

    out_path = 'data/processed/aqi_combined_long.csv'
    combined.to_csv(out_path, index=False)
    print(f"\nSaved combined long-format dataset: {out_path}")
    print(f"Total rows: {len(combined)}")
    print(combined.head(10))

    # Also produce a wide format (one column per station) for quick eyeballing / correlation
    wide = combined.pivot_table(index='datetime', columns='station', values='aqi')
    wide_path = 'data/processed/aqi_combined_wide.csv'
    wide.to_csv(wide_path)
    print(f"Saved wide-format dataset: {wide_path}")


if __name__ == '__main__':
    main()
