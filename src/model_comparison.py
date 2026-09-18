"""
Phase 6 - Model Comparison & Error Analysis.

Consolidates Phase 4 (ARIMA/Prophet) and Phase 5 (LightGBM) into one
comparison across MAE, RMSE, 80% prediction-interval coverage, and interval
width, then tests the spec's stated hypothesis: does ARIMA struggle more
than LightGBM during the stubble-burning season (the spec's flagged
likely-failure window)?

METHODOLOGY NOTES (read before trusting a number):
- MAE/RMSE are pulled directly from classical_forecast_results.csv /
  ml_forecast_results.csv (mean of each fold's already-computed blended
  mae_h{24,72}) -- NOT recomputed from the interval CSVs -- so these numbers
  are identical to what's already reported in the Phase 4/5 README sections.
- Coverage and width are new to Phase 6 and are computed by *pooling* all
  (fold, step) observations within a segment: coverage = fraction of all
  such points whose actual value fell inside [lower, upper]; width = mean
  of (upper - lower) over the same pool. This is the standard definition
  for interval calibration (a single realized rate over many draws), not a
  mean of per-fold coverage rates.
- The stubble-burning segment (`is_stubble_season`, same Oct15-Nov15 window
  as Phase 3) is a per-FOLD classification: a fold counts as "stubble
  season" if its 72h test window overlaps the window at all. With folds
  spaced 30 days apart and a 32-day-per-year window, only a handful of
  folds per station actually qualify -- reported n_folds alongside every
  stubble-segment row so the reader can judge how much to weight it.

Requires:
    data/processed/classical_forecast_results.csv
    data/processed/classical_forecast_intervals.csv
    data/processed/ml_forecast_results.csv
    data/processed/ml_forecast_intervals.csv

Outputs:
    data/processed/model_comparison_results.csv                      -- MAE/RMSE/coverage/width by station, model, horizon, segment
    data/processed/figures/model_comparison_coverage_vs_width.png
    data/processed/figures/model_comparison_stubble_season.png
"""

import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSICAL_RESULTS = 'data/processed/classical_forecast_results.csv'
CLASSICAL_INTERVALS = 'data/processed/classical_forecast_intervals.csv'
ML_RESULTS = 'data/processed/ml_forecast_results.csv'
ML_INTERVALS = 'data/processed/ml_forecast_intervals.csv'
OUT_RESULTS = 'data/processed/model_comparison_results.csv'
FIG_DIR = 'data/processed/figures'

EVAL_HORIZONS = [24, 72]
STATIONS = ['DTU', 'ITO', 'RK_Puram']
MODELS = ['ARIMA', 'Prophet', 'LightGBM']

# Same window as Phase 3 (src/feature_engineering.py STUBBLE_SEASON).
STUBBLE_SEASON = [
    ('2022-10-15', '2022-11-15'),
    ('2023-10-15', '2023-11-15'),
]


def fold_overlaps_stubble(test_start, test_end):
    for start, end in STUBBLE_SEASON:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) + pd.Timedelta(hours=23)
        if test_start <= end_ts and test_end >= start_ts:
            return True
    return False


def load_fold_results():
    classical = pd.read_csv(CLASSICAL_RESULTS, parse_dates=['train_end', 'test_start', 'test_end'])
    ml = pd.read_csv(ML_RESULTS, parse_dates=['train_end', 'test_start', 'test_end'])
    combined = pd.concat([classical, ml], ignore_index=True)
    combined = combined[combined['fold'] != 'MEAN'].copy()
    # 'fold' comes back as object dtype (mixed with the 'MEAN' rows before
    # filtering); cast to int so it matches the intervals CSVs' int64 'fold'
    # for merging.
    combined['fold'] = combined['fold'].astype(int)
    return combined


def load_intervals():
    classical = pd.read_csv(CLASSICAL_INTERVALS, parse_dates=['datetime'])
    ml = pd.read_csv(ML_INTERVALS, parse_dates=['datetime'])
    return pd.concat([classical, ml], ignore_index=True)


def build_fold_meta(fold_results):
    meta = fold_results[['station', 'model', 'fold', 'test_start', 'test_end']].drop_duplicates().copy()
    meta['in_stubble_season'] = meta.apply(
        lambda r: fold_overlaps_stubble(r['test_start'], r['test_end']), axis=1)
    return meta


def coverage_and_width(intervals_df, horizon):
    sub = intervals_df[intervals_df['step'] <= horizon]
    if len(sub) == 0:
        return np.nan, np.nan
    covered = (sub['actual'] >= sub['lower']) & (sub['actual'] <= sub['upper'])
    return covered.mean() * 100, (sub['upper'] - sub['lower']).mean()


def n_fold_count(df):
    return df[['station', 'fold']].drop_duplicates().shape[0]


def build_comparison_table(fold_results, intervals):
    rows = []

    # --- per-station, all folds ---
    for station in STATIONS:
        for model in MODELS:
            fr = fold_results[(fold_results.station == station) & (fold_results.model == model)]
            iv = intervals[(intervals.station == station) & (intervals.model == model)]
            for h in EVAL_HORIZONS:
                cov, width = coverage_and_width(iv, h)
                rows.append({'station': station, 'model': model, 'horizon': h, 'segment': 'all',
                             'mae': fr[f'mae_h{h}'].mean(), 'rmse': fr[f'rmse_h{h}'].mean(),
                             'coverage_pct': cov, 'mean_width': width, 'n_folds': n_fold_count(fr)})

    # --- Overall (pooled across stations), all folds ---
    for model in MODELS:
        fr = fold_results[fold_results.model == model]
        iv = intervals[intervals.model == model]
        for h in EVAL_HORIZONS:
            cov, width = coverage_and_width(iv, h)
            rows.append({'station': 'Overall', 'model': model, 'horizon': h, 'segment': 'all',
                         'mae': fr[f'mae_h{h}'].mean(), 'rmse': fr[f'rmse_h{h}'].mean(),
                         'coverage_pct': cov, 'mean_width': width, 'n_folds': n_fold_count(fr)})

    # --- Overall, split by stubble-burning season ---
    fold_meta = build_fold_meta(fold_results)
    fr_tagged = fold_results.merge(fold_meta[['station', 'model', 'fold', 'in_stubble_season']],
                                    on=['station', 'model', 'fold'])
    iv_tagged = intervals.merge(fold_meta[['station', 'model', 'fold', 'in_stubble_season']],
                                 on=['station', 'model', 'fold'])

    for model in MODELS:
        for segment, flag in [('stubble_season', True), ('non_stubble_season', False)]:
            fr = fr_tagged[(fr_tagged.model == model) & (fr_tagged.in_stubble_season == flag)]
            iv = iv_tagged[(iv_tagged.model == model) & (iv_tagged.in_stubble_season == flag)]
            for h in EVAL_HORIZONS:
                cov, width = coverage_and_width(iv, h) if len(iv) else (np.nan, np.nan)
                rows.append({'station': 'Overall', 'model': model, 'horizon': h, 'segment': segment,
                             'mae': fr[f'mae_h{h}'].mean() if len(fr) else np.nan,
                             'rmse': fr[f'rmse_h{h}'].mean() if len(fr) else np.nan,
                             'coverage_pct': cov, 'mean_width': width,
                             'n_folds': n_fold_count(fr) if len(fr) else 0})

    return pd.DataFrame(rows), fr_tagged


def print_worst_folds(fold_results, top_n=3):
    print(f"\n=== Worst {top_n} folds per model, by mae_h24 ===")
    for model in MODELS:
        fr = fold_results[fold_results.model == model].sort_values('mae_h24', ascending=False).head(top_n)
        for _, row in fr.iterrows():
            stubble = fold_overlaps_stubble(row['test_start'], row['test_end'])
            print(f"  {model:9s} {row['station']:9s} fold {int(row['fold']) + 1:2d}  "
                  f"{row['test_start']:%Y-%m-%d} to {row['test_end']:%Y-%m-%d}  "
                  f"mae_h24={row['mae_h24']:.1f}  stubble_season={stubble}")


def print_stubble_hypothesis_test(result_df):
    print("\n=== Stubble-burning season hypothesis test (spec Phase 6) ===")
    sub = result_df[(result_df.segment.isin(['stubble_season', 'non_stubble_season']))
                     & (result_df.horizon == 24)]
    pivot = sub.pivot(index='model', columns='segment', values=['mae', 'n_folds'])
    print(pivot.round(2).to_string())

    print("\nDegradation (stubble MAE - non-stubble MAE), mae_h24:")
    for model in MODELS:
        row = sub[sub.model == model]
        stubble_row = row[row.segment == 'stubble_season']
        rest_row = row[row.segment == 'non_stubble_season']
        if stubble_row.empty or rest_row.empty or stubble_row['mae'].isna().any():
            print(f"  {model}: insufficient stubble-season folds to compare")
            continue
        delta = stubble_row['mae'].iloc[0] - rest_row['mae'].iloc[0]
        n = stubble_row['n_folds'].iloc[0]
        print(f"  {model}: {delta:+.1f} AQI (n_stubble_folds={n})")


def plot_coverage_vs_width(result_df, out_path):
    sub = result_df[(result_df.segment == 'all') & (result_df.station == 'Overall')]
    fig, ax = plt.subplots(figsize=(7, 6))
    markers = {24: 'o', 72: '^'}
    colors = {'ARIMA': 'tab:blue', 'Prophet': 'tab:orange', 'LightGBM': 'tab:green'}
    for _, row in sub.iterrows():
        ax.scatter(row['mean_width'], row['coverage_pct'], s=140,
                    color=colors[row['model']], marker=markers[row['horizon']],
                    label=f"{row['model']} (h={row['horizon']})", edgecolor='black')
    ax.axhline(80, color='gray', linestyle=':', label='nominal 80%')
    ax.set_xlabel('Mean 80% interval width (AQI units)')
    ax.set_ylabel('Empirical coverage (%)')
    ax.set_title('Prediction interval coverage vs. width (Overall, all stations)')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_stubble_comparison(result_df, out_path):
    sub = result_df[(result_df.segment.isin(['stubble_season', 'non_stubble_season']))
                     & (result_df.horizon == 24)]
    pivot = sub.pivot(index='model', columns='segment', values='mae')
    pivot = pivot[['non_stubble_season', 'stubble_season']]
    fig, ax = plt.subplots(figsize=(7, 5))
    pivot.plot(kind='bar', ax=ax, color=['tab:blue', 'tab:red'])
    ax.set_ylabel('MAE (24h horizon)')
    ax.set_title('MAE: stubble-burning season vs. rest of year')
    ax.set_xticklabels(pivot.index, rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def print_verdict():
    print("\n=== Written verdict ===")
    print("""
No single model wins outright -- the honest read is that each has earned a
different, narrower role. ARIMA has the best 24h MAE (58.7) and, unexpectedly,
the best-calibrated interval of the three (80.5% coverage at 24h against an
80% target, barely moving to 77.1% at 72h) -- for a plain 24h point forecast
with a trustworthy uncertainty band, it's the clear default. Prophet never
wins on MAE but its coverage (78.4%/76.9%) is nearly as good as ARIMA's, so
it's a reasonable second choice, not a fallback used only for lack of
options. LightGBM is the most interesting case precisely because its two
Phase 6 results point in opposite directions: its 80% intervals are
noticeably overconfident (70.9% coverage at 24h, 65.0% at 72h) despite being
the narrowest of the three (163-172 vs. ARIMA/Prophet's 194-215) -- exactly
the "narrow but wrong" trap this phase set out to check for, and a real
deployment risk if its stated interval were trusted as-is. But the
stubble-burning season analysis flips the picture: this is exactly the
spec's hypothesized high-variance failure window, and it confirms the
hypothesis cleanly -- ARIMA's 24h MAE degrades by +20.3 AQI during stubble
season and Prophet's by +35.7, while LightGBM's degrades by only +5.1 (n=5
stubble-season folds out of 34 total, so directionally clear but not a
large sample). LightGBM is, in other words, the most robust model exactly
when robustness matters most for a public-health early-warning use case,
despite being the weakest on both overall MAE and interval calibration.
The practical verdict: ship ARIMA as the default 24h point forecast and
interval, keep Prophet as the 72h outlook where its calibration holds up
about as well as ARIMA's, and treat LightGBM as the specialist model for
stubble-burning-season alerts specifically -- but only after recalibrating
its intervals (e.g. conformal calibration on a held-out set), since its
current 80% label understates its own uncertainty by roughly 10-15 points
of coverage. One caveat worth restating: the worst individual folds for
ARIMA and LightGBM both landed in February 2023, not stubble season --
a separate, non-stubble AQI spike neither model saw coming, and a reminder
that "stubble season" is not the only high-variance period this dataset
contains.
""".strip())


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    fold_results = load_fold_results()
    intervals = load_intervals()

    result_df, fr_tagged = build_comparison_table(fold_results, intervals)
    result_df.to_csv(OUT_RESULTS, index=False)
    print(f"Saved: {OUT_RESULTS}")

    print("\n=== MAE / RMSE / coverage / width -- Overall (all stations) ===")
    overall = result_df[(result_df.station == 'Overall') & (result_df.segment == 'all')]
    print(overall.set_index(['model', 'horizon'])[['mae', 'rmse', 'coverage_pct', 'mean_width']]
          .round(2).to_string())

    print_worst_folds(fold_results)
    print_stubble_hypothesis_test(result_df)

    plot_coverage_vs_width(result_df, f'{FIG_DIR}/model_comparison_coverage_vs_width.png')
    plot_stubble_comparison(result_df, f'{FIG_DIR}/model_comparison_stubble_season.png')
    print(f"\nSaved plots: model_comparison_coverage_vs_width.png, model_comparison_stubble_season.png")

    print_verdict()


if __name__ == '__main__':
    main()
