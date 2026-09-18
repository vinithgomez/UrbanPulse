"""
Phase 5 - ML Forecasting (LightGBM).

Same rolling-origin backtest as Phase 4 (src/forecast_classical.py: same
NaN-dropped rows per station, same fold origins, same 24h/72h evaluation
horizons), so LightGBM's numbers are directly comparable to ARIMA/Prophet's.

WEATHER LEAKAGE DECISION (see README > Data decisions for the full
reasoning): weather columns (temperature_c, humidity_pct, wind_speed_kmh,
precipitation_mm, pressure_hpa) are EXCLUDED from the feature set entirely.
Using their actual historical values as "known" inputs for a forecast 24-72h
ahead would assume a perfect weather forecast and would also break the
apples-to-apples comparison with Phase 4, where the same reasoning excluded
weather from Prophet. Only past-AQI-derived features (lag/rolling) and
deterministic calendar/traffic-proxy features (known arbitrarily far in
advance) are used.

Modeling approach: direct multi-horizon regression with horizon-as-feature.
For a given fold with training cutoff `origin`, one LightGBM model is
trained on stacked (origin_i, horizon_h) examples drawn only from the
training portion (i + h < origin, so no leakage into the test window):
features = lag/rolling values AS OF row i, calendar features AT the target
row i+h (known in advance), and h itself. At test time, the model is
queried once per h in 1..72 using the last known row (origin-1) for
lag/rolling and the target row's calendar features -- producing a full
72-step forecast comparable to ARIMA/Prophet's, using the exact same
compute_metrics() definition as Phase 4.

NOTE ON POSITIONAL VS. REAL TIME (same caveat as Phase 4): the lag/rolling
*feature* columns were computed on the original, continuous hourly grid in
Phase 3, so they correctly reflect real elapsed time. But once NaN rows are
dropped and the per-station series is reindexed, the horizon `h` in
"h steps ahead" refers to the h-th *remaining* row, not literally h real
hours ahead, whenever a dropped gap falls in between (up to ~18 days for
ITO). This is the same simplification Phase 4 discloses for ARIMA, applied
here for a fair, consistent comparison.

Requires:
    data/processed/urbanpulse_features.csv
    data/processed/classical_forecast_results.csv (for the final 3-way comparison print)

Outputs:
    data/processed/ml_forecast_results.csv                     -- per-fold + per-station mean MAE/RMSE (same schema as Phase 4)
    data/processed/ml_forecast_intervals.csv                   -- per-step actual/pred/80% interval bounds (for Phase 6 coverage/width analysis)
    data/processed/figures/ml_forecast_<station>.png           -- actual vs. predicted, incl. a spike-event fold
    data/processed/figures/ml_feature_importance_<station>.png -- mean gain importance across folds

PHASE 6 ADDENDUM: 80% prediction intervals via quantile regression -- two
extra LightGBM models per fold (objective='quantile', alpha=0.1 and 0.9),
same feature set and same fold as the point (MSE-objective) model, additive
to it rather than a replacement.
"""

import os
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

IN_PATH = 'data/processed/urbanpulse_features.csv'
CLASSICAL_RESULTS_PATH = 'data/processed/classical_forecast_results.csv'
OUT_RESULTS = 'data/processed/ml_forecast_results.csv'
OUT_INTERVALS = 'data/processed/ml_forecast_intervals.csv'
FIG_DIR = 'data/processed/figures'

INTERVAL_WIDTH = 0.80  # matches Phase 4's ARIMA/Prophet 80% intervals
Q_LOW, Q_HIGH = 0.1, 0.9  # alpha=0.1/0.9 quantiles -> 80% interval

STATIONS = ['DTU', 'ITO', 'RK_Puram']
LAG_ROLL_PREFIXES = ('aqi_lag', 'aqi_rolling')
LAG_ROLL_COLS = ['aqi_lag_1h', 'aqi_lag_6h', 'aqi_lag_24h',
                  'aqi_rolling_mean_6h', 'aqi_rolling_std_6h',
                  'aqi_rolling_mean_24h', 'aqi_rolling_std_24h']
# Deterministic, known-arbitrarily-far-in-advance features. traffic_proxy is
# included deliberately -- Phase 2 found it only weakly correlated with AQI,
# and this is the test of whether it earns a real place in the model or
# ranks near the bottom of feature importance.
CALENDAR_COLS = ['hour', 'day_of_week', 'is_weekend', 'is_holiday',
                  'is_stubble_season', 'is_rush_hour', 'traffic_proxy']
FEATURE_COLS = LAG_ROLL_COLS + CALENDAR_COLS + ['horizon']

# Must match Phase 4 (src/forecast_classical.py) exactly so folds line up.
MIN_TRAIN_HOURS = 24 * 180
STEP_HOURS = 24 * 30
HORIZON = 72
EVAL_HORIZONS = [24, 72]
HORIZONS = list(range(1, HORIZON + 1))

# Subsample training origins by this stride when building the direct-horizon
# frame -- adjacent hourly origins are highly autocorrelated, so this keeps
# LightGBM training time reasonable (up to ~800k rows/fold otherwise)
# without materially reducing the information available to the model.
ORIGIN_STRIDE = 3

LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
                    importance_type='gain', random_state=42, verbosity=-1)
# Same capacity/learning-rate as the point model -- only objective/alpha
# differ, so the interval reflects the same modeling choices, not a
# separately-tuned model.
LGBM_PARAMS_Q_LOW = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
                          objective='quantile', alpha=Q_LOW, random_state=42, verbosity=-1)
LGBM_PARAMS_Q_HIGH = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           objective='quantile', alpha=Q_HIGH, random_state=42, verbosity=-1)


def load_station_data(path, station):
    """Same drop rule as Phase 4's load_station_series: drop rows with NaN
    in aqi_clean or any lag/rolling feature. Keeps the full feature set
    (Phase 4 only needed aqi_clean + datetime)."""
    df = pd.read_csv(path, parse_dates=['datetime'])
    df = df[df['station'] == station].sort_values('datetime').reset_index(drop=True)
    required_cols = ['aqi_clean'] + [c for c in df.columns if c.startswith(LAG_ROLL_PREFIXES)]
    n_before = len(df)
    keep_mask = ~df[required_cols].isna().any(axis=1)
    n_dropped = int((~keep_mask).sum())

    keep_cols = ['datetime', 'aqi_clean'] + LAG_ROLL_COLS + CALENDAR_COLS
    clean = df.loc[keep_mask, keep_cols].reset_index(drop=True)
    for c in ['is_weekend', 'is_holiday', 'is_stubble_season', 'is_rush_hour']:
        clean[c] = clean[c].astype(int)
    return clean, n_before, n_dropped


def rolling_origin_folds(n, min_train, step, horizon):
    """Identical to Phase 4's version -- same signature, same output, so
    folds line up exactly given the same (n, min_train, step, horizon)."""
    folds = []
    origin = min_train
    while origin + horizon <= n:
        folds.append(origin)
        origin += step
    return folds


def compute_metrics(actual, predicted, horizons):
    """Identical to Phase 4's compute_metrics: blended MAE/RMSE over steps
    1..h, so mae_h24 and mae_h72 mean the same thing in both CSVs."""
    out = {}
    for h in horizons:
        err = actual[:h] - predicted[:h]
        out[f'mae_h{h}'] = float(np.mean(np.abs(err)))
        out[f'rmse_h{h}'] = float(np.sqrt(np.mean(err ** 2)))
    return out


def build_horizon_frame(df, max_origin, horizons, stride, min_origin=0):
    """Stack direct-horizon examples for origins i with min_origin <= i and i+h < max_origin.

    Features: lag/rolling AS OF row i (unshifted); calendar AT the target
    row i+h (via a vectorized shift(-h), aligned back to index i); horizon=h.
    Target: aqi_clean AT row i+h.

    `min_origin` (default 0, i.e. the original full-training-range behavior)
    lets a caller restrict origins to a sub-range -- used by
    src/calibrate_intervals.py to build a calibration-only frame from the
    tail of a fold's training period, disjoint from the range used to fit
    the models being calibrated (split conformal requires this disjointness;
    see that script's docstring).
    """
    aqi = df['aqi_clean']
    frames = []
    for h in horizons:
        valid_end = max_origin - h
        if valid_end <= min_origin:
            continue
        target = aqi.shift(-h)
        calendar_shifted = df[CALENDAR_COLS].shift(-h)

        idx = slice(min_origin, valid_end, stride)
        frame = df[LAG_ROLL_COLS].iloc[idx].reset_index(drop=True)
        cal = calendar_shifted.iloc[idx].reset_index(drop=True)
        frame[CALENDAR_COLS] = cal
        frame['horizon'] = h
        frame['target'] = target.iloc[idx].to_numpy()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_test_frame(df, origin, horizons):
    """One row per horizon h: lag/rolling from the last known row (origin-1,
    constant across h), calendar from the target row (origin-1+h)."""
    base_row = df[LAG_ROLL_COLS].iloc[origin - 1]
    rows = []
    for h in horizons:
        target_row = df[CALENDAR_COLS].iloc[origin - 1 + h]
        row = dict(base_row)
        row.update(target_row.to_dict())
        row['horizon'] = h
        rows.append(row)
    return pd.DataFrame(rows)[FEATURE_COLS]


def run_station(station):
    print(f"\n{'=' * 70}\nStation: {station}\n{'=' * 70}")
    df, n_before, n_dropped = load_station_data(IN_PATH, station)
    print(f"Rows before drop: {n_before} | dropped (NaN in aqi_clean/lag/rolling): "
          f"{n_dropped} ({n_dropped / n_before:.1%}) | remaining: {len(df)} "
          f"[should match Phase 4's counts exactly]")

    y = df['aqi_clean']
    ts = df['datetime']
    n = len(df)

    folds = rolling_origin_folds(n, MIN_TRAIN_HOURS, STEP_HOURS, HORIZON)
    print(f"Backtest folds: {len(folds)} (expanding window, step={STEP_HOURS}h, "
          f"horizon={HORIZON}h) -- same fold origins as Phase 4")

    results = []
    interval_rows = []
    fold_data = {}
    importances = []

    for i, origin in enumerate(folds):
        train_frame = build_horizon_frame(df, origin, HORIZONS, ORIGIN_STRIDE)
        X_train, y_train = train_frame[FEATURE_COLS], train_frame['target']

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(X_train, y_train)
        importances.append(pd.Series(model.feature_importances_, index=FEATURE_COLS))

        # 80% prediction interval: two extra quantile models, same features,
        # same fold -- additive to the point model, not a replacement.
        model_low = lgb.LGBMRegressor(**LGBM_PARAMS_Q_LOW)
        model_low.fit(X_train, y_train)
        model_high = lgb.LGBMRegressor(**LGBM_PARAMS_Q_HIGH)
        model_high.fit(X_train, y_train)

        test_ts = ts.iloc[origin:origin + HORIZON]
        test_y = y.iloc[origin:origin + HORIZON].to_numpy()
        X_test = build_test_frame(df, origin, HORIZONS)
        pred = model.predict(X_test)
        pred_lower = model_low.predict(X_test)
        pred_upper = model_high.predict(X_test)
        # Quantile models are fit independently, so crossing (lower > upper)
        # is possible in principle, especially with sparse data; enforce the
        # ordering rather than silently reporting an inverted interval.
        pred_lower, pred_upper = np.minimum(pred_lower, pred_upper), np.maximum(pred_lower, pred_upper)

        metrics = compute_metrics(test_y, pred, EVAL_HORIZONS)
        results.append({'station': station, 'fold': i, 'n_train': origin,
                         'train_end': ts.iloc[origin - 1], 'test_start': test_ts.iloc[0],
                         'test_end': test_ts.iloc[-1], 'model': 'LightGBM', **metrics})
        fold_data[i] = {'test_ts': test_ts, 'test_y': test_y, 'pred': pred}

        for step in range(HORIZON):
            interval_rows.append({'station': station, 'model': 'LightGBM', 'fold': i,
                                    'step': step + 1, 'datetime': test_ts.iloc[step],
                                    'actual': test_y[step], 'pred': pred[step],
                                    'lower': pred_lower[step], 'upper': pred_upper[step]})

        print(f"  fold {i + 1}/{len(folds)} (train_n={origin}, "
              f"train_rows_used={len(X_train)}): "
              f"mae_h24={metrics['mae_h24']:.1f} rmse_h24={metrics['rmse_h24']:.1f} "
              f"mae_h72={metrics['mae_h72']:.1f} rmse_h72={metrics['rmse_h72']:.1f}")

    spike_fold = max(fold_data, key=lambda k: fold_data[k]['test_y'].max())
    example_idx = {0: 'first fold'}
    example_idx[spike_fold] = 'largest observed spike' if spike_fold != 0 else 'first fold (also largest spike)'
    examples = [{'fold': idx, 'tag': tag, **fold_data[idx]} for idx, tag in example_idx.items()]

    importance_df = pd.DataFrame(importances)
    return pd.DataFrame(results), examples, importance_df, pd.DataFrame(interval_rows)


def plot_station_examples(station, examples, out_path):
    fig, axes = plt.subplots(len(examples), 1, figsize=(11, 4 * len(examples)), squeeze=False)
    for ax, ex in zip(axes[:, 0], examples):
        ts = ex['test_ts']
        ax.plot(ts, ex['test_y'], label='Actual', color='black', linewidth=1.6)
        ax.plot(ts, ex['pred'], label='LightGBM', linestyle='--', color='tab:green')
        ax.set_title(f"{station} - fold {ex['fold'] + 1} ({ex['tag']}): "
                      f"{ts.iloc[0]:%Y-%m-%d %H:%M} to {ts.iloc[-1]:%Y-%m-%d %H:%M}")
        ax.set_ylabel('AQI')
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_feature_importance(station, importance_df, out_path):
    avg = importance_df.mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['tab:red' if f == 'traffic_proxy' else 'tab:blue' for f in avg.index]
    ax.barh(avg.index, avg.to_numpy(), color=colors)
    ax.set_xlabel('Mean gain importance (avg across folds)')
    ax.set_title(f'{station} - LightGBM feature importance (traffic_proxy in red)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    all_results = []
    all_intervals = []

    for station in STATIONS:
        results_df, examples, importance_df, interval_df = run_station(station)
        all_results.append(results_df)
        all_intervals.append(interval_df)

        plot_station_examples(station, examples, f'{FIG_DIR}/ml_forecast_{station}.png')
        plot_feature_importance(station, importance_df, f'{FIG_DIR}/ml_feature_importance_{station}.png')
        print(f"Saved plots: ml_forecast_{station}.png, ml_feature_importance_{station}.png")

        avg_imp = importance_df.mean().sort_values(ascending=False)
        rank = list(avg_imp.index).index('traffic_proxy') + 1
        print(f"{station}: traffic_proxy ranks {rank}/{len(avg_imp)} by mean gain "
              f"({avg_imp['traffic_proxy']:.1f}); top feature is '{avg_imp.index[0]}' "
              f"({avg_imp.iloc[0]:.1f})")

    pd.concat(all_intervals, ignore_index=True).to_csv(OUT_INTERVALS, index=False)
    print(f"\nSaved: {OUT_INTERVALS}")

    results_df = pd.concat(all_results, ignore_index=True)

    summary = (results_df.groupby(['station', 'model'])[
        ['mae_h24', 'rmse_h24', 'mae_h72', 'rmse_h72']
    ].mean().reset_index())
    summary['fold'] = 'MEAN'

    combined = pd.concat([results_df, summary], ignore_index=True, sort=False)
    combined.to_csv(OUT_RESULTS, index=False)
    print(f"\nSaved: {OUT_RESULTS}")

    print("\n=== LightGBM summary: mean MAE/RMSE per station ===")
    print(summary.set_index(['station', 'model'])[['mae_h24', 'rmse_h24', 'mae_h72', 'rmse_h72']]
          .round(2).to_string())

    # --- 3-way comparison against Phase 4's saved results ---
    if os.path.exists(CLASSICAL_RESULTS_PATH):
        classical = pd.read_csv(CLASSICAL_RESULTS_PATH)
        classical_folds = classical[classical['fold'] != 'MEAN']
        classical_summary = classical[classical['fold'] == 'MEAN']

        three_way_summary = pd.concat([classical_summary, summary], ignore_index=True, sort=False)
        print("\n=== 3-way comparison: mean MAE/RMSE per station (ARIMA vs Prophet vs LightGBM) ===")
        print(three_way_summary.set_index(['station', 'model'])[
            ['mae_h24', 'rmse_h24', 'mae_h72', 'rmse_h72']
        ].round(2).sort_index().to_string())

        overall_all = pd.concat([classical_folds, results_df], ignore_index=True, sort=False)
        overall = overall_all.groupby('model')[['mae_h24', 'rmse_h24', 'mae_h72', 'rmse_h72']].mean()
        print("\n=== Overall (all stations, all folds): ARIMA vs Prophet vs LightGBM ===")
        print(overall.round(2).to_string())
    else:
        print(f"\n{CLASSICAL_RESULTS_PATH} not found -- skipping 3-way comparison print.")


if __name__ == '__main__':
    main()
