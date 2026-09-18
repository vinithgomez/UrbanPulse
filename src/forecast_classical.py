"""
Phase 4 - Classical Forecasting (ARIMA + Prophet baselines).

Resolves the open NaN-handling decision from Phase 3 (see README > Data
decisions): rows with NaN in aqi_clean or any lag/rolling feature are
dropped before modeling, per station, rather than imputed. Exact drop
counts are printed per station and recorded in README.

For each station:
  1. ADF test for stationarity on aqi_clean -> chooses differencing order d
  2. AIC grid search over (p, q) at that d -> chooses ARIMA order
  3. Prophet fit as a second, univariate baseline (see note below on why
     weather is not used as a Prophet regressor here)
  4. Rolling-origin (expanding-window) backtesting: multiple folds sliding
     forward through the dataset, each forecasting HORIZON hours ahead,
     with MAE/RMSE reported at a short (24h) and the full (72h) horizon

Requires:
    data/processed/urbanpulse_features.csv

Outputs:
    data/processed/classical_forecast_results.csv      -- per-fold + per-station/model mean MAE/RMSE
    data/processed/classical_forecast_intervals.csv     -- per-step actual/pred/80% interval bounds (for Phase 6 coverage/width analysis)
    data/processed/figures/classical_forecast_<station>.png -- actual vs. predicted, example folds

PHASE 6 ADDENDUM: prediction intervals (80%) are extracted from the SAME
fitted models used for point forecasts -- ARIMA via get_forecast().conf_int()
instead of forecast() (no retraining, just a richer read of the same fit),
and Prophet's yhat_lower/yhat_upper, which it already computes internally at
interval_width=0.80 but which the original Phase 4 script discarded.
"""

import contextlib
import io
import logging
import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from prophet import Prophet

logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')  # statsmodels emits noisy convergence warnings during the AIC grid search

IN_PATH = 'data/processed/urbanpulse_features.csv'
OUT_RESULTS = 'data/processed/classical_forecast_results.csv'
OUT_INTERVALS = 'data/processed/classical_forecast_intervals.csv'
FIG_DIR = 'data/processed/figures'

INTERVAL_WIDTH = 0.80  # 80% prediction interval, matched across ARIMA/Prophet/LightGBM (Phase 6)
ARIMA_CI_ALPHA = 1 - INTERVAL_WIDTH  # conf_int()'s alpha is the two-tailed exclusion, e.g. 0.2 for 80%

STATIONS = ['DTU', 'ITO', 'RK_Puram']
LAG_ROLL_PREFIXES = ('aqi_lag', 'aqi_rolling')

MIN_TRAIN_HOURS = 24 * 180  # ~6 months initial (expanding-window) training size
STEP_HOURS = 24 * 30        # advance ~1 month per fold
HORIZON = 72                # forecast 72h ahead per fold (spec: 24-72h)
EVAL_HORIZONS = [24, 72]    # report metrics at a short and the full horizon

ARIMA_P_GRID = [0, 1, 2, 3]
ARIMA_Q_GRID = [0, 1, 2, 3]
ADF_ALPHA = 0.05
MAX_DIFF = 2


def load_station_series(path, station):
    """Load one station's series, dropping rows with NaN in aqi_clean or
    any lag/rolling feature -- resolving the Phase 3 open decision by
    dropping rather than imputing, consistent with this project's
    flag-don't-paper-over philosophy."""
    df = pd.read_csv(path, parse_dates=['datetime'])
    df = df[df['station'] == station].sort_values('datetime').reset_index(drop=True)
    required_cols = ['aqi_clean'] + [c for c in df.columns if c.startswith(LAG_ROLL_PREFIXES)]
    n_before = len(df)
    keep_mask = ~df[required_cols].isna().any(axis=1)
    n_dropped = int((~keep_mask).sum())
    clean = df.loc[keep_mask, ['datetime', 'aqi_clean']].reset_index(drop=True)
    return clean, n_before, n_dropped


def determine_d(series, alpha=ADF_ALPHA, max_diff=MAX_DIFF):
    """ADF test; difference until stationary (p < alpha) or max_diff reached."""
    s = series.copy()
    d = 0
    stat, p = adfuller(s, autolag='AIC')[:2]
    while p >= alpha and d < max_diff:
        s = s.diff().dropna()
        d += 1
        stat, p = adfuller(s, autolag='AIC')[:2]
    return d, stat, p


def select_arima_order(train_series, d):
    """Small AIC grid search over (p, q) at the fixed differencing order d."""
    best_order, best_aic = None, np.inf
    for p in ARIMA_P_GRID:
        for q in ARIMA_Q_GRID:
            if p == 0 and q == 0 and d == 0:
                continue
            try:
                fit = ARIMA(train_series, order=(p, d, q)).fit()
            except Exception:
                continue
            if fit.aic < best_aic:
                best_order, best_aic = (p, d, q), fit.aic
    return best_order, best_aic


def rolling_origin_folds(n, min_train, step, horizon):
    folds = []
    origin = min_train
    while origin + horizon <= n:
        folds.append(origin)
        origin += step
    return folds


def compute_metrics(actual, predicted, horizons):
    out = {}
    for h in horizons:
        err = actual[:h] - predicted[:h]
        out[f'mae_h{h}'] = float(np.mean(np.abs(err)))
        out[f'rmse_h{h}'] = float(np.sqrt(np.mean(err ** 2)))
    return out


def fit_prophet(train_ts, train_y):
    m = Prophet(interval_width=INTERVAL_WIDTH)
    df = pd.DataFrame({'ds': train_ts.to_numpy(), 'y': train_y.to_numpy()})
    with contextlib.redirect_stdout(io.StringIO()):
        m.fit(df)
    return m


def run_station(station):
    print(f"\n{'=' * 70}\nStation: {station}\n{'=' * 70}")
    clean, n_before, n_dropped = load_station_series(IN_PATH, station)
    print(f"Rows before drop: {n_before} | dropped (NaN in aqi_clean/lag/rolling): "
          f"{n_dropped} ({n_dropped / n_before:.1%}) | remaining: {len(clean)}")

    y = clean['aqi_clean']
    ts = clean['datetime']

    initial_train = y.iloc[:MIN_TRAIN_HOURS]
    d, adf_stat, adf_p = determine_d(initial_train)
    print(f"ADF test on initial {MIN_TRAIN_HOURS}h training window: "
          f"d={d} chosen (final ADF stat={adf_stat:.3f}, p={adf_p:.4g})")

    order, aic = select_arima_order(initial_train, d)
    print(f"ARIMA order selected via AIC grid search over p,q in "
          f"{ARIMA_P_GRID}x{ARIMA_Q_GRID}: order={order}, AIC={aic:.1f}")

    folds = rolling_origin_folds(len(y), MIN_TRAIN_HOURS, STEP_HOURS, HORIZON)
    print(f"Backtest folds: {len(folds)} (expanding window, step={STEP_HOURS}h, horizon={HORIZON}h)")

    results = []
    interval_rows = []
    examples = []
    example_idx = {0, len(folds) // 2} if len(folds) > 1 else {0}

    for i, origin in enumerate(folds):
        train_y = y.iloc[:origin]
        train_ts = ts.iloc[:origin]
        test_y = y.iloc[origin:origin + HORIZON].to_numpy()
        test_ts = ts.iloc[origin:origin + HORIZON]

        try:
            arima_fit = ARIMA(train_y, order=order).fit()
            # get_forecast() (not forecast()) re-reads the SAME fit for both
            # the point forecast and its 80% CI -- no retraining involved.
            arima_forecast_obj = arima_fit.get_forecast(steps=HORIZON)
            arima_pred = np.asarray(arima_forecast_obj.predicted_mean)
            arima_ci = np.asarray(arima_forecast_obj.conf_int(alpha=ARIMA_CI_ALPHA))
            arima_lower, arima_upper = arima_ci[:, 0], arima_ci[:, 1]
        except Exception:
            arima_pred = np.full(HORIZON, np.nan)
            arima_lower = np.full(HORIZON, np.nan)
            arima_upper = np.full(HORIZON, np.nan)

        try:
            m = fit_prophet(train_ts, train_y)
            future = pd.DataFrame({'ds': test_ts.to_numpy()})
            fcst = m.predict(future)
            prophet_pred = fcst['yhat'].to_numpy()
            prophet_lower = fcst['yhat_lower'].to_numpy()
            prophet_upper = fcst['yhat_upper'].to_numpy()
        except Exception:
            prophet_pred = np.full(HORIZON, np.nan)
            prophet_lower = np.full(HORIZON, np.nan)
            prophet_upper = np.full(HORIZON, np.nan)

        arima_metrics = compute_metrics(test_y, arima_pred, EVAL_HORIZONS)
        prophet_metrics = compute_metrics(test_y, prophet_pred, EVAL_HORIZONS)

        base = {'station': station, 'fold': i, 'n_train': origin,
                'train_end': train_ts.iloc[-1], 'test_start': test_ts.iloc[0],
                'test_end': test_ts.iloc[-1]}
        results.append({**base, 'model': 'ARIMA', **arima_metrics})
        results.append({**base, 'model': 'Prophet', **prophet_metrics})

        for step in range(HORIZON):
            interval_rows.append({'station': station, 'model': 'ARIMA', 'fold': i,
                                    'step': step + 1, 'datetime': test_ts.iloc[step],
                                    'actual': test_y[step], 'pred': arima_pred[step],
                                    'lower': arima_lower[step], 'upper': arima_upper[step]})
            interval_rows.append({'station': station, 'model': 'Prophet', 'fold': i,
                                    'step': step + 1, 'datetime': test_ts.iloc[step],
                                    'actual': test_y[step], 'pred': prophet_pred[step],
                                    'lower': prophet_lower[step], 'upper': prophet_upper[step]})

        print(f"  fold {i + 1}/{len(folds)} (train_n={origin}): "
              f"ARIMA mae_h24={arima_metrics['mae_h24']:.1f} rmse_h24={arima_metrics['rmse_h24']:.1f} | "
              f"Prophet mae_h24={prophet_metrics['mae_h24']:.1f} rmse_h24={prophet_metrics['rmse_h24']:.1f}")

        if i in example_idx:
            examples.append({'fold': i, 'test_ts': test_ts, 'test_y': test_y,
                              'arima_pred': arima_pred, 'prophet_pred': prophet_pred})

    return pd.DataFrame(results), order, d, adf_p, examples, pd.DataFrame(interval_rows)


def plot_station_examples(station, examples, out_path):
    fig, axes = plt.subplots(len(examples), 1, figsize=(11, 4 * len(examples)), squeeze=False)
    for ax, ex in zip(axes[:, 0], examples):
        ts = ex['test_ts']
        ax.plot(ts, ex['test_y'], label='Actual', color='black', linewidth=1.6)
        ax.plot(ts, ex['arima_pred'], label='ARIMA', linestyle='--')
        ax.plot(ts, ex['prophet_pred'], label='Prophet', linestyle='--')
        ax.set_title(f"{station} - fold {ex['fold'] + 1}: "
                      f"{ts.iloc[0]:%Y-%m-%d %H:%M} to {ts.iloc[-1]:%Y-%m-%d %H:%M}")
        ax.set_ylabel('AQI')
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    all_results = []
    all_intervals = []

    for station in STATIONS:
        results_df, order, d, adf_p, examples, interval_df = run_station(station)
        all_results.append(results_df)
        all_intervals.append(interval_df)

        plot_path = f'{FIG_DIR}/classical_forecast_{station}.png'
        plot_station_examples(station, examples, plot_path)
        print(f"Saved plot: {plot_path}")

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

    print("\n=== Summary: mean MAE/RMSE per station, ARIMA vs Prophet ===")
    print(summary.set_index(['station', 'model'])[['mae_h24', 'rmse_h24', 'mae_h72', 'rmse_h72']]
          .round(2).to_string())

    overall = results_df.groupby('model')[['mae_h24', 'rmse_h24', 'mae_h72', 'rmse_h72']].mean()
    print("\n=== Overall (all stations, all folds) ===")
    print(overall.round(2).to_string())


if __name__ == '__main__':
    main()
