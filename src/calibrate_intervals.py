"""
Split conformal calibration for LightGBM's 80% prediction intervals.

Closes Future Work item #2 from report/UrbanPulse_Report.md. Phase 6 found
LightGBM's stated 80% interval was actually overconfident -- 71.0%/65.0%
real coverage at 24h/72h (see README > Phase 6 results). This script
applies split conformal prediction, specifically Conformalized Quantile
Regression (CQR; Romano, Patterson & Candes, 2019), to the same quantile
models introduced in Phase 5/6, per rolling-origin fold -- an actual fix,
not just a documented recommendation.

METHOD, per fold (same fold origins as forecast_ml.py, so results stay
directly comparable to Phase 6's):
  1. Split the fold's training period [0, origin) into:
       - proper_train: [0, calib_start)       -- used to fit the quantile models
       - calibration:  [calib_start, origin)   -- held out, never seen by the models
     calib_start = origin - int(CALIB_FRAC * origin), i.e. the last 20% of
     the training period. Split conformal's coverage guarantee specifically
     requires this calibration slice to be disjoint from model fitting --
     that's the "split" in split conformal.
  2. Fit fresh q10/q90 LightGBM quantile models on proper_train only (same
     hyperparameters as forecast_ml.py's). This is a calibration-only
     refit using slightly less data than the original Phase 5/6 models;
     it does NOT touch or replace the point-forecast model, so Phase 5's
     reported MAE/RMSE are completely unaffected by this script.
  3. Predict q10/q90 on the calibration set and compute each point's
     nonconformity score: max(q10_pred - actual, actual - q90_pred) --
     positive exactly when the actual value fell outside the raw interval
     (CQR's defining score; Romano et al. eq. 4).
  4. Q = the finite-sample-corrected empirical 80th percentile of those
     scores (eq. 6: the ceil((n+1)*0.8)/n order statistic, not a plain
     quantile -- this is what gives the finite-sample coverage guarantee).
     The calibrated interval is [q10_pred - Q, q90_pred + Q] for that
     fold's real test-period predictions -- widened (Q>0) or narrowed
     (Q<0) by the same scalar Q at every horizon step.
  5. Coverage/width are recomputed on the calibrated intervals using the
     identical pooled-observation definition Phase 6 used (coverage = %
     of all (fold, step) points within [lower, upper]; width = mean of
     upper-lower over the same pool), so before/after numbers are directly
     comparable. The "before" side is Phase 6's already-saved, unchanged
     ml_forecast_intervals.csv -- not recomputed here -- so this script
     can't quietly shift the baseline it's claiming to improve on.

This gives split conformal's standard marginal coverage guarantee (over
the calibration+test distribution), not a per-fold guarantee -- each fold
only has one held-out test window -- but pooling every fold's calibration
points, per station, gives a substantial empirical check regardless.

Requires:
    data/processed/urbanpulse_features.csv
    data/processed/ml_forecast_intervals.csv   (Phase 6's original, uncalibrated intervals -- the "before" side)

Outputs:
    data/processed/ml_forecast_intervals_calibrated.csv  -- per-step calibrated lower/upper (actual/pred columns carried over unchanged)
    data/processed/ml_forecast_results_calibrated.csv    -- before/after coverage & width by station/horizon, plus mean Q
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

from forecast_ml import (
    IN_PATH, STATIONS, FEATURE_COLS, HORIZON, HORIZONS, EVAL_HORIZONS,
    MIN_TRAIN_HOURS, STEP_HOURS, ORIGIN_STRIDE,
    LGBM_PARAMS_Q_LOW, LGBM_PARAMS_Q_HIGH,
    load_station_data, rolling_origin_folds, build_horizon_frame, build_test_frame,
)

ORIGINAL_INTERVALS_PATH = 'data/processed/ml_forecast_intervals.csv'
OUT_INTERVALS = 'data/processed/ml_forecast_intervals_calibrated.csv'
OUT_RESULTS = 'data/processed/ml_forecast_results_calibrated.csv'

CALIB_FRAC = 0.20   # last 20% of each fold's training period held out for calibration
TARGET_COVERAGE = 0.80


def conformal_quantile(scores, target_coverage):
    """Finite-sample-corrected empirical quantile (Romano et al. 2019, eq. 6).
    Using the ceil((n+1)*coverage)/n order statistic instead of a plain
    np.quantile(scores, coverage) is what gives split conformal its
    finite-sample marginal coverage guarantee rather than only an
    asymptotic one."""
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * target_coverage) / n)
    return float(np.quantile(scores, level, method='higher'))


def coverage_and_width(actual, lower, upper, horizons):
    out = {}
    for h in horizons:
        a, lo, hi = actual[:h], lower[:h], upper[:h]
        covered = (a >= lo) & (a <= hi)
        out[f'coverage_h{h}'] = float(covered.mean() * 100)
        out[f'width_h{h}'] = float(np.mean(hi - lo))
    return out


def calibrate_fold(df, origin):
    """Fit q10/q90 on proper_train, derive Q from the held-out calibration
    slice, and return the calibrated [lower, upper] for the fold's real
    72h test window (plus Q and the calibration sample size, for reporting).
    """
    calib_hours = int(CALIB_FRAC * origin)
    calib_start = origin - calib_hours

    train_frame = build_horizon_frame(df, calib_start, HORIZONS, ORIGIN_STRIDE)
    X_train, y_train = train_frame[FEATURE_COLS], train_frame['target']

    model_low = lgb.LGBMRegressor(**LGBM_PARAMS_Q_LOW)
    model_low.fit(X_train, y_train)
    model_high = lgb.LGBMRegressor(**LGBM_PARAMS_Q_HIGH)
    model_high.fit(X_train, y_train)

    # Calibration examples: origins in [calib_start, origin) -- disjoint
    # from proper_train above, and strictly before the real test window.
    calib_frame = build_horizon_frame(df, origin, HORIZONS, ORIGIN_STRIDE, min_origin=calib_start)
    X_calib, y_calib = calib_frame[FEATURE_COLS], calib_frame['target'].to_numpy()
    q_low_calib = model_low.predict(X_calib)
    q_high_calib = model_high.predict(X_calib)
    q_low_calib, q_high_calib = np.minimum(q_low_calib, q_high_calib), np.maximum(q_low_calib, q_high_calib)

    scores = np.maximum(q_low_calib - y_calib, y_calib - q_high_calib)
    Q = conformal_quantile(scores, TARGET_COVERAGE)

    X_test = build_test_frame(df, origin, HORIZONS)
    q_low_test = model_low.predict(X_test)
    q_high_test = model_high.predict(X_test)
    q_low_test, q_high_test = np.minimum(q_low_test, q_high_test), np.maximum(q_low_test, q_high_test)

    calibrated_lower = q_low_test - Q
    calibrated_upper = q_high_test + Q
    # Q can be negative (narrowing) in principle; guard against ever
    # reporting an inverted interval, same convention as the raw
    # quantile-crossing guard in forecast_ml.py.
    calibrated_lower, calibrated_upper = (np.minimum(calibrated_lower, calibrated_upper),
                                           np.maximum(calibrated_lower, calibrated_upper))
    return calibrated_lower, calibrated_upper, Q, len(scores)


def run_station(station, original_intervals):
    print(f"\n{'=' * 70}\nStation: {station}\n{'=' * 70}")
    df, _, _ = load_station_data(IN_PATH, station)
    n = len(df)
    folds = rolling_origin_folds(n, MIN_TRAIN_HOURS, STEP_HOURS, HORIZON)

    station_orig = original_intervals[original_intervals.station == station]

    calibrated_rows = []
    fold_summaries = []

    for i, origin in enumerate(folds):
        calibrated_lower, calibrated_upper, Q, n_calib = calibrate_fold(df, origin)

        fold_orig = station_orig[station_orig.fold == i].sort_values('step')
        actual = fold_orig['actual'].to_numpy()
        pred = fold_orig['pred'].to_numpy()  # unchanged point forecast, carried over as-is
        raw_lower = fold_orig['lower'].to_numpy()
        raw_upper = fold_orig['upper'].to_numpy()
        test_ts = fold_orig['datetime'].to_numpy()

        raw_metrics = coverage_and_width(actual, raw_lower, raw_upper, EVAL_HORIZONS)
        cal_metrics = coverage_and_width(actual, calibrated_lower, calibrated_upper, EVAL_HORIZONS)

        print(f"  fold {i + 1}/{len(folds)} (n_calib={n_calib}, Q={Q:+.1f}): "
              f"coverage_h24 raw={raw_metrics['coverage_h24']:.0f}% -> "
              f"calibrated={cal_metrics['coverage_h24']:.0f}% | "
              f"width_h24 raw={raw_metrics['width_h24']:.0f} -> "
              f"calibrated={cal_metrics['width_h24']:.0f}")

        fold_summaries.append({
            'station': station, 'fold': i, 'n_calib_points': n_calib, 'Q': Q,
            **{f'{k}_raw': v for k, v in raw_metrics.items()},
            **{f'{k}_calibrated': v for k, v in cal_metrics.items()},
        })

        for step in range(HORIZON):
            calibrated_rows.append({
                'station': station, 'model': 'LightGBM_calibrated', 'fold': i,
                'step': step + 1, 'datetime': test_ts[step], 'actual': actual[step],
                'pred': pred[step], 'lower': calibrated_lower[step], 'upper': calibrated_upper[step],
            })

    return pd.DataFrame(fold_summaries), pd.DataFrame(calibrated_rows)


def _pooled_coverage_width(df, horizon_col='step', horizons=EVAL_HORIZONS):
    rows = {}
    for h in horizons:
        sub = df[df[horizon_col] <= h]
        cov = ((sub.actual >= sub.lower) & (sub.actual <= sub.upper)).mean() * 100
        width = (sub.upper - sub.lower).mean()
        rows[h] = (cov, width)
    return rows


def main():
    original_intervals = pd.read_csv(ORIGINAL_INTERVALS_PATH, parse_dates=['datetime'])

    all_summaries, all_calibrated = [], []
    for station in STATIONS:
        fold_summary, calibrated_df = run_station(station, original_intervals)
        all_summaries.append(fold_summary)
        all_calibrated.append(calibrated_df)

    calibrated_intervals = pd.concat(all_calibrated, ignore_index=True)
    calibrated_intervals.to_csv(OUT_INTERVALS, index=False)
    print(f"\nSaved: {OUT_INTERVALS}")

    fold_summary = pd.concat(all_summaries, ignore_index=True)

    results_rows = []
    for station in STATIONS:
        raw_cw = _pooled_coverage_width(original_intervals[original_intervals.station == station])
        cal_cw = _pooled_coverage_width(calibrated_intervals[calibrated_intervals.station == station])
        sub_fold_summary = fold_summary[fold_summary.station == station]
        for h in EVAL_HORIZONS:
            results_rows.append({
                'station': station, 'horizon': h,
                'coverage_pct_raw': raw_cw[h][0], 'mean_width_raw': raw_cw[h][1],
                'coverage_pct_calibrated': cal_cw[h][0], 'mean_width_calibrated': cal_cw[h][1],
                'mean_Q': sub_fold_summary['Q'].mean(), 'n_folds': sub_fold_summary['fold'].nunique(),
            })

    raw_cw_overall = _pooled_coverage_width(original_intervals)
    cal_cw_overall = _pooled_coverage_width(calibrated_intervals)
    for h in EVAL_HORIZONS:
        results_rows.append({
            'station': 'Overall', 'horizon': h,
            'coverage_pct_raw': raw_cw_overall[h][0], 'mean_width_raw': raw_cw_overall[h][1],
            'coverage_pct_calibrated': cal_cw_overall[h][0], 'mean_width_calibrated': cal_cw_overall[h][1],
            'mean_Q': fold_summary['Q'].mean(),
            'n_folds': fold_summary[['station', 'fold']].drop_duplicates().shape[0],
        })

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(OUT_RESULTS, index=False)
    print(f"Saved: {OUT_RESULTS}")

    print("\n=== Before vs. after: coverage & width (target 80% coverage) ===")
    print(results_df.set_index(['station', 'horizon'])[
        ['coverage_pct_raw', 'coverage_pct_calibrated', 'mean_width_raw', 'mean_width_calibrated', 'mean_Q']
    ].round(2).to_string())


if __name__ == '__main__':
    main()
