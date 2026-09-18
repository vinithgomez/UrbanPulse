"""
Phase 2 - EDA & Correlation Analysis.

Answers the core question from UrbanPulse_Spec.md Phase 2: does the traffic
proxy predict AQI, at what lag, and does that relationship survive
controlling for weather (the main confound - AQI is heavily weather-driven,
so a raw traffic-AQI correlation alone would be misleading)?

Requires:
    data/processed/urbanpulse_master_with_traffic.csv

Outputs:
    data/processed/correlation_results.csv       -- lag correlations (raw + partial), with 95% CIs
    data/processed/figures/aqi_decomposition.png -- STL trend/seasonal/residual decomposition
    data/processed/figures/traffic_aqi_ccf.png   -- cross-correlation function, -48h to +48h
"""

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.seasonal import STL

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IN_PATH = 'data/processed/urbanpulse_master_with_traffic.csv'
FIG_DIR = 'data/processed/figures'
LAGS_HOURS = [0, 6, 12, 24]
WEATHER_COLS = ['temperature_c', 'humidity_pct', 'wind_speed_kmh',
                'precipitation_mm', 'pressure_hpa']


def load_city_wide_series(path):
    """Collapse the 3 stations into one Delhi-wide hourly series.

    aqi_clean is averaged across stations; weather and traffic_proxy are
    already identical across stations per hour (see README > Data
    decisions on weather granularity), so 'first' just picks that shared
    value.
    """
    df = pd.read_csv(path, parse_dates=['datetime'])
    city = (
        df.groupby('datetime')
          .agg(aqi_clean=('aqi_clean', 'mean'),
               traffic_proxy=('traffic_proxy', 'first'),
               temperature_c=('temperature_c', 'first'),
               humidity_pct=('humidity_pct', 'first'),
               wind_speed_kmh=('wind_speed_kmh', 'first'),
               precipitation_mm=('precipitation_mm', 'first'),
               pressure_hpa=('pressure_hpa', 'first'))
          .sort_index()
    )
    city = city.asfreq('h')
    return city


def pearson_with_ci(x, y, alpha=0.05):
    """Pearson r with a 95% CI via the Fisher z-transform."""
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    n = len(x)
    r, p = stats.pearsonr(x, y)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo, hi = np.tanh(z - z_crit * se), np.tanh(z + z_crit * se)
    return r, p, lo, hi, n


def residualize(y, X):
    """Residuals of y ~ X (OLS with intercept), row-aligned, NaNs dropped."""
    data = pd.concat([y, X], axis=1).dropna()
    Xc = sm.add_constant(data[X.columns])
    model = sm.OLS(data[y.name], Xc).fit()
    return pd.Series(model.resid, index=data.index, name=y.name)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    city = load_city_wide_series(IN_PATH)

    print(f"City-wide hourly series: {len(city)} hours "
          f"({city.index.min()} to {city.index.max()})")
    print(f"Missing aqi_clean after averaging stations: "
          f"{city['aqi_clean'].isna().sum()} hours "
          f"({city['aqi_clean'].isna().mean():.1%})")

    # --- Time-series decomposition (STL, daily period) ---
    # STL requires a continuous series, so remaining long gaps are
    # interpolated here for this visualization step only - this does not
    # touch the saved master dataset or its documented missing-data policy.
    aqi_interp = city['aqi_clean'].interpolate(limit_direction='both')
    stl = STL(aqi_interp, period=24, robust=True).fit()

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(aqi_interp.index, stl.observed)
    axes[0].set_ylabel('Observed')
    axes[0].set_title('Delhi city-wide AQI - STL decomposition (24h period)')
    axes[1].plot(aqi_interp.index, stl.trend)
    axes[1].set_ylabel('Trend')
    axes[2].plot(aqi_interp.index, stl.seasonal)
    axes[2].set_ylabel('Seasonal (24h)')
    axes[3].plot(aqi_interp.index, stl.resid)
    axes[3].set_ylabel('Residual')
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/aqi_decomposition.png', dpi=120)
    plt.close(fig)
    print(f"\nSaved decomposition plot: {FIG_DIR}/aqi_decomposition.png")

    # --- Cross-correlation: traffic_proxy(t) vs aqi_clean(t+lag), raw + partial ---
    weather_X = city[WEATHER_COLS]
    results = []

    print("\nCross-correlation: traffic_proxy(t) vs aqi_clean(t+lag)")
    header = (f"{'lag_h':>6} {'r_raw':>8} {'95% CI (raw)':>18} {'p_raw':>10} "
              f"{'r_partial':>10} {'95% CI (partial)':>20} {'n':>7}")
    print(header)

    for lag in LAGS_HOURS:
        aqi_lagged = city['aqi_clean'].shift(-lag)
        r, p, lo, hi, n = pearson_with_ci(city['traffic_proxy'], aqi_lagged)

        # Partial correlation: residualize both traffic_proxy and lagged AQI
        # on the weather covariates, then correlate the residuals. This
        # answers whether traffic still tracks AQI once weather - the main
        # confound - is controlled for.
        traffic_resid = residualize(city['traffic_proxy'].rename('traffic_proxy'), weather_X)
        aqi_resid = residualize(aqi_lagged.rename('aqi_lagged'), weather_X)
        common_idx = traffic_resid.index.intersection(aqi_resid.index)
        r_p, p_p, lo_p, hi_p, n_p = pearson_with_ci(
            traffic_resid.loc[common_idx], aqi_resid.loc[common_idx])

        print(f"{lag:>6} {r:>8.3f} [{lo:>6.3f},{hi:>6.3f}] {p:>10.2e} "
              f"{r_p:>10.3f} [{lo_p:>6.3f},{hi_p:>6.3f}] {n_p:>7}")

        results.append({
            'lag_hours': lag,
            'r_raw': r, 'ci_low_raw': lo, 'ci_high_raw': hi, 'p_raw': p, 'n_raw': n,
            'r_partial': r_p, 'ci_low_partial': lo_p, 'ci_high_partial': hi_p,
            'p_partial': p_p, 'n_partial': n_p,
        })

    results_df = pd.DataFrame(results)
    out_path = 'data/processed/correlation_results.csv'
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # --- CCF plot across a wider lag range, for visual context ---
    max_lag = 48
    ccf_lags = list(range(-max_lag, max_lag + 1))
    ccf_vals = [
        pearson_with_ci(city['traffic_proxy'], city['aqi_clean'].shift(-lag))[0]
        for lag in ccf_lags
    ]

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.stem(ccf_lags, ccf_vals)
    ax2.axvline(0, color='gray', linewidth=0.8)
    ax2.set_xlabel('Lag (hours): traffic_proxy(t) vs aqi_clean(t+lag)')
    ax2.set_ylabel('Pearson r')
    ax2.set_title('Cross-correlation: traffic proxy vs city-wide AQI')
    fig2.tight_layout()
    fig2.savefig(f'{FIG_DIR}/traffic_aqi_ccf.png', dpi=120)
    plt.close(fig2)
    print(f"Saved: {FIG_DIR}/traffic_aqi_ccf.png")


if __name__ == '__main__':
    main()
