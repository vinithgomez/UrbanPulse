"""
UrbanPulse - Streamlit viewer (Phase 7).

A read-only viewer over the analysis already computed in Phases 1-6. Not a
live forecasting tool -- no model runs when the app opens; every chart and
table here reads from CSVs/PNGs already saved under data/processed/. See
UrbanPulse_Spec.md Phase 7 and README.md for the full analysis writeup.

Run with:
    streamlit run app/streamlit_app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Resolve paths relative to this file (not the process's cwd), so the app
# works whether launched as `streamlit run app/streamlit_app.py` from the
# project root or from elsewhere.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
FIG_DIR = DATA_DIR / 'figures'
STATIONS = ['DTU', 'ITO', 'RK_Puram']
MODELS = ['ARIMA', 'Prophet', 'LightGBM']
MODEL_COLORS = {'ARIMA': '#1f77b4', 'Prophet': '#ff7f0e', 'LightGBM': '#2ca02c'}
CORR_LAGS = [0, 6, 12, 24]
CORR_VARS = ['traffic_proxy', 'temperature_c', 'humidity_pct', 'wind_speed_kmh',
             'precipitation_mm', 'pressure_hpa']

st.set_page_config(page_title="UrbanPulse", layout="wide")


def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip('#')
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'


@st.cache_data
def load_features():
    return pd.read_csv(f'{DATA_DIR}/urbanpulse_features.csv', parse_dates=['datetime'])


@st.cache_data
def load_intervals():
    classical = pd.read_csv(f'{DATA_DIR}/classical_forecast_intervals.csv', parse_dates=['datetime'])
    ml = pd.read_csv(f'{DATA_DIR}/ml_forecast_intervals.csv', parse_dates=['datetime'])
    return pd.concat([classical, ml], ignore_index=True)


@st.cache_data
def load_comparison():
    return pd.read_csv(f'{DATA_DIR}/model_comparison_results.csv')


@st.cache_data
def city_wide_series(features_df):
    return (features_df.groupby('datetime')
            .agg(aqi_clean=('aqi_clean', 'mean'),
                 traffic_proxy=('traffic_proxy', 'first'),
                 temperature_c=('temperature_c', 'first'),
                 humidity_pct=('humidity_pct', 'first'),
                 wind_speed_kmh=('wind_speed_kmh', 'first'),
                 precipitation_mm=('precipitation_mm', 'first'),
                 pressure_hpa=('pressure_hpa', 'first'))
            .sort_index())


@st.cache_data
def compute_correlation_heatmap(series_df, variables, lags):
    rows = []
    for var in variables:
        for lag in lags:
            shifted_target = series_df['aqi_clean'].shift(-lag)
            mask = series_df[var].notna() & shifted_target.notna()
            r = series_df.loc[mask, var].corr(shifted_target[mask])
            rows.append({'variable': var, 'lag': lag, 'r': r})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- header ---
st.title("UrbanPulse: Delhi Traffic/Pollution Correlation & Forecasting")
st.caption(
    "A read-only viewer over Phases 1-6 of the analysis (data cleaning, "
    "correlation study, and a 3-model forecasting comparison). Not a live "
    "forecasting tool -- nothing here re-runs a model."
)

st.warning(
    "**Traffic data note:** no real historical Delhi traffic congestion data "
    "was available (TomTom/Google historical APIs are enterprise-gated; no "
    "confirmed open Delhi Traffic Police dataset exists). Every 'traffic' "
    "signal shown in this app (`traffic_proxy`, rush-hour flags) is a "
    "**time-of-day / day-of-week congestion proxy**, not measured traffic. "
    "Any traffic-AQI relationship shown here is a proxy relationship, not "
    "evidence of a measured one -- see README.md > Data decisions."
)

features = load_features()
intervals = load_intervals()
comparison = load_comparison()

# --------------------------------------------------------------- sidebar ---
st.sidebar.header("Filters")
station = st.sidebar.selectbox("Station", STATIONS)

station_intervals = intervals[intervals.station == station]
min_date = station_intervals['datetime'].min().date()
max_date = station_intervals['datetime'].max().date()
date_range = st.sidebar.date_input(
    "Date range (forecast view)",
    value=(min_date, max_date), min_value=min_date, max_value=max_date,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

n_folds = station_intervals['fold'].nunique()
st.sidebar.caption(
    f"{station} has {n_folds} backtested rolling-origin windows (72h each) "
    f"between {min_date} and {max_date} — forecasts only exist inside those "
    f"windows, not for every hour in range."
)

tab1, tab2, tab3 = st.tabs(["Forecast Explorer", "Correlation Explorer", "Model Comparison"])

# ------------------------------------------------------ Forecast Explorer ---
with tab1:
    st.subheader(f"{station}: actual AQI vs. forecasts (80% prediction intervals)")

    quick_cols = st.columns(len(MODELS))
    for col, model in zip(quick_cols, MODELS):
        row = comparison[(comparison.station == station) & (comparison.model == model)
                          & (comparison.segment == 'all') & (comparison.horizon == 24)]
        if not row.empty:
            col.metric(f"{model} — MAE (24h)", f"{row['mae'].iloc[0]:.1f}",
                       help=f"80% interval coverage: {row['coverage_pct'].iloc[0]:.0f}% "
                            f"(target 80%), mean width {row['mean_width'].iloc[0]:.0f}")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(hours=23)

    actual = features[(features.station == station)
                       & (features.datetime >= start_ts) & (features.datetime <= end_ts)]
    sub_iv = station_intervals[(station_intervals.datetime >= start_ts)
                                & (station_intervals.datetime <= end_ts)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual['datetime'], y=actual['aqi_clean'], mode='lines',
                              name='Actual AQI', line=dict(color='black', width=1.5)))

    for model in MODELS:
        model_iv = sub_iv[sub_iv.model == model]
        if model_iv.empty:
            continue
        color = MODEL_COLORS[model]
        first_fold = model_iv['fold'].min()
        for fold, seg in model_iv.groupby('fold'):
            seg = seg.sort_values('step')
            band_x = pd.concat([seg['datetime'], seg['datetime'][::-1]])
            band_y = pd.concat([seg['upper'], seg['lower'][::-1]])
            fig.add_trace(go.Scatter(x=band_x, y=band_y, fill='toself',
                                      fillcolor=hex_to_rgba(color, 0.18),
                                      line=dict(width=0), hoverinfo='skip', showlegend=False))
            fig.add_trace(go.Scatter(x=seg['datetime'], y=seg['pred'], mode='lines',
                                      name=f"{model} (80% band)", legendgroup=model,
                                      showlegend=bool(fold == first_fold),
                                      line=dict(color=color, dash='dash', width=2)))

    fig.update_layout(xaxis_title='Date', yaxis_title='AQI', height=550,
                       legend=dict(orientation='h', yanchor='bottom', y=1.02),
                       margin=dict(t=40))
    st.plotly_chart(fig, width='stretch')

    if sub_iv.empty:
        st.info("No backtested forecast windows fall inside the selected date "
                "range for this station — widen the range to see forecast bands. "
                "The black line above still shows the real observed AQI.")
    else:
        st.caption(
            "Dashed lines are each model's point forecast; shaded bands are its "
            "80% prediction interval. Forecasts appear only inside each model's "
            "72h backtested test windows (see sidebar), so gaps in the dashed "
            "lines are expected, not missing data."
        )

# --------------------------------------------------- Correlation Explorer ---
with tab2:
    st.subheader("Correlation heatmap: traffic proxy & weather vs. AQI (Phase 2's tested lags)")

    scope = st.radio("Scope", ["City-wide (all stations — matches Phase 2)", f"{station} only"],
                      horizontal=True)
    if scope.startswith("City-wide"):
        series = city_wide_series(features)
    else:
        series = (features[features.station == station]
                  .set_index('datetime').sort_index()[['aqi_clean'] + CORR_VARS])

    corr_df = compute_correlation_heatmap(series, CORR_VARS, CORR_LAGS)
    pivot = corr_df.pivot(index='variable', columns='lag', values='r')

    fig2 = px.imshow(pivot, text_auto='.2f', color_continuous_scale='RdBu_r',
                      zmin=-0.6, zmax=0.6, aspect='auto',
                      labels=dict(x='Lag (hours)', y='Variable vs. AQI(t+lag)', color='Pearson r'))
    fig2.update_layout(height=400, margin=dict(t=30))
    st.plotly_chart(fig2, width='stretch')

    st.caption(
        "r = Pearson correlation between the row's variable at time t and AQI "
        "at t+lag. `traffic_proxy`'s weak correlations (|r| ≤ 0.12 in the "
        "city-wide Phase 2 analysis) are a real but modest effect, not zero and "
        "not strong — see README.md > Phase 2 findings for the full discussion "
        "of why this shouldn't be read as evidence of a causal traffic effect."
    )

# ------------------------------------------------------- Model Comparison ---
with tab3:
    st.subheader("Model comparison: MAE / RMSE / 80% interval coverage & width")

    station_scope = st.selectbox("Station scope", STATIONS + ['Overall'],
                                  index=len(STATIONS), key='comparison_station_scope')

    overall_tbl = (comparison[(comparison.segment == 'all') & (comparison.station == station_scope)]
                   [['model', 'horizon', 'mae', 'rmse', 'coverage_pct', 'mean_width', 'n_folds']]
                   .sort_values(['model', 'horizon'])
                   .rename(columns={'model': 'Model', 'horizon': 'Horizon (h)', 'mae': 'MAE',
                                     'rmse': 'RMSE', 'coverage_pct': 'Coverage % (target 80%)',
                                     'mean_width': 'Mean interval width', 'n_folds': 'N folds'}))
    overall_tbl['N folds'] = overall_tbl['N folds'].astype(int)
    st.dataframe(
        overall_tbl.style.format({'MAE': '{:.1f}', 'RMSE': '{:.1f}',
                                    'Coverage % (target 80%)': '{:.1f}',
                                    'Mean interval width': '{:.1f}'}),
        width='stretch', hide_index=True,
    )

    st.markdown("#### Stubble-burning season vs. rest of year (24h horizon)")
    st.caption(
        "Computed Overall (pooled across all 3 stations) only — per-station "
        "sample sizes are too small (5 stubble-season folds total across all "
        "stations) to split further."
    )
    seg_tbl = comparison[(comparison.station == 'Overall')
                          & (comparison.segment.isin(['non_stubble_season', 'stubble_season']))
                          & (comparison.horizon == 24)]
    seg_pivot = (seg_tbl.pivot(index='model', columns='segment', values=['mae', 'n_folds'])
                 [[('mae', 'non_stubble_season'), ('mae', 'stubble_season'),
                   ('n_folds', 'non_stubble_season'), ('n_folds', 'stubble_season')]])
    seg_pivot.columns = ['MAE (rest of year)', 'MAE (stubble season)',
                          'N folds (rest of year)', 'N folds (stubble season)']
    seg_pivot['N folds (rest of year)'] = seg_pivot['N folds (rest of year)'].astype(int)
    seg_pivot['N folds (stubble season)'] = seg_pivot['N folds (stubble season)'].astype(int)
    st.dataframe(seg_pivot.style.format({'MAE (rest of year)': '{:.1f}',
                                           'MAE (stubble season)': '{:.1f}'}),
                 width='stretch')

    st.markdown(
        "**Key finding:** ARIMA and Prophet degrade sharply during the "
        "stubble-burning season (+20.3 and +35.7 AQI MAE) while LightGBM "
        "degrades only slightly (+5.1) — LightGBM is the most robust model "
        "exactly when robustness matters most for an early-warning use case, "
        "despite not winning on overall MAE or interval calibration. See "
        "README.md > Phase 6 results for the full written verdict."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image(f'{FIG_DIR}/model_comparison_coverage_vs_width.png',
                 caption="Coverage vs. width (Overall, all stations)", width='stretch')
    with col2:
        st.image(f'{FIG_DIR}/model_comparison_stubble_season.png',
                 caption="MAE: stubble season vs. rest of year", width='stretch')
