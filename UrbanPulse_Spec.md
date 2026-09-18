# UrbanPulse — Delhi Traffic/Pollution Correlation Analysis & Forecasting

**Type:** Data Science portfolio project
**Goal:** Demonstrate statistical rigor + time-series forecasting depth (not just "built a dashboard")

---

## 1. Problem Statement

Does traffic congestion in Delhi measurably drive short-term air quality (AQI) changes, and can we forecast AQI 24–72 hours ahead using traffic + weather signals better than a naive baseline?

Two deliverables come out of this:
1. A **correlation study** — is the traffic→AQI relationship real, and how strong/lagged is it?
2. A **forecasting model comparison** — classical statistical (ARIMA/Prophet) vs. ML (LightGBM) for short-term AQI prediction.

---

## 2. Data Sources

| Data | Source | Notes |
|---|---|---|
| AQI (hourly) | CPCB (Central Pollution Control Board) API / data.gov.in | Multiple Delhi monitoring stations — pick 2–3 (e.g. Anand Vihar, ITO, RK Puram) for cross-station comparison |
| Traffic | Google/TomTom historical congestion data or Delhi Traffic Police open data (if available) — fallback: use OpenStreetMap + proxy via known peak-hour patterns | Traffic data is the hardest to get cleanly — flagged as the first risk to validate before committing |
| Weather | OpenWeatherMap historical API or IMD (India Meteorological Dept) | Needed as a confound — AQI is heavily weather-driven (wind speed, humidity, temperature inversion), so any traffic correlation claim is meaningless without controlling for weather |

**Action item before writing any analysis code:** verify Delhi traffic data is actually obtainable at the granularity/timespan needed. If real traffic data isn't accessible, the honest fallback is to use **time-of-day/day-of-week as a traffic proxy** (rush hour patterns) — this is a legitimate, defensible substitution as long as it's stated explicitly in the report, not hidden.

---

## 3. Project Structure

```
urbanpulse/
├── data/
│   ├── raw/                  # untouched downloads
│   └── processed/            # cleaned, joined, feature-engineered
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda_and_correlation.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_forecasting_classical.ipynb   # ARIMA/Prophet
│   ├── 05_forecasting_ml.ipynb          # LightGBM
│   └── 06_model_comparison.ipynb
├── src/
│   ├── data_loader.py
│   ├── features.py
│   └── models.py
├── app/
│   └── streamlit_app.py       # viewer dashboard
├── report/
│   └── UrbanPulse_Report.pdf  # written findings
├── requirements.txt
└── README.md
```

---

## 4. Phase-by-Phase Plan

### Phase 1 — Data Collection & Cleaning
- Pull AQI data for 2–3 Delhi stations, 12–18 months of hourly data
- Pull weather data for the same window
- Pull/construct traffic signal (real data or time-of-day proxy — decide per Section 2)
- Handle missing data (documented strategy: forward-fill for short gaps, flag longer gaps rather than silently interpolating)

### Phase 2 — EDA & Correlation Analysis
- Time-series decomposition (trend/seasonality/residual) on AQI
- Cross-correlation analysis between traffic proxy and AQI at multiple lags (0h, 6h, 12h, 24h) — this answers "does traffic predict AQI, and with what delay?"
- **Hypothesis test:** is the traffic-AQI correlation statistically significant after controlling for weather? (partial correlation or a simple regression with weather as covariates)
- Report correlation coefficients **with confidence intervals**, not just point estimates

### Phase 3 — Feature Engineering
- Lag features (AQI_t-1, AQI_t-6, AQI_t-24)
- Rolling window stats (6h/24h rolling mean, std)
- Weather joins (temperature, humidity, wind speed)
- Calendar features (hour, day-of-week, holiday flag)

### Phase 4 — Forecasting: Classical Approach
- ARIMA (with proper stationarity testing — ADF test, differencing as needed)
- Prophet as a second classical baseline (handles seasonality well, easier to defend in interviews as "why Prophet over pure ARIMA")
- Backtesting via rolling-origin cross-validation (not a single train/test split — this is the #1 thing that separates a rigorous forecasting project from a toy one)

### Phase 5 — Forecasting: ML Approach
- LightGBM with the engineered lag/rolling/weather features
- Same rolling-origin backtesting for fair comparison
- Feature importance analysis (which lag/weather feature matters most — good interview talking point)

### Phase 6 — Model Comparison
- Metrics: MAE, RMSE, and **prediction interval coverage** (not just point accuracy — show you understand forecasting isn't just point predictions)
- Error analysis: where does each model fail? (e.g., ARIMA likely struggles on sudden pollution spikes from external events like stubble burning season; LightGBM likely better there but worse at long-horizon smooth trends)
- Written verdict: which model for which use case, and why

### Phase 7 — Streamlit Viewer
- Station selector, date range picker
- Actual vs. predicted AQI chart with confidence bands
- Correlation heatmap (traffic/weather/AQI lags)
- Model comparison table

### Phase 8 — Report
- 4–6 page written report: problem, data, methodology, findings, limitations
- Explicitly state the traffic-data limitation if the proxy fallback was used — a self-aware limitations section reads as more credible than pretending the data was perfect

---

## 5. Key Interview Talking Points This Project Builds

- Why rolling-origin backtesting instead of a single train/test split
- Why partial correlation was needed (confound control) rather than raw correlation
- ARIMA vs Prophet vs LightGBM trade-offs — interpretability vs accuracy vs handling of external shocks
- How you validated the traffic data question before committing to the full build (shows engineering judgment, not just "I found a tutorial and followed it")
- What prediction intervals mean and why point forecasts alone are misleading

---

## 6. Risks / Open Decisions

1. **Traffic data availability** — needs to be verified first; determines whether Phase 1 uses real data or the time-of-day proxy (stated openly either way).
2. **CPCB API access** — may require registration; data.gov.in is the fallback if CPCB's own API is unavailable.
3. **Scope creep** — it's tempting to add a third model (LSTM) or extra cities; recommend resisting this to keep the depth-over-breadth story intact.

---

## 7. Estimated Build Time

- Data collection/cleaning: 2–3 days (mostly data-source wrangling)
- EDA/correlation: 1–2 days
- Forecasting (both approaches + backtesting): 3–4 days
- Streamlit app: 1 day
- Report writing: 1 day

**Total: ~8–11 days** of focused work.
