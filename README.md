# UrbanPulse — Delhi Traffic/Pollution Correlation & Forecasting

A data science project on Delhi air quality: (1) does traffic congestion
measurably drive short-term AQI changes, once weather is controlled for,
and (2) can AQI be forecast 24-72h ahead better than a naive baseline
using traffic + weather signals — and if so, which model (ARIMA, Prophet,
or LightGBM) actually earns that job?

**Project status:** complete — all 8 phases (data cleaning through the
written report), a pytest suite, and a post-report follow-up (conformal
calibration for LightGBM's intervals) are done. For the full write-up,
methodology, and honest discussion of limitations, see
**[`report/UrbanPulse_Report.md`](report/UrbanPulse_Report.md)** — this
README covers setup, how to run it, and a summary of what was found.

## Key findings

- **Correlation is real but weak** (|r| ≤ 0.12 at every tested lag) and
  survives controlling for weather without getting stronger — most
  consistent with the traffic proxy's own daily cycle correlating with
  AQI's daily cycle, not a large causal traffic effect.
- **Forecasting:** ARIMA has the best 24h accuracy (MAE 58.7), Prophet
  the best 72h accuracy (MAE 70.5). LightGBM never wins on point
  accuracy, but is dramatically more robust during Delhi's
  stubble-burning season — the highest-stakes period for an
  early-warning use case — degrading only +5.1 MAE vs. ARIMA's +20.3 and
  Prophet's +35.7.
- **Calibration:** ARIMA's 80% prediction intervals are well-calibrated
  (80.5% real coverage). LightGBM's were overconfident (71.0%/65.0% real
  coverage at 24h/72h) until a split-conformal-prediction follow-up
  fixed the 24h case almost exactly (80.6%) and substantially improved
  the 72h case (72.8%, still short of target).
- Full findings, methodology, model verdict, and limitations: see the
  **[report](report/UrbanPulse_Report.md)**.

## The traffic data limitation (read this before the rest)

**No real historical Delhi traffic congestion data exists for this
project's analysis window (Jul 2022 - Dec 2023).** TomTom's and Google's
historical congestion APIs are enterprise/paid-gated for Delhi; no open
Delhi Traffic Police historical dataset was found in two independent
search passes (commercial APIs, then academic/research datasets). Every
"traffic" signal used anywhere in this project — the correlation study,
every forecasting model, the app — is a **synthetic time-of-day /
day-of-week congestion proxy** (`src/build_traffic_proxy.py`), not
measured traffic. This is this project's permanent data source for
traffic, not a placeholder, and it means the correlation/importance
findings above describe how a calendar-based stand-in relates to AQI,
not a measured traffic effect. Full reasoning and the two search
attempts: report Sections 2 and 7.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins exact versions (via `pip freeze`, Python 3.14.3)
rather than `>=` ranges, for reproducibility.

## How to Run Locally

Run in order from the project root:

```bash
python src/reshape_raw.py          # 1. CPCB month-block CSVs -> long format
python src/gap_analysis.py         # 2. missingness diagnostics
python src/clean_window.py         # 3. window filter + missing-data policy
python src/fetch_weather.py        # 4. fetch Open-Meteo weather (needs network; no API key)
python src/merge_weather_aqi.py    # 5. join AQI + weather
python src/build_traffic_proxy.py  # 6. add the synthetic traffic proxy
python src/eda_correlation.py      # 7. correlation study (raw + partial, weather-controlled)
python src/feature_engineering.py  # 8. lag/rolling/calendar features
python src/forecast_classical.py   # 9. ARIMA + Prophet, rolling-origin backtest + 80% intervals
python src/forecast_ml.py          # 10. LightGBM, same folds + quantile intervals
python src/model_comparison.py     # 11. MAE/RMSE/coverage/width comparison + stubble-season analysis
python src/calibrate_intervals.py  # 12. split conformal calibration for LightGBM's intervals
```

Each script prints a summary and writes its output to `data/processed/`
(figures to `data/processed/figures/`). Re-running a script is safe — it
overwrites its own output.

## Running the Viewer App

Once the pipeline above has been run at least once:

```bash
streamlit run app/streamlit_app.py
```

A read-only viewer (it never re-runs a model) with three tabs: a
Forecast Explorer (actual AQI vs. all three models' forecasts and 80%
intervals), a Correlation Explorer (live traffic/weather-vs-AQI
heatmap), and a Model Comparison tab (MAE/RMSE/coverage/width +
stubble-season breakdown). The traffic-proxy limitation is shown as a
permanent banner on every tab.

## Testing

```bash
pytest tests/ -v
```

Ten fast tests (<1s, synthetic inline fixtures — no large files touched)
covering the correctness claims that were previously only checked by
eyeballing script output:

- **`test_reshape_raw.py`** — invalid calendar dates (e.g. "Feb 30") are
  skipped cleanly, not crashed or silently misdated.
- **`test_feature_engineering.py`** — the highest-value test: proves a
  row's rolling stats never leak its own value, and a station's history
  never leaks into the next station's early rows.
- **`test_merge_pipeline.py`** — the AQI-weather join doesn't
  duplicate/drop rows, and a missing weather timestamp is caught as NaN,
  not silently mismatched.
- **`test_traffic_proxy.py`** — weekday rush-hour scores exceed weekend
  scores, and all scores stay within [0, 1].

All ten pass against the existing pipeline — no bugs were found.

## Project structure

```
urbanpulse/
├── app/
│   └── streamlit_app.py       # read-only viewer over the full analysis
├── report/
│   └── UrbanPulse_Report.md   # full write-up: methodology, findings, limitations, verdict
├── data/
│   ├── raw/                   # untouched downloads (gitignored)
│   └── processed/             # cleaned/joined/feature-engineered outputs (gitignored)
│       └── figures/           # plots (tracked — small and worth seeing directly)
├── src/
│   ├── reshape_raw.py          # CPCB month-block CSV -> long format
│   ├── gap_analysis.py         # missingness diagnostics
│   ├── clean_window.py         # window filter + missing-data policy
│   ├── fetch_weather.py        # Open-Meteo historical weather pull
│   ├── merge_weather_aqi.py    # join AQI + weather -> master dataset
│   ├── build_traffic_proxy.py  # synthetic time-of-day/day-of-week traffic proxy
│   ├── eda_correlation.py      # STL decomposition + lag/partial correlation
│   ├── feature_engineering.py  # per-station lag/rolling + calendar features
│   ├── forecast_classical.py   # ARIMA + Prophet, rolling-origin backtesting + 80% intervals
│   ├── forecast_ml.py          # LightGBM, same folds/horizons + 80% quantile intervals
│   ├── model_comparison.py     # MAE/RMSE/coverage/width consolidation + error analysis
│   └── calibrate_intervals.py  # split conformal calibration for LightGBM's intervals
├── tests/
│   ├── conftest.py
│   ├── test_reshape_raw.py
│   ├── test_feature_engineering.py
│   ├── test_merge_pipeline.py
│   └── test_traffic_proxy.py
├── requirements.txt
└── README.md
```

## Further reading

- **[report/UrbanPulse_Report.md](report/UrbanPulse_Report.md)** — the
  full write-up: problem statement, data, methodology, findings, model
  verdict, limitations, and future work.
- **[UrbanPulse_Spec.md](UrbanPulse_Spec.md)** — the original project
  spec this build followed.
