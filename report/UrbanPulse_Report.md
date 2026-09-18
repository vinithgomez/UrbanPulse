# UrbanPulse: Delhi Traffic/Pollution Correlation & Forecasting

*A data science portfolio project. Full pipeline, code, and per-phase decision log: `README.md`. This report synthesizes that work into a single narrative.*

---

## 1. Problem Statement

This project answers two questions about air quality in Delhi:

1. **Is there a measurable relationship between traffic congestion and short-term AQI changes, and does it survive controlling for weather?** AQI is heavily weather-driven (wind, humidity, temperature inversion), so a raw traffic-AQI correlation on its own would be close to meaningless — the real question is whether a traffic effect remains once weather is accounted for.
2. **Can AQI be forecast 24-72 hours ahead better than a naive baseline, using traffic and weather signals?** And if forecasting is possible, which modeling approach — classical statistical (ARIMA, Prophet) or gradient-boosted ML (LightGBM) — is actually the better choice, and for which use case?

Both questions turn out to have honest, somewhat unglamorous answers, which is the point: this report states what the data supports and what it doesn't, rather than reaching for a cleaner story than the evidence justifies.

---

## 2. Data

**Sources:**

- **AQI** — OpenCity's pre-scraped CPCB (Central Pollution Control Board) dataset (`data.opencity.in`), hourly readings from three Delhi monitoring stations: ITO, R K Puram, and DTU.
- **Weather** — Open-Meteo's Historical Weather Archive, a free, keyless hourly source, used in preference to OpenWeatherMap because OpenWeatherMap's free tier excludes historical data entirely.
- **Traffic** — no real data. This is stated here, in the second section of the report, deliberately, not left for a limitations footnote.

**The traffic proxy, stated plainly:** no accessible historical Delhi traffic congestion dataset exists. TomTom's and Google's historical congestion APIs are enterprise-gated; no confirmed open Delhi Traffic Police historical dataset was found. Every "traffic" signal used anywhere in this project — in the correlation study, in every forecasting model, in the app — is a **synthetic time-of-day/day-of-week congestion proxy**: a continuous 0-1 score built from well-documented Delhi rush-hour patterns (AM peak roughly 8-10, PM peak roughly 6-9, a flatter weekend profile), plus a simple binary rush-hour flag as an alternative feature. It is not measured congestion. This is the single most important fact to hold in mind when reading every result in Section 4 that mentions "traffic" — a weak correlation between this proxy and AQI is not evidence that real traffic has a weak effect on AQI; it is evidence about the relationship between a deterministic calendar function and AQI, which happen to be constructed similarly (both have daily/weekly cycles) but are not the same claim.

**Analysis window:** 2022-07-01 to 2023-12-31 (18 months), not the full 2017-2023 range the raw data covers. 2017-2018 had severe, non-random missingness — R K Puram was offline 47 consecutive days in mid-2017 — that would bias any correlation or backtesting result. The chosen window has stable 5-9% missingness across all three stations with no gap longer than about 7 days.

**Missing-data handling:** gaps of 3 consecutive hours or less are forward-filled; longer gaps are left as `NaN` and explicitly flagged rather than silently interpolated. This is a "flag, don't paper over" policy applied consistently through the project: Phase 3's engineered lag/rolling features inherit this missingness (8.8% of rows end up with at least one incomplete lag/rolling feature, mostly propagated from the underlying AQI gaps rather than simple feature-warmup), and the forecasting phases resolve that by **dropping** affected rows per station rather than imputing — 4.9% of rows for DTU, 10.7% for ITO, 11.4% for R K Puram. Imputing the modeling target itself before backtesting would have let a model "predict" values manufactured from their own neighbors, a leakage risk judged worse than losing the rows outright.

---

## 3. Methodology

**Correlation approach.** Cross-correlation of the traffic proxy against city-wide AQI (mean across the three stations) was computed at lags of 0, 6, 12, and 24 hours, both raw and *partial* — i.e., after residualizing both the traffic proxy and the lagged AQI against the five weather variables (temperature, humidity, wind speed, precipitation, pressure) via OLS and correlating the residuals. This partial-correlation step is what actually tests whether a traffic effect survives controlling for the dominant confound; a raw correlation alone cannot distinguish "traffic causes AQI changes" from "traffic and AQI both happen to follow the weather."

**Feature engineering.** Per station (never leaking across station boundaries), lag features at t-1/t-6/t-24 hours and rolling 6h/24h mean and standard deviation of AQI, computed strictly on past data (a row's rolling stats never include its own value). Calendar features — hour, day-of-week, a 26-holiday flag for the analysis window, and a stubble-burning-season flag (mid-October to mid-November, both years, per satellite/CPCB reporting on Punjab/Haryana crop-residue burning) — are deterministic and knowable arbitrarily far in advance, unlike weather.

**Three forecasting models**, deliberately scoped to exactly these three (no LSTM, no ensemble, per the project's own scope discipline):

- **ARIMA**, order chosen per station via an AIC grid search after an ADF stationarity test determined the differencing order.
- **Prophet**, a univariate second baseline.
- **LightGBM**, a direct multi-horizon gradient-boosted model using horizon itself as a feature, trained on lag/rolling AQI features plus the deterministic calendar/traffic features above.

**A leakage rule applied consistently across all three models:** none of them use weather as a forecast input, even though weather is available and predictive. Using *actual* historical weather as a "known" input for a forecast 24-72 hours out would implicitly assume a perfect weather forecast, which a real deployed system would not have — and would also make the model comparison unfair, since it would hand LightGBM information the classical baselines don't get. All three models are evaluated on the same information budget: past AQI and its derived features, plus deterministic calendar/traffic signals.

**Backtesting: rolling-origin, not a single train/test split, and this choice matters.** A single train/test split tests a model on one arbitrary slice of time and says nothing about how consistently it performs across different seasons, weather regimes, or data volumes. This project instead uses an expanding-window scheme: starting from an initial 180-day training window, the origin advances by 30 days each fold, and each fold forecasts 72 hours ahead — producing 11-12 independent folds per station spread across the full 18-month window. MAE and RMSE are reported at both the 24-hour mark and the full 72-hour horizon, since blending them into one number would hide how quickly accuracy degrades with horizon length. This design is also what makes the stubble-burning-season finding in Section 4e possible at all: a single train/test split would have given, at best, one data point on how a model handles that period, not five independent test windows to look for a pattern across.

---

## 4. Findings

### 4a. Correlation: weak, oscillating, and not causal evidence

Cross-correlation of the traffic proxy against city-wide AQI, at lags of 0/6/12/24 hours:

| lag (h) | r (raw) | 95% CI (raw) | r (partial, weather-controlled) | 95% CI (partial) |
|---:|---:|---|---:|---|
| 0  | -0.104 | [-0.121, -0.087] | -0.106 | [-0.123, -0.089] |
| 6  | -0.033 | [-0.050, -0.016] |  0.013 | [-0.005, 0.030]  |
| 12 |  0.121 | [ 0.104,  0.138] |  0.092 | [ 0.075, 0.109]  |
| 24 | -0.105 | [-0.122, -0.088] | -0.101 | [-0.118, -0.084] |

All four are statistically significant (n ≈ 13,150-13,170, so even small effect sizes reach very low p-values), but the effect sizes are weak (the largest magnitude in the table above is r=0.121, at the 12h lag) and the sign flips by lag, including going negative at both lag 0 and lag 24. A wider cross-correlation function out to ±48 hours shows this pattern is periodic, not noise — it tracks the traffic proxy's own daily cycle correlating with AQI's own daily cycle, which is close to guaranteed given both are built from time-of-day.

Controlling for weather barely moves any of the four correlations. That is worth reading carefully, because it is the *opposite* of the intuitive story: it is not that "the raw correlation was confounded by weather and mostly disappears once you control for it" — the correlation was never large to begin with, and it survives weather-controlling almost unchanged. The honest interpretation is that the traffic *proxy* — a deterministic hour-of-day/day-of-week function — has a real but small independent daily-cycle relationship with AQI, on top of whatever weather is doing. Given that this proxy is a stated stand-in for real congestion data, not a measurement of it, this result should not be read as evidence about real traffic's effect on Delhi's AQI one way or the other. It is evidence about how a synthetic calendar-based signal relates to AQI, which is a narrower and less interesting claim.

### 4b. Forecasting: ARIMA best short-horizon, all three converge by 72h

Rolling-origin backtest, mean MAE/RMSE across 11-12 folds per station:

| model | MAE (24h) | RMSE (24h) | MAE (72h) | RMSE (72h) |
|---|---:|---:|---:|---:|
| ARIMA | 58.67 | 73.31 | 71.46 | 86.23 |
| Prophet | 65.29 | 78.65 | 70.54 | 85.90 |

ARIMA beats Prophet at the 24-hour horizon at every one of the three stations. By 72 hours the two are close enough that Prophet edges ahead very slightly overall. Both errors are large relative to typical AQI swings, and the reason becomes clear when looking at individual forecast plots: both models converge to a **near-flat forecast within about a day** — ARIMA reverts to its short-run mean, Prophet falls back on its learned daily/weekly seasonal cycle — and both miss the sharp, short-lived AQI spikes visible in the real series (one DTU fold shows the actual series spiking to roughly 330 while neither model comes within 100 points of it). This flat-forecast failure mode is exactly what motivated building a third, feature-based model.

(One data artifact worth flagging so it isn't mistaken for a modeling bug: one DTU fold's actual AQI series flatlines at exactly 500 for several consecutive hours. CPCB's AQI index is capped at 500 — this is a real reporting ceiling in the raw data, not an error introduced by this pipeline.)

### 4c. LightGBM: doesn't win on MAE, does reduce the flat-forecast problem

| model | MAE (24h) | RMSE (24h) | MAE (72h) | RMSE (72h) |
|---|---:|---:|---:|---:|
| ARIMA | 58.67 | 73.31 | 71.46 | 86.23 |
| LightGBM | 62.42 | 76.03 | 73.69 | 89.12 |
| Prophet | 65.29 | 78.65 | 70.54 | 85.90 |

LightGBM's position depends on which horizon and which baseline: at 24h it **beats Prophet but trails ARIMA** (62.42 vs. Prophet's 65.29 and ARIMA's 58.67), while at 72h it is **the worst of the three** (73.69 vs. ARIMA's 71.46 and Prophet's 70.54). Put simply, it never beats ARIMA at either horizon, and only edges out Prophet at the shorter one — despite having direct access to recent AQI lags/rolling statistics and calendar/traffic signals that Prophet doesn't use. What it does do, visible in the forecast plots rather than the summary table, is track the daily up/down oscillation more dynamically instead of collapsing to a flat line — it still substantially under- and over-shoots the sharp spikes and troughs in the actual series, so the core failure mode from 4b is reduced, not solved.

Feature importance (gain-based, averaged across folds) shows why the traffic proxy doesn't move the needle much here either. It ranks mid-pack at every station — 6th of 15 features at DTU, 9th at ITO, 5th at R K Puram — never top, never bottom:

| station | traffic_proxy rank (of 15) | top feature |
|---|---:|---|
| DTU | 6 | `aqi_rolling_mean_24h` |
| ITO | 9 | `aqi_rolling_mean_24h` |
| R K Puram | 5 | `aqi_rolling_mean_24h` |

`aqi_rolling_mean_24h` dominates every station by roughly an order of magnitude over the next-highest feature, consistent with AQI's strong day-to-day persistence (also visible in the Phase 2 time-series decomposition). `hour` and `day_of_week` outrank `traffic_proxy` at two of the three stations, and the coarser binary flags (`is_weekend`, `is_rush_hour`) rank at or near zero — the model appears to extract what it needs about the daily/weekly cycle from the continuous `traffic_proxy` and the raw calendar fields directly, leaving the blunter binary versions redundant. `traffic_proxy` earning a real, modest, mid-pack place — not dismissed, not dominant — is the same finding as 4a arrived at independently through a completely different model and method, which is reassuring: two different techniques applied to the same weak-but-real signal reached the same conclusion about its size.

### 4d. Calibration: ARIMA well-calibrated, LightGBM overconfident

MAE/RMSE alone don't tell you whether a model's *stated uncertainty* can be trusted, which matters as much as point accuracy for any real early-warning use — a forecast that says "AQI will be 200, ±20" is a different, more useful product than one that says "AQI will be 200" with no stated confidence, but only if the ±20 is actually close to right roughly as often as claimed. All three models were given 80% prediction intervals (ARIMA via its confidence interval, Prophet via its native interval, LightGBM via two additive quantile-regression models at the 10th and 90th percentiles), and each interval's real-world coverage was checked: what fraction of actual AQI values actually fell inside the stated 80% band.

| model | horizon | Coverage (target 80%) | Mean interval width |
|---|---:|---:|---:|
| ARIMA | 24h | **80.5%** | 193.6 |
| ARIMA | 72h | 77.1% | 215.9 |
| Prophet | 24h | 78.4% | 205.1 |
| Prophet | 72h | 76.9% | 205.3 |
| LightGBM | 24h | 71.0% | **163.5** |
| LightGBM | 72h | 65.0% | **171.8** |

ARIMA's interval is essentially exactly calibrated at 24 hours and only mildly overconfident by 72 hours. Prophet trails ARIMA slightly at both horizons but is still close to the 80% target. **LightGBM is the outlier, and in the wrong direction to be reassuring:** its intervals are the narrowest of the three — roughly 15-20% tighter than ARIMA's or Prophet's — but its actual coverage is the worst, at 71.0% and 65.0% against an 80% target. A narrower interval that is wrong more often is not a better interval; it is a worse one wearing a more confident label. In practice, LightGBM's stated "80% interval" behaves like a true ~65-71% interval. For any downstream use where someone — a health advisory system, a person deciding whether to keep children indoors — trusts the stated confidence level, this gap between claimed and actual coverage is a real risk, not a cosmetic statistic: it means LightGBM's forecasts will be surprised by out-of-band AQI values noticeably more often than its own uncertainty estimate promises.

**Update — this has since been fixed, partially.** Split conformal calibration (Conformalized Quantile Regression) was applied per fold, holding out the last 20% of each fold's training period to derive a calibration adjustment (see `README.md` > Data decisions for the method). Result: 24h coverage rose from 71.0% to **80.6%** — landing almost exactly on target — while 72h coverage rose from 65.0% to **72.8%**, a real improvement that still falls short of 80%. The gap at 72h is a direct, disclosed consequence of using one global adjustment per fold across all horizons rather than a horizon-specific one (see Section 7). The cost was real too: mean interval width grew by roughly 28-29%. Full before/after detail: `data/processed/ml_forecast_results_calibrated.csv`.

### 4e. Stubble-burning season: LightGBM's real advantage, with an honest caveat

The project spec flagged Delhi's stubble-burning season (mid-October to mid-November, when Punjab/Haryana crop-residue burning drives some of the city's worst AQI spikes) as a likely failure window for classical time-series models, and hypothesized that a feature-based ML model might handle it better. This was tested directly by splitting the 34 backtest folds into 5 that overlap the stubble-burning window and 29 that don't, and comparing 24-hour MAE:

| model | non-stubble MAE | stubble MAE | degradation |
|---|---:|---:|---:|
| ARIMA | 55.7 | 76.0 | +20.3 |
| Prophet | 60.0 | 95.7 | **+35.7** |
| LightGBM | 61.7 | 66.8 | **+5.1** |

The hypothesis holds, and holds cleanly: both classical models degrade sharply during stubble season — ARIMA by 20.3 AQI points, Prophet by 35.7 — while LightGBM's degradation is a fraction of that, at 5.1 points. LightGBM is the most robust of the three models specifically during the highest-stakes period for a public-health early-warning use case, despite not winning on overall MAE (4c) or on interval calibration (4d).

Two things need to be said plainly rather than left implicit. First, **n=5**: only 5 of the 34 total folds overlap the stubble-burning window, because folds are spaced 30 days apart against a roughly 32-day window per year. This is a real, directionally consistent pattern — not a coin-flip result, given the 4-7x difference in degradation size is large relative to what 5 folds of noise would typically produce — but it is not a large-sample finding, and a future version of this project with more stubble seasons of data (see Section 7) would make it a much stronger claim than it currently is. Second, the single worst individual fold for both ARIMA and LightGBM occurred in February 2023, not during stubble season — a separate winter AQI spike that neither model saw coming, which was originally left as an open question in this report and has since been investigated (below).

**The February 2023 spike, investigated.** The single worst fold overall — both ARIMA's and LightGBM's #1 worst by 24h MAE — is ITO's test window from 2023-02-15 22:00 to 2023-02-18 21:00. External reporting describes February 2023 as Delhi's second-highest February AQI on record (~237 mean), attributed to low winter rainfall, calm winds, and temperature-inversion trapping — mechanically the same low-wind/inversion story as stubble season, just without stubble burning's contribution. Checking this project's *own* `wind_speed_kmh` and `precipitation_mm` for that exact window against the rest of the same month: mean wind speed was 6.6 km/h during the spike window versus 11.0 km/h for the rest of February 2023 (roughly 40% calmer), and the fraction of hours with wind below 5 km/h nearly quadrupled (26% versus 7%); humidity was also notably higher (75.6% versus 61.4%), consistent with the stagnant, moist air a temperature inversion produces. The low-wind/stagnant-air mechanism holds up clearly.

The rainfall half of the explanation does not, however — precipitation was zero for the *entire* month of February 2023, not specifically during this window, so "low rainfall" explains why February 2023 as a whole was a bad month (consistent with the external year-over-year comparison), but it does not explain why *this specific 3-day window* was the single worst forecasting failure within an already-dry month. **Verdict: the external explanation is partially confirmed, not fully** — real in its wind/inversion component, not a differentiator in its rainfall component for this specific window. What caused the wind to calm further during exactly this window remains open, but it's now a narrower and more specific question than the original "what caused the February spike," and answering it further would need synoptic weather data this project doesn't have.

---

## 5. Model Verdict

No single model wins outright, and the honest synthesis is that each has earned a genuinely different, narrower role rather than a ranked 1-2-3:

- **ARIMA** is the right default for a plain 24-hour point forecast with a trustworthy uncertainty band. It has the best 24h MAE of the three and, more surprisingly, the best-calibrated prediction interval — close to its stated 80% coverage at both horizons.
- **Prophet** is a reasonable second choice: it trails both other models at 24h, but is actually the best of the three on 72h MAE (70.54 vs. ARIMA's 71.46 and LightGBM's 73.69), and its interval calibration stays close to ARIMA's at both horizons. It's specifically worth keeping for the 72-hour outlook, where it's not just competitive but the top performer on point accuracy.
- **LightGBM's value is invisible if you only look at the headline MAE table.** It loses on point accuracy and, before calibration, on interval calibration too — but it is the most robust model exactly during the stubble-burning season, the period that matters most for a real early-warning product. Since Section 4d's writing, its interval has been recalibrated via split conformal prediction: 24h coverage now lands almost exactly on target (80.6%), and 72h coverage improves substantially though not completely (72.8% against 80%). That makes it a credible *specialist* choice for the stubble-burning window with its 24h interval now trustworthy as stated, and its 72h interval better but still somewhat optimistic — a real, mostly-closed gap rather than a hypothetical one to fix later.

The practical recommendation: **ARIMA as the default 24-hour forecast and interval, Prophet for the 72-hour outlook, LightGBM reserved for stubble-season alerting now that its 24h interval is recalibrated and trustworthy (its 72h interval is improved but should still be treated with some caution until it's brought fully to target).**

---

## 6. Limitations

Ranked by how much they should change a reader's confidence in this project's conclusions, most important first.

1. **The traffic proxy is not real traffic data.** This is restated here because it is the limitation that most directly bears on the project's first research question. Every traffic-related number in this report — the weak correlation in 4a, the mid-pack feature importance in 4c — describes the relationship between a synthetic time-of-day/day-of-week function and AQI, not the relationship between measured Delhi traffic congestion and AQI. It is a reasonable, disclosed, and well-precedented substitution when real data is genuinely unavailable, but it means this project cannot answer "does real traffic congestion affect Delhi's AQI" — only "does a plausible calendar-based stand-in for it show a measurable relationship," which it does, weakly.
2. **Remaining AQI data gaps.** Even after the Phase 1 missing-data policy, DTU is missing 2.0% of hours, R K Puram 4.3%, and ITO 6.8% post-cleaning; this compounds into an 8.8%-of-rows incompleteness in the engineered lag/rolling features (Phase 3), and ultimately into the 4.9-11.4% of rows dropped entirely before any model fitting (Phase 4/5). None of this is hidden — every number above is reported and the dropping decision is explained — but a reader should know the forecasting models were trained and evaluated on 89-95% of each station's hours, not all of them, and that dropping (rather than imputing) the gaps means the backtested series has real, sometimes multi-week, discontinuities in elapsed time that the position-indexed models (ARIMA, LightGBM) do not "see."
3. **Small stubble-season sample.** Section 4e's finding — LightGBM's relative robustness during stubble season — rests on 5 backtest folds. The direction and size of the effect are consistent with a real pattern, not noise, but this is not a large-n result and should be weighted accordingly against the classical models' larger-sample MAE and calibration advantages elsewhere.
4. **One weather series for three stations.** Weather (temperature, humidity, wind speed, precipitation, pressure) is a single Delhi-wide hourly series from one central coordinate, applied identically to all three AQI stations, rather than three station-specific series. AQI's weather sensitivity is driven by broad meteorology rather than hyper-local conditions a few kilometers apart, so this is a reasonable simplification — but it does mean any station-to-station difference in results (e.g., ITO's feature-importance rank for `traffic_proxy` differing from DTU's) cannot be attributed to a real weather difference between stations, because there isn't one in this dataset.

---

## 7. Future Work

Three of this section's original four items have since been closed out —
each is reported here as what was actually found, not as a restated plan.

### Closed since the initial report

1. **Conformal calibration for LightGBM's intervals — done, partially successful.** Split conformal prediction (Conformalized Quantile Regression) was applied per fold: the last 20% of each fold's training period was held out as a calibration set, fresh quantile models were fit on the remaining 80%, and a single scalar adjustment was derived from the calibration set's nonconformity scores and applied to that fold's real test-period interval. Result: 24h coverage rose from 71.0% to 80.6% (the target essentially hit), and 72h coverage rose from 65.0% to 72.8% (substantially better, but still short of 80%). The 72h shortfall is a direct, disclosed consequence of using one global adjustment per fold across all horizons rather than a separate adjustment per horizon — the simpler design choice, not a bug, and a legitimate next refinement if 72h calibration needs to be closed further. The fix wasn't free: mean interval width grew by roughly 28-29%, confirming that LightGBM's original narrow interval wasn't actually cheaper, just wrong more often than it claimed.
2. **Real traffic data — investigated twice, closed.** A first pass ruled out the obvious commercial options (TomTom/Google, both enterprise-gated for Delhi). A second, independent pass checked academic and research-oriented sources specifically: a Nature Scientific Data paper confirms TomTom's Delhi coverage is paid-only; a relevant ESSD 2023 paper models Delhi traffic, but for 2018 (outside this project's window) and as modeled rather than raw congestion values; and the only two open-ish Delhi traffic datasets found — a "Delhi Traffic Probe and Analytics 2024" dataset and a Zenodo 2024 rush-hour dataset — both fall entirely in 2024, after this project's window ends. **The synthetic time-of-day/day-of-week proxy is this project's permanent traffic data source, not a provisional one.** If a dataset covering 2022-2023 specifically surfaces later, re-running the Phase 2 correlation study and Phase 5's LightGBM feature set with it would still be worthwhile — but that is a hypothetical prompted by a future data availability change, not an open task this project is currently waiting on.
3. **The February 2023 spike — investigated, partially explained.** See Section 4e for the full finding: this project's own weather data confirms the low-wind/stagnant-air half of the external explanation (wind was ~40% calmer during the spike window than the rest of the same month, with calm hours roughly 3-4x more frequent), but not the low-rainfall half, since precipitation was zero for the entire month, not distinctively so during the spike window. What remains open is narrower than originally framed: not "why did February 2023 spike" (reasonably well explained by low wind + inversion, consistent with external reporting) but "why did wind calm specifically during this 3-day window within an already-dry month" — a question this project's data can't answer further without additional synoptic weather data it doesn't have.

### Still open

4. **More stubble seasons of data.** The current dataset covers exactly two stubble-burning seasons (Oct-Nov 2022 and 2023), yielding 5 usable backtest folds. Extending the analysis window backward or forward to include additional years would directly address the small-sample caveat in Sections 4e/6.3 and turn a credible-but-thin finding into a well-powered one. This is the one item from the original four that genuinely requires data this project doesn't have and can't generate from what it already has — unlike the three above, there's no further investigation of the *existing* dataset that would resolve it.

---

*Full code, per-phase decision log, and every intermediate CSV referenced above: see `README.md` and `data/processed/` in the project repository.*
