# Results summary for the report writer

**Project:** Carbon-aware scheduling of data-centre workloads on the Irish grid (H9ETS CA, NCI, 2026)
**Research question:** Can official weather forecasts improve 24-hour-ahead CO₂ intensity prediction for Ireland's grid,
and how much CO₂ can data centres avoid by shifting flexible workloads to predicted low-carbon hours, once the
forecaster's own training energy is counted?

All numbers below come from the files in `results/` or from the printed outputs of `notebooks/01_eda.ipynb`.
Code: `src/` (download.py, build_dataset.py, features.py, train.py, scheduler.py).

---

## 1. Dataset

| Item | Value |
|---|---|
| Sources | EirGrid Smart Grid Dashboard (ROI): CO₂ intensity, CO₂ emissions, wind, demand, solar (15-min → hourly mean); Open-Meteo Previous Model Runs API: 100 m wind speed at 4 wind-farm regions (Donegal, Mayo, Kerry, Cork), hourly |
| Period (UTC) | 1 Jun 2024 → 2 Nov 2026 |
| Rows | 21,241 hourly rows, 21 columns (`data/processed/clean_hourly.csv`) |
| Target missing | CO₂ intensity missing for 3.34% of hours |
| Filled values | 2,409 hours of CO₂ intensity derived as emissions ÷ demand × 1000 (flag column `co2_intensity_filled`) |
| Other missing | demand 2.18%, wind 0.03%, CO₂ emissions 22.27%, solar 39.97% (solar only reported from May 2025), Open-Meteo 0% |
| Unrecoverable days | 16 days of CO₂ data and 19 days of demand returned empty from EirGrid after repeated retries |

**Summary statistics** (`results/summary_stats.csv`): CO₂ intensity mean 195.3 g/kWh (SD 63.9, min 21.7, max 471.3);
wind generation mean 1,275 MW; demand mean 3,934 MW; wind share of demand mean 32%; 24 h-ahead forecast wind
speed (4-site mean) 7.98 m/s.

**Cleaning steps**
1. Downloaded one day per request from EirGrid (CO₂ chart + grid chart), cached as JSON; retried failed days.
2. For 189 days where the grid chart omitted demand, demand was fetched from EirGrid's separate demand chart.
3. Converted Europe/Dublin local time to UTC, handling the October repeated hour and March missing hour.
4. Averaged 15-min values to hourly.
5. Filled missing CO₂ intensity from emissions ÷ demand (flagged).
6. Merged Open-Meteo forecasts; added calendar features (hour, weekday, month, Irish public holiday) and wind share.

## 2. Exploratory findings (notebook `01_eda.ipynb`)

- CO₂ intensity vs wind share: Pearson r = −0.75 (n = 20,123 hours).
- 24 h-ahead wind forecast vs realised wind (4-site mean): MAE 0.77 m/s, r = 0.95; forecast vs EirGrid wind
  generation r = 0.86.
- Daily profile: in spring/summer the cleanest hours are 14:00–15:00 local (~173–176 g/kWh) and the dirtiest ~05:00
  (~221–231 g/kWh); in winter the profile is flat (~177–193 g/kWh).
- Monthly mean CO₂ intensity ranged from 155 g/kWh (Mar 2026) to 255 g/kWh (Nov 2024), moving opposite to wind share.

## 3. Experiment setup

| Item | Setting |
|---|---|
| Task | Predict hourly ROI CO₂ intensity (g/kWh) 24 h ahead |
| No-leakage rule | Grid features use only values ≥ 24 h old; weather input is the Open-Meteo `_previous_day1` forecast (issued ~24 h earlier). Realised wind is never a model input |
| Feature set A (21) | Calendar (hour, weekday, month, holiday, sin/cos of hour and day-of-year) + lagged grid features (CO₂ intensity, wind MW, demand MW, wind share at lags 24, 48, 168 h; 24 h rolling mean of CO₂ lagged 24 h) |
| Feature set B (26) | A + 24 h-ahead 100 m wind-speed forecasts at the 4 sites + their mean |
| Split (time-based) | Train 8 Jun 2024 → ~2 Mar 2026 (9,873 h); validation ~2 Mar → 2 May 2026 (939 h); test 2 May → 2 Nov 2026 (3,779 h). Same rows for every model |
| Models | Naive (same hour yesterday); Linear Regression (standardised); Random Forest (300 trees, min leaf 3, max_features 0.5); XGBoost (lr 0.03, depth 6, early stopping on validation); LSTM (1 layer, 32 units, 24-h input window, Adam, early stopping) |
| Metrics | MAE, RMSE, MAPE on test; training time, energy (kWh) and emissions (g CO₂) via CodeCarbon (`OfflineEmissionsTracker`, country IRL) |
| Tools | Python 3.14, pandas, scikit-learn, XGBoost, PyTorch (CPU), CodeCarbon, SHAP, matplotlib/seaborn |
| Seeds | Fixed (42) |

## 4. Results

### 4.1 Forecast accuracy and training cost (`results/model_results.csv`)

| Model | Set | MAE (g/kWh) | RMSE | MAPE (%) | Train time (s) | Energy (Wh) | Emissions (g CO₂) |
|---|---|---|---|---|---|---|---|
| Naive | – | 32.30 | 42.63 | 18.90 | 0 | 0 | 0 |
| Linear Regression | A | 30.99 | 38.46 | 18.95 | 0.01 | 0.00001 | 0.00000 |
| Random Forest | A | 32.97 | 40.98 | 20.35 | 3.37 | 0.0183 | 0.00532 |
| XGBoost | A | 34.64 | 42.00 | 22.25 | 0.96 | 0.0048 | 0.00141 |
| LSTM | A | 32.08 | 40.31 | 18.82 | 6.92 | 0.0314 | 0.00912 |
| **Linear Regression** | **B** | **23.25** | **29.04** | **14.42** | 0.02 | 0.00001 | 0.00000 |
| Random Forest | B | 25.57 | 32.17 | 15.52 | 4.01 | 0.0217 | 0.00631 |
| XGBoost | B | 26.21 | 32.69 | 15.99 | 1.55 | 0.0082 | 0.00240 |
| LSTM | B | 23.45 | 29.60 | 14.03 | 9.03 | 0.0431 | 0.01252 |

MAE reduction from A to B: Linear Regression −25.0%, Random Forest −22.5%, XGBoost −24.3%, LSTM −26.9%.
Best model (Linear Regression, B) is 28.0% better than the naive baseline in MAE.
Total training emissions for all 8 runs: 0.037 g CO₂.

### 4.2 Explainability (`results/shap_importance.csv`, Random Forest set B, the best tree model)

Top features by mean |SHAP| (g/kWh): 24 h-ahead wind forecast, 4-site mean (24.74); CO₂ intensity lag 24 h (9.48);
wind forecast Mayo (6.93); Donegal (5.17); Cork (4.56); weekday (2.94); 24 h rolling CO₂ lagged 24 h (2.89);
CO₂ lag 168 h (2.86). Weather-forecast features dominate.

### 4.3 Scheduler simulation (`results/scheduler_results.csv`)

Setup: 1,000 simulated flexible jobs in the test period, random arrival hour, duration 4–8 h, constant 1 MW, must
finish before the deadline. All policies are scored with **actual** CO₂ intensity. Same jobs for every deadline.

| Deadline | Policy | Total t CO₂ | Saved t CO₂ | % saved vs run-now | % of oracle savings |
|---|---|---|---|---|---|
| 12 h | Run now | 1,081.11 | 0 | 0 | 0 |
| 12 h | Shift (naive forecast) | 1,043.70 | 37.42 | 3.46 | 48.7 |
| 12 h | Shift (model forecast) | 1,031.87 | 49.24 | 4.55 | 64.1 |
| 12 h | Oracle | 1,004.25 | 76.86 | 7.11 | 100 |
| 24 h | Shift (naive forecast) | 1,006.68 | 74.44 | 6.89 | 45.4 |
| 24 h | **Shift (model forecast)** | **974.44** | **106.67** | **9.87** | **65.0** |
| 24 h | Oracle | 916.99 | 164.12 | 15.18 | 100 |
| 48 h | Shift (naive forecast) | 1,009.67 | 71.44 | 6.61 | 30.5 |
| 48 h | Shift (model forecast) | 925.99 | 155.13 | 14.35 | 66.1 |
| 48 h | Oracle | 846.54 | 234.57 | 21.70 | 100 |

### 4.4 Carbon payback (`results/carbon_payback.csv`)

| Item | Value |
|---|---|
| CO₂ saved, model-forecast shifting, 24 h deadline | 106.67 t (1.07 × 10⁸ g) |
| CO₂ emitted training the best model (Linear Regression B) | < 0.00001 g |
| CO₂ emitted training all 8 models | 0.037 g |
| Ratio saved ÷ training emissions (all models) | ≈ 2.9 × 10⁹ |

## 5. Figures (`figures/`, PNG, 300 dpi)

| File | Caption |
|---|---|
| `co2_by_hour_season.png` | Mean CO₂ intensity by hour of day (Irish local time) for each season. |
| `co2_vs_wind_share.png` | Hourly CO₂ intensity vs share of demand met by wind (r = −0.75). |
| `co2_monthly_trend.png` | Monthly mean CO₂ intensity with hourly min–max range and mean wind share. |
| `wind_forecast_accuracy.png` | 24 h-ahead Open-Meteo wind forecast vs realised wind speed and vs EirGrid wind output. |
| `A_vs_B.png` | Test MAE per model with feature set A vs B, against the naive baseline. |
| `accuracy_vs_energy.png` | Test MAE vs CodeCarbon training energy for each model and feature set. |
| `forecast_example_week.png` | Actual vs predicted CO₂ intensity (best model and naive) for the most variable test week (12–19 Oct 2026). |
| `shap_summary.png` | SHAP summary of feature impact for the Random Forest (set B). |
| `scheduler_savings.png` | % CO₂ saved vs run-now by policy for 12, 24 and 48 h deadlines. |

## 6. Five key findings

1. **Weather forecasts improve CO₂ prediction.** Adding 24 h-ahead wind forecasts cut test MAE by 22–27% for every
   model. Without them, models barely beat "same hour yesterday" (MAE 31–35 vs 32.3 g/kWh).
2. **A simple model is enough.** Linear Regression with set B was the most accurate (MAE 23.2 g/kWh, MAPE 14.4%),
   narrowly ahead of the LSTM (23.4) and ahead of Random Forest and XGBoost.
3. **Wind forecasts are what the model relies on.** In SHAP, the mean wind forecast has about 2.6× the impact of the
   next feature (yesterday's CO₂).
4. **Carbon-aware shifting saves CO₂.** With a 24 h deadline, shifting by the model forecast saved 9.9% of job CO₂
   (106.7 t over 1,000 jobs), about 65% of the perfect-knowledge maximum and more than the naive forecast (6.9%).
   Savings grow with flexibility: 4.5% (12 h) → 9.9% (24 h) → 14.3% (48 h).
5. **The forecaster's own footprint is negligible.** Training all models emitted ~0.04 g CO₂, many orders of
   magnitude below the CO₂ saved. The energy used by the jobs is unchanged; only its timing, and therefore its CO₂, changes.

## 7. Limitations

- Forecast history is only available from June 2024 (~2.4 years of data; validation window only 939 h due to data gaps).
- The EirGrid endpoint is unofficial and undocumented; 16 CO₂ days and 19 demand days could not be recovered.
- 2,409 CO₂ intensity hours are derived (emissions ÷ demand), not published values.
- Jobs are simulated (no public Irish data-centre workload data); constant 1 MW load; no capacity or queue constraints.
- Average, not marginal, emission factors are used: shifting large loads could itself change which plants run.
- The 48 h case reuses 24 h-ahead forecasts for hours 24–48 h ahead, so it is optimistic.
- CodeCarbon energy is estimated (CPU power from rated TDP on Windows), and very short runs are hard to measure.
- Only one test period (May–Nov 2026, mostly summer/autumn); results may differ in winter.

## 8. Suggested design improvements

- Use marginal emission factors (or a dispatch model) to capture the effect of moving load.
- Add 48–72 h forecasts and solar/demand forecasts for longer-deadline scheduling.
- Produce probabilistic forecasts (prediction intervals) so the scheduler can account for risk.
- Retrain on a rolling basis and test across all seasons (rolling-origin evaluation).
- Validate with real data-centre workload traces and power limits; include cooling (PUE).
- Move to an official, documented data source (e.g. EirGrid / ENTSO-E transparency platform).
