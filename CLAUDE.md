# Project: Carbon-Aware Scheduling of Data Centre Workloads on the Irish Grid

MSc AI group CA, module H9ETS "Emerging AI Technologies and Sustainability", National College of Ireland.
Deliverable: 12-page report (PDF, Harvard referencing). Deadline: **10 November 2026**.
This code produces the data, experiments, tables and figures for that report.

## Research question
Can official weather forecasts improve 24-hour-ahead CO₂ intensity prediction for Ireland's grid, and how much CO₂
can data centres avoid by shifting flexible workloads to predicted low-carbon hours, once the forecaster's
own training energy is counted?

## Pain point
Data centres used 23% of Ireland's metered electricity in 2025 (CSO). Grid CO₂ intensity swings hour to hour between
windy (clean) and gas-heavy (dirty) periods. Flexible jobs (AI training, batch, backups) run regardless of grid
cleanliness. We forecast clean hours and simulate shifting jobs into them.

## Pipeline
1. `src/download.py`      → raw JSON per day into `data/raw/` (cached; reruns skip finished days)
2. `src/build_dataset.py` → `data/processed/clean_hourly.csv`
3. `notebooks/01_eda.ipynb` → exploratory figures
4. `src/train.py`         → models A vs B, CodeCarbon energy, results table, SHAP
5. `src/scheduler.py`     → job-shifting simulation, % CO₂ saved
6. `results/summary.md`   → all numbers/figures summarised for the report writer

## Data sources (VERIFIED working on 28 Oct 2026)

### EirGrid Smart Grid Dashboard (unofficial chart endpoint used by the website)
- URL: `https://www.smartgriddashboard.com/api/chart/`
- MUST send header `User-Agent: Mozilla/5.0`. Use timeout 60 s, retries with backoff, ~0.5 s delay between calls.
- Blocks/times out from cloud IPs (Google Colab). Run downloads on the local laptop.
- One day per request: `dateRange=day`, `dateFrom=dateTo=DD-Mon-YYYY` (e.g. `15-Jun-2025`), `region=ROI`.
- CO₂: `chartType=co2`, `areas=co2emission,co2intensity`
- Grid: `chartType=default`, `areas=windactual,demandactual` (optionally `solaractual`)
- Response: `{"Rows":[{"EffectiveTime":"15-Jun-2025 00:15:00","FieldName":"CO2_EMISSIONS","Region":"ROI","Value":533}, ...]}`
- FieldNames: `CO2_EMISSIONS` (tCO₂/hr), `CO2_INTENSITY` (gCO₂/kWh), `WIND_ACTUAL` (MW), `SYSTEM_DEMAND` (MW), `SOLAR_ACTUAL` (MW)
- EffectiveTime is Europe/Dublin local time, 15-min resolution (~96 rows/day). Convert to UTC; handle the DST
  duplicate/missing hour in October/March.
- Known gap: some days return no `CO2_INTENSITY` (e.g. 15-Jun-2025) while emissions exist.
  Fill: `co2_intensity = co2_emissions / demand * 1000` and keep a flag column `co2_intensity_filled`.

### Open-Meteo Previous Model Runs API (24 h-ahead weather forecasts)
- URL: `https://previous-runs-api.open-meteo.com/v1/forecast`
- Params: `latitude`, `longitude`, `hourly=wind_speed_100m,wind_speed_100m_previous_day1`, `start_date`, `end_date`
  (YYYY-MM-DD), `timezone=UTC`, `models=best_match`. Can request long ranges (fetch month by month).
- `_previous_day1` = forecast issued 24 h earlier → this is the ONLY forecast feature allowed in models (no leakage).
- `wind_speed_100m` (no suffix) is close to actual → use only for analysis, never as a model input.
- Available from **June 2024** (March 2024 returned nothing). Study period: **2024-06-01 → yesterday**.
- Locations (wind-farm regions): Donegal (54.9, -8.0), Mayo (54.1, -9.5), Kerry (52.2, -9.7), Cork (51.9, -8.9).
- Free for non-commercial use; cite Open-Meteo.

## Experiment rules
- Target: hourly ROI CO₂ intensity (g/kWh) for each hour of the next day (24 h ahead).
- No leakage: at forecast time only data up to the issue time is known. Lag features must be ≥ 24 h old.
- Time-based split, never random: test = last ~6 months; validation carved from end of train.
- Models: naive baseline (same hour yesterday), Linear Regression, Random Forest, XGBoost, LSTM.
- Feature sets: **A** = calendar + lagged grid features; **B** = A + Open-Meteo `_previous_day1` wind forecasts.
- Metrics: MAE, RMSE, MAPE; plus training time, energy (kWh) and emissions (gCO₂) via **CodeCarbon**
  (`EmissionsTracker`, country_iso_code="IRL") around every training run.
- Fixed random seeds; save every table to `results/` (CSV) and every figure to `figures/` (PNG, 300 dpi, clear titles/axis units).
- Scheduler: score every policy with ACTUAL CO₂ intensity, not predicted.

## Coding conventions
- Python 3.12+, Windows laptop (`py` launcher). Use `pathlib` paths, not hard-coded `/` or `\`.
- Keep scripts small and runnable: `py src/download.py`, etc. Print progress and a short summary at the end.
- Dependencies in `requirements.txt`; install with `py -m pip install -r requirements.txt`.
- Do not commit `data/raw/` (large). Processed CSV is fine.
- Prefer simple, readable code over clever code; the group must be able to explain it in the report.

## Known limitations (to be stated in report)
- Forecast history only from June 2024.
- EirGrid endpoint is unofficial/undocumented and may change.
- Some CO₂ intensity values are derived (emissions ÷ demand).
- Jobs are simulated (no public data-centre workload data for Ireland); average not marginal emissions used.
- CodeCarbon energy figures are estimates.
- 48 h scheduler deadline reuses 24 h-ahead forecasts for hours 24-48 h ahead (optimistic for that case).

## Status
- [x] Phase 1 – download + build dataset (done 3 Nov 2026: 21,241 hourly rows; CO2 intensity 3.3% missing, 2,409 filled; solar only from May 2025; 16 CO2 + 19 demand days unrecoverable)
- [x] Phase 2 – EDA (done 3 Nov 2026: 4 figures + dataset_description.csv, summary_stats.csv; r(wind share, CO2) = -0.75)
- [x] Phase 3 – models A vs B + CodeCarbon + SHAP (done 4 Nov 2026: best = Linear Regression set B, test MAE 23.2 vs naive 32.3; B beats A for all 4 models; top SHAP feature = wind100_fc_mean)
- [x] Phase 4 – scheduler simulation (done 4 Nov 2026: 24 h deadline saves 9.9% CO2 vs run-now = 65% of oracle; naive forecast 6.9%; training CO2 ~0.04 g vs ~107 t saved)
- [ ] Phase 5 – results summary for report
