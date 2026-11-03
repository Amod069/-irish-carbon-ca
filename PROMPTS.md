# Claude Code prompts – run one phase at a time

How to use:
1. Open this folder in VS Code.
2. Open the terminal and run `claude`.
3. Paste ONE prompt below. Check the result. Then paste the next one.
   (Claude Code reads CLAUDE.md automatically, so the prompts stay short.)

---

## Phase 1 – Data download + clean dataset

```
Read CLAUDE.md. Do Phase 1.
1. Create requirements.txt (requests, pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost,
   tensorflow or torch (pick one, CPU), codecarbon, shap, holidays, jupyter, tqdm) and .gitignore.
2. Write src/download.py: download EirGrid ROI CO2 + wind/demand day by day from 2024-06-01 to yesterday,
   and Open-Meteo previous-runs wind for the 4 locations month by month. Cache raw JSON per day/month,
   skip files that already exist, retry on failure, show a progress bar, log failed days to data/raw/failed.txt.
3. Write src/build_dataset.py: parse raw files, convert Dublin time to UTC, fill missing CO2 intensity
   from emissions/demand*1000 with a flag column, resample to hourly, merge Open-Meteo, add calendar
   features (hour, weekday, month, Irish holiday, wind_share = wind/demand), save data/processed/clean_hourly.csv.
4. Print a data-quality report: date range, rows, missing % per column, number of filled CO2 values.
First test download.py on 3 days only, then tell me the command to run the full download myself.
Update the Status section of CLAUDE.md when done.
```

---

## Phase 2 – Exploratory analysis

```
Read CLAUDE.md. Do Phase 2.
Create notebooks/01_eda.ipynb using data/processed/clean_hourly.csv:
dataset description table (source, rows, features, period, missing values), summary statistics,
CO2 intensity by hour of day (per season), CO2 vs wind share scatter with correlation,
monthly CO2 trend, forecast wind vs actual wind accuracy. Save each figure to figures/ as PNG.
Under each figure write 2–3 plain-English lines on what it shows.
```

---

## Phase 3 – Models (A vs B) + energy + explainability

```
Read CLAUDE.md. Do Phase 3.
Write src/features.py and src/train.py to forecast hourly CO2 intensity 24 h ahead.
Models: naive, Linear Regression, Random Forest, XGBoost, LSTM. Feature sets A and B as in CLAUDE.md.
Time-based split (test = last 6 months). No leakage: lags ≥ 24 h.
Wrap every training run in CodeCarbon. Save:
- results/model_results.csv (model, feature set, MAE, RMSE, MAPE, train_time_s, energy_kWh, emissions_g)
- figures/accuracy_vs_energy.png (each model/feature-set as a point)
- figures/A_vs_B.png (bar chart of MAE per model, A vs B)
- figures/forecast_example_week.png (actual vs predicted for one test week)
- figures/shap_summary.png for the best tree model
Keep the LSTM small so it trains in under ~10 minutes on a laptop CPU.
Save test-set predictions of the best model to results/best_predictions.csv for Phase 4.
```

---

## Phase 4 – Scheduler simulation

```
Read CLAUDE.md. Do Phase 4.
Write src/scheduler.py using results/best_predictions.csv and actual CO2 in the test period.
Simulate 1,000 flexible jobs (random start times, duration 4–8 h, deadline 24 h, constant power e.g. 1 MW).
Policies: (1) run immediately, (2) start at the window with lowest PREDICTED average CO2 before the deadline,
(3) oracle using ACTUAL CO2 (upper bound). Score all policies with ACTUAL CO2.
Save results/scheduler_results.csv (total tCO2 per policy, % saved vs run-now, % of oracle savings achieved)
and figures/scheduler_savings.png. Also run a sensitivity check: deadline 12 h vs 24 h vs 48 h.
Finally compare total CO2 saved by scheduling vs CO2 emitted training the model (from CodeCarbon).
```

---

## Phase 5 – Results summary for the report

```
Read CLAUDE.md. Do Phase 5.
Create results/summary.md for the report writer: dataset description (source, period, rows, features,
missing/filled values, cleaning steps), experiment setup (models, features A/B, split, metrics, tools),
all result tables, list of figures with one-line captions, the 5 key findings in plain English,
limitations, and suggested design improvements. Use only numbers from the results/ files – do not invent any.
Update the Status section of CLAUDE.md.
```

---

## Useful extra prompts

- `Something failed: <paste error>. Fix it and explain in one line what was wrong.`
- `Explain src/train.py to me in simple English, section by section, so I can describe it in the report.`
- `Make all figures consistent: same fonts, colours, 300 dpi, readable at half-page width.`
