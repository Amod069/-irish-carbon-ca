"""Generate notebooks/01_eda.ipynb (run: py src/make_eda_notebook.py, then execute the notebook)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("# 01 – Exploratory data analysis\nIrish grid (ROI) CO₂ intensity, wind, demand and Open-Meteo 24 h-ahead wind forecasts, hourly, UTC.")
code('''from pathlib import Path
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
FIG, RES = ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="talk", font_scale=0.8, font="DejaVu Sans")  # DejaVu has the ₂ glyph
df = pd.read_csv(ROOT / "data/processed/clean_hourly.csv", parse_dates=["time_utc"]).set_index("time_utc")
df = df[df.index >= "2024-06-01"]  # drop the single 23:00 UTC hour from 31 May
local = df.index.tz_convert("Europe/Dublin")
df["season"] = pd.Categorical(local.month.map({12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
    6:"Summer",7:"Summer",8:"Summer",9:"Autumn",10:"Autumn",11:"Autumn"}), ["Winter","Spring","Summer","Autumn"])
def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, dpi=300); plt.show()''')

md("## 1. Dataset description")
code('''desc = pd.DataFrame([
    ["EirGrid Smart Grid Dashboard (ROI)", "CO₂ intensity, CO₂ emissions, wind, demand, solar", "15-min → hourly mean"],
    ["Open-Meteo Previous Model Runs", "100 m wind speed, 24 h-ahead forecast + near-actual, 4 sites", "hourly"],
], columns=["source", "variables", "resolution"])
desc["period (UTC)"] = f"{df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}"
desc["rows"] = len(df); desc["features"] = df.shape[1] - 1
desc["CO₂ intensity missing %"] = round(df.co2_intensity.isna().mean() * 100, 2)
desc["CO₂ intensity filled (emissions/demand)"] = int(df.co2_intensity_filled.sum())
desc.to_csv(RES / "dataset_description.csv", index=False); desc.T''')
md("**What it shows:** ~21,200 hourly rows from June 2024 to early November 2026. The target (CO₂ intensity) is missing for only "
   "3.3% of hours; ~2,400 hours were derived from emissions ÷ demand where EirGrid did not publish intensity. "
   "Open-Meteo wind data is complete.")

md("## 2. Summary statistics")
code('''cols = ["co2_intensity", "co2_emissions_t_per_h", "wind_mw", "demand_mw", "solar_mw", "wind_share",
        "wind100_fc_mean", "wind100_act_mean"]
stats = df[cols].describe().T.round(2)
stats["missing_%"] = (df[cols].isna().mean() * 100).round(2)
stats.to_csv(RES / "summary_stats.csv"); stats''')
md("**What it shows:** mean CO₂ intensity is ~195 g/kWh but ranges from ~22 to ~471 g/kWh, so there is large room to shift load. "
   "Wind supplies ~32% of demand on average with very high variability. Solar is only reported from May 2025 (40% missing) and has "
   "small negative values (metering artefacts), so it is not used as a model input.")

md("## 3. CO₂ intensity by hour of day, per season")
code('''fig, ax = plt.subplots(figsize=(9, 5))
sns.lineplot(data=df.assign(hour_local=local.hour), x="hour_local", y="co2_intensity", hue="season",
             estimator="mean", errorbar=None, ax=ax, marker="o")
ax.set(title="Mean CO₂ intensity by hour of day (Irish local time)", xlabel="Hour of day", ylabel="CO₂ intensity (g/kWh)")
ax.set_xticks(range(0, 24, 3)); save(fig, "co2_by_hour_season.png")
df.assign(hour_local=local.hour).pivot_table(index="hour_local", columns="season", values="co2_intensity", observed=True).round(0).T''')
md("**What it shows:** in spring and summer the grid is cleanest in the early afternoon (~173–176 g/kWh at 14:00–15:00, solar + wind) "
   "and dirtiest around 05:00 (~221–231 g/kWh). In winter the daily curve is flat (~177–193 g/kWh), so savings from shifting "
   "depend mostly on day-to-day wind, not on the clock — a fixed 'run at night' rule is not enough.")

md("## 4. CO₂ intensity vs wind share")
code('''d = df[["wind_share", "co2_intensity"]].dropna()
r = d.corr().iloc[0, 1]
fig, ax = plt.subplots(figsize=(8, 6))
ax.hexbin(d.wind_share * 100, d.co2_intensity, gridsize=50, cmap="viridis", mincnt=1)
ax.set(title=f"CO₂ intensity vs wind share (Pearson r = {r:.2f})", xlabel="Wind share of demand (%)", ylabel="CO₂ intensity (g/kWh)")
save(fig, "co2_vs_wind_share.png"); print(f"Pearson r = {r:.3f}, n = {len(d)}")''')
md("**What it shows:** a strong negative relationship (r ≈ −0.75, so wind share alone explains ~56% of the variance): the more of "
   "demand met by wind, the cleaner the grid. This is the main reason we expect wind *forecasts* (feature set B) to improve "
   "24 h-ahead CO₂ prediction.")

md("## 5. Monthly CO₂ intensity trend")
code('''m = df.co2_intensity.resample("MS").agg(["mean", "min", "max"])
m2 = df.wind_share.resample("MS").mean() * 100
fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(m.index, m["min"], m["max"], alpha=0.2, label="hourly min–max")
ax.plot(m.index, m["mean"], marker="o", label="monthly mean CO₂ intensity")
ax.set(title="Monthly CO₂ intensity, Irish grid (ROI)", xlabel="Month", ylabel="CO₂ intensity (g/kWh)")
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
ax2 = ax.twinx(); ax2.plot(m2.index, m2, color="tab:green", ls="--", label="wind share (%)"); ax2.set_ylabel("Mean wind share (%)"); ax2.grid(False)
fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88), fontsize=9); save(fig, "co2_monthly_trend.png")
pd.concat([m["mean"].round(0), m2.round(1).rename("wind_share_%")], axis=1)''')
md("**What it shows:** monthly mean intensity moves opposite to wind share (e.g. Feb 2025: 49% wind, 160 g/kWh; Nov 2024: 29% wind, "
   "255 g/kWh). Months in 2026 are generally cleaner than 2024 (mostly 155–197 vs 181–255 g/kWh). The last point (Nov 2026) covers "
   "only 2 days.")

md("## 6. 24 h-ahead wind forecast accuracy")
code('''d = df[["wind100_fc_mean", "wind100_act_mean", "wind_mw"]].dropna()
mae = (d.wind100_fc_mean - d.wind100_act_mean).abs().mean()
r_ws = d.wind100_fc_mean.corr(d.wind100_act_mean); r_mw = d.wind100_fc_mean.corr(d.wind_mw)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
axes[0].hexbin(d.wind100_act_mean, d.wind100_fc_mean, gridsize=50, cmap="viridis", mincnt=1)
lim = [0, d[["wind100_fc_mean", "wind100_act_mean"]].max().max()]; axes[0].plot(lim, lim, "r--", lw=1)
axes[0].set(title=f"Forecast vs actual 100 m wind (MAE {mae:.2f} m/s, r = {r_ws:.2f})",
            xlabel="Actual wind speed, 4-site mean (m/s)", ylabel="24 h-ahead forecast (m/s)")
axes[1].hexbin(d.wind100_fc_mean, d.wind_mw, gridsize=50, cmap="viridis", mincnt=1)
axes[1].set(title=f"Forecast wind speed vs EirGrid wind output (r = {r_mw:.2f})",
            xlabel="24 h-ahead forecast wind speed (m/s)", ylabel="Actual wind generation (MW)")
save(fig, "wind_forecast_accuracy.png")
print(f"MAE = {mae:.3f} m/s, r(fc, actual speed) = {r_ws:.3f}, r(fc, wind MW) = {r_mw:.3f}")''')
md("**What it shows:** the 24 h-ahead forecast is close to the realised wind (MAE ≈ 0.77 m/s, r ≈ 0.95) and correlates strongly with "
   "actual Irish wind generation (r ≈ 0.86). So a forecast that is legitimately available a day ahead carries most of the wind "
   "signal — without leaking future information.")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
out = ROOT / "notebooks" / "01_eda.ipynb"
nbf.write(nb, out); print(f"Wrote {out}")
