"""Carbon-aware job-shifting simulation on the test period.

Each simulated job: random arrival hour, duration 4-8 h, constant 1 MW, must finish within the deadline.
Policies choose a start hour; ALL policies are scored with ACTUAL CO2 intensity.
  1. Run now                 - start at arrival
  2. Shift (model forecast)  - window with lowest PREDICTED mean CO2 (best Phase 3 model)
  3. Shift (naive forecast)  - same, using "same hour yesterday" as the forecast (extra reference)
  4. Oracle                  - window with lowest ACTUAL mean CO2 (upper bound)
Usage: py src/scheduler.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import ROOT

RES, FIG = ROOT / "results", ROOT / "figures"
N_JOBS, POWER_MW, SEED = 1000, 1.0, 42
DEADLINES = [12, 24, 48]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

POLICIES = {"Run now": None, "Shift (model forecast)": "predicted",
            "Shift (naive forecast)": "naive_pred", "Oracle (perfect knowledge)": "actual"}


def window_means(x, dur):
    """Mean of x over every window of length dur starting at each hour (NaN if any hour missing)."""
    return pd.Series(x).rolling(dur).mean().shift(-(dur - 1)).values


def main():
    df = pd.read_csv(RES / "best_predictions.csv", parse_dates=["time_utc"]).set_index("time_utc").asfreq("1h")
    model_name = df["model"].dropna().iloc[0]
    rng = np.random.default_rng(SEED)
    n, max_dl = len(df), max(DEADLINES)
    means = {dur: {c: window_means(df[c].values, dur) for c in ["actual", "predicted", "naive_pred"]} for dur in range(4, 9)}

    # Draw jobs once (same jobs for every deadline). Keep only jobs whose run-now window has data.
    jobs = []
    while len(jobs) < N_JOBS:
        a, dur = int(rng.integers(0, n - max_dl)), int(rng.integers(4, 9))
        if not np.isnan(means[dur]["actual"][a]):
            jobs.append((a, dur))

    rows = []
    for dl in DEADLINES:
        totals = dict.fromkeys(POLICIES, 0.0)
        for a, dur in jobs:
            m = means[dur]
            starts = np.arange(a, a + dl - dur + 1)
            ok = starts[~np.isnan(m["actual"][starts]) & ~np.isnan(m["predicted"][starts]) & ~np.isnan(m["naive_pred"][starts])]
            if a not in ok:
                ok = np.append(ok, a)
            for pol, col in POLICIES.items():
                s = a if col is None else ok[np.nanargmin(m[col][ok])]
                # actual tCO2 = mean g/kWh * MW * hours (g/kWh * MWh = kg... /1000 -> t)
                totals[pol] += m["actual"][s] * POWER_MW * dur / 1000
        base, oracle = totals["Run now"], totals["Oracle (perfect knowledge)"]
        for pol, t in totals.items():
            rows.append({"deadline_h": dl, "policy": pol, "total_tCO2": round(t, 2),
                         "saved_tCO2": round(base - t, 2), "pct_saved_vs_run_now": round((base - t) / base * 100, 2),
                         "pct_of_oracle_savings": round((base - t) / (base - oracle) * 100, 1) if base > oracle else np.nan})
    res = pd.DataFrame(rows)
    res.to_csv(RES / "scheduler_results.csv", index=False)
    print(f"{N_JOBS} jobs, 4-8 h, {POWER_MW} MW, test period, forecaster = {model_name}\n")
    print(res.to_string(index=False))

    # Figure: % CO2 saved vs run-now, by policy and deadline
    piv = res[res.policy != "Run now"].pivot(index="deadline_h", columns="policy", values="pct_saved_vs_run_now")
    piv = piv[["Shift (naive forecast)", "Shift (model forecast)", "Oracle (perfect knowledge)"]]
    ax = piv.plot.bar(figsize=(9, 5), rot=0, color=["#bbbbbb", "#fc8d62", "#66c2a5"])
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f%%", fontsize=9)
    ax.set_ylim(0, piv.values.max() * 1.2)
    ax.set(title=f"CO₂ saved by shifting {N_JOBS:,} flexible 1 MW jobs vs running immediately",
           xlabel="Deadline after job arrival (hours)", ylabel="CO₂ saved vs run-now (%)")
    ax.legend(title=None, loc="upper left", fontsize=9)
    ax.figure.tight_layout(); ax.figure.savefig(FIG / "scheduler_savings.png", dpi=300); plt.close(ax.figure)

    # Carbon payback: CO2 saved by scheduling vs CO2 emitted training the forecasters
    mr = pd.read_csv(RES / "model_results.csv")
    best_g = mr.loc[(mr.model + " (" + mr.feature_set + ")") == model_name, "emissions_g"].iloc[0]
    all_g = mr.emissions_g.sum()
    saved_g = res.query("deadline_h == 24 and policy == 'Shift (model forecast)'").saved_tCO2.iloc[0] * 1e6
    pay = pd.DataFrame([
        {"item": "CO2 saved, model-forecast shifting, 24 h deadline (g)", "value": saved_g},
        {"item": f"CO2 emitted training best model {model_name} (g)", "value": best_g},
        {"item": "CO2 emitted training all 8 models (g)", "value": all_g},
        {"item": "Saved / training emissions (all models), ratio", "value": saved_g / all_g if all_g else np.nan},
    ])
    pay.to_csv(RES / "carbon_payback.csv", index=False)
    print("\n" + pay.to_string(index=False))


if __name__ == "__main__":
    main()
