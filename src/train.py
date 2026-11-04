"""Train and compare 24 h-ahead CO2 intensity forecasters (feature set A vs B), with CodeCarbon energy tracking.

Usage:  py src/train.py           (full run)
        py src/train.py --quick   (tiny smoke test, results not saved)
Outputs: results/model_results.csv, results/best_predictions.csv,
         figures/accuracy_vs_energy.png, A_vs_B.png, forecast_example_week.png, shap_summary.png
"""
import argparse
import logging
import random
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from codecarbon import OfflineEmissionsTracker
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from features import FEATURE_SETS, ROOT, TARGET, load_features, model_table

warnings.filterwarnings("ignore")
logging.getLogger("codecarbon").setLevel(logging.ERROR)
SEED = 42
RES, FIG = ROOT / "results", ROOT / "figures"
SEQ_LEN = 24  # LSTM looks at the feature rows of the last 24 hours (each row already leak-free)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def set_seeds():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {"MAE": np.mean(np.abs(y - p)), "RMSE": np.sqrt(np.mean((y - p) ** 2)),
            "MAPE": np.mean(np.abs((y - p) / y)) * 100}


def tracked(fn):
    """Run fn() inside CodeCarbon; return (result, seconds, kWh, gCO2)."""
    tracker = OfflineEmissionsTracker(country_iso_code="IRL", measure_power_secs=1,
                                      save_to_file=False, log_level="error")
    tracker.start(); t0 = time.time()
    out = fn()
    secs = time.time() - t0
    kg = tracker.stop() or 0.0
    kwh = tracker.final_emissions_data.energy_consumed
    return out, secs, kwh, kg * 1000


# ---------- LSTM ----------
class LSTMNet(torch.nn.Module):
    def __init__(self, n_features, hidden=32):
        super().__init__()
        self.lstm = torch.nn.LSTM(n_features, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def make_sequences(X_all, row_pos):
    """For each target row position i, stack feature rows i-23..i -> (n, 24, f)."""
    idx = row_pos[:, None] + np.arange(-SEQ_LEN + 1, 1)[None, :]
    return X_all[idx]


def train_lstm(X_tr, y_tr, X_va, y_va, epochs, patience=5):
    set_seeds()
    model = LSTMNet(X_tr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    Xt, yt = torch.tensor(X_tr), torch.tensor(y_tr)
    Xv, yv = torch.tensor(X_va), torch.tensor(y_va)
    best, best_state, wait = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for b in range(0, len(Xt), 256):
            i = perm[b:b + 256]
            opt.zero_grad(); loss = loss_fn(model(Xt[i]), yt[i]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xv), yv).item()
        if v < best:
            best, best_state, wait = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    print(f"    LSTM stopped after {ep + 1} epochs")
    return model


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); args = ap.parse_args()
    set_seeds()
    full = load_features()
    data = model_table(full)

    # Keep only rows whose previous 23 hours are also usable, so the LSTM sees the SAME rows as other models
    pos_in_full = full.index.get_indexer(data.index)
    usable = np.zeros(len(full), bool); usable[pos_in_full] = True
    run = np.convolve(usable, np.ones(SEQ_LEN, int), "full")[:len(full)]  # count of usable rows in window ending at i
    data = data[run[pos_in_full] == SEQ_LEN]
    pos_in_data = np.arange(len(data))
    if args.quick:
        data = data.iloc[-3000:]; pos_in_data = np.arange(len(data))

    # Time-based split: test = last 6 months, validation = last 2 months before test
    test_start = data.index.max() - pd.DateOffset(months=6)
    val_start = test_start - pd.DateOffset(months=2)
    if args.quick:
        test_start, val_start = data.index[-1000], data.index[-1500]
    tr, va, te = data.index < val_start, (data.index >= val_start) & (data.index < test_start), data.index >= test_start
    y = data[TARGET].values.astype(np.float32)
    print(f"Rows: train {tr.sum()}, val {va.sum()}, test {te.sum()} | test {data.index[te].min():%Y-%m-%d} -> {data.index[te].max():%Y-%m-%d}")

    rows, preds = [], {}
    # Naive baseline: same hour yesterday (nothing to train)
    p = data["co2_intensity_lag24"].values[te]
    rows.append({"model": "Naive", "feature_set": "-", **metrics(y[te], p), "train_time_s": 0, "energy_kWh": 0, "emissions_g": 0})
    preds[("Naive", "-")] = p

    n_rf, n_xgb, n_ep = (20, 50, 2) if args.quick else (300, 2000, 40)
    fitted = {}
    for fs, cols in FEATURE_SETS.items():
        X = data[cols].values.astype(np.float32)
        Xtv, ytv = X[tr | va], y[tr | va]
        print(f"Feature set {fs} ({len(cols)} features)")

        models = {
            "Linear Regression": lambda: make_pipeline(StandardScaler(), LinearRegression()).fit(Xtv, ytv),
            "Random Forest": lambda: RandomForestRegressor(n_estimators=n_rf, min_samples_leaf=3, max_features=0.5,
                                                           n_jobs=-1, random_state=SEED).fit(Xtv, ytv),
            "XGBoost": lambda: XGBRegressor(n_estimators=n_xgb, learning_rate=0.03, max_depth=6, subsample=0.8,
                                            colsample_bytree=0.8, early_stopping_rounds=100, random_state=SEED,
                                            n_jobs=-1).fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False),
        }
        for name, fit in models.items():
            m, secs, kwh, g = tracked(fit)
            p = m.predict(X[te])
            rows.append({"model": name, "feature_set": fs, **metrics(y[te], p), "train_time_s": secs, "energy_kWh": kwh, "emissions_g": g})
            preds[(name, fs)] = p; fitted[(name, fs)] = m
            print(f"  {name:18s} MAE {rows[-1]['MAE']:6.2f}  ({secs:.1f} s)")

        # LSTM: scale with train statistics only
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        ym, ys = y[tr].mean(), y[tr].std()
        Xs = ((X - mu) / sd).astype(np.float32)
        seq = lambda mask: make_sequences(Xs, pos_in_data[mask])
        model, secs, kwh, g = tracked(lambda: train_lstm(seq(tr), (y[tr] - ym) / ys, seq(va), (y[va] - ym) / ys, n_ep))
        with torch.no_grad():
            p = model(torch.tensor(seq(te))).numpy() * ys + ym
        rows.append({"model": "LSTM", "feature_set": fs, **metrics(y[te], p), "train_time_s": secs, "energy_kWh": kwh, "emissions_g": g})
        preds[("LSTM", fs)] = p
        print(f"  {'LSTM':18s} MAE {rows[-1]['MAE']:6.2f}  ({secs:.1f} s)")

    res = pd.DataFrame(rows).round({"MAE": 3, "RMSE": 3, "MAPE": 3, "train_time_s": 2, "energy_kWh": 8, "emissions_g": 5})
    print("\n", res.to_string(index=False))
    if args.quick:
        print("\nQuick test OK (nothing saved)."); return

    RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    res.to_csv(RES / "model_results.csv", index=False)

    # Best model (lowest test MAE, excluding naive) -> predictions for Phase 4
    trained = res[res.model != "Naive"]
    best = trained.loc[trained.MAE.idxmin()]
    key = (best.model, best.feature_set)
    out = pd.DataFrame({"actual": y[te], "predicted": preds[key], "naive_pred": preds[("Naive", "-")]},
                       index=data.index[te])
    out["model"] = f"{best.model} ({best.feature_set})"
    out.to_csv(RES / "best_predictions.csv", index_label="time_utc")
    print(f"\nBest model: {best.model} set {best.feature_set}, MAE {best.MAE:.2f}")

    # Figure: accuracy vs energy
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for fs, mk in [("A", "o"), ("B", "s")]:
        d = trained[trained.feature_set == fs]
        ax.scatter(d.energy_kWh * 1000, d.MAE, marker=mk, s=80, label=f"Feature set {fs}")
        for _, r in d.iterrows():
            ax.annotate(r.model, (r.energy_kWh * 1000, r.MAE), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.axhline(res.loc[res.model == "Naive", "MAE"].iloc[0], ls="--", color="grey", label="Naive baseline")
    ax.set_xscale("log"); ax.legend()
    ax.set(title="Forecast accuracy vs training energy (CodeCarbon)", xlabel="Training energy (Wh, log scale)",
           ylabel="Test MAE (g CO₂/kWh)")
    fig.tight_layout(); fig.savefig(FIG / "accuracy_vs_energy.png", dpi=300); plt.close(fig)

    # Figure: A vs B
    piv = trained.pivot(index="model", columns="feature_set", values="MAE").loc[["Linear Regression", "Random Forest", "XGBoost", "LSTM"]]
    ax = piv.plot.bar(figsize=(8, 5), rot=0, color=["#8da0cb", "#fc8d62"])
    ax.axhline(res.loc[res.model == "Naive", "MAE"].iloc[0], ls="--", color="grey", label="Naive baseline")
    ax.legend(["Naive baseline", "A: calendar + lagged grid", "B: A + wind forecasts"], loc="upper center", ncol=3, fontsize=9)
    ax.set_ylim(0, piv.values.max() * 1.18)
    ax.set(title="Test MAE by model: feature set A vs B", xlabel="", ylabel="Test MAE (g CO₂/kWh)")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f", fontsize=9)
    ax.figure.tight_layout(); ax.figure.savefig(FIG / "A_vs_B.png", dpi=300); plt.close(ax.figure)

    # Figure: example test week (the week with the largest CO2 swings)
    weekly_std = out.actual.resample("W-MON").std()
    wk_start = weekly_std.idxmax() - pd.Timedelta(days=7)
    w = out[(out.index > wk_start) & (out.index <= wk_start + pd.Timedelta(days=7))]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(w.index, w.actual, color="black", lw=2, label="Actual")
    ax.plot(w.index, w.predicted, color="tab:orange", label=f"{best.model} (set {best.feature_set})")
    ax.plot(w.index, w.naive_pred, color="grey", ls=":", label="Naive (same hour yesterday)")
    ax.set(title=f"24 h-ahead forecast vs actual, test week from {w.index.min():%d %b %Y}", xlabel="Time (UTC)",
           ylabel="CO₂ intensity (g/kWh)")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "forecast_example_week.png", dpi=300); plt.close(fig)

    # SHAP for best tree model
    trees = trained[trained.model.isin(["Random Forest", "XGBoost"])]
    bt = trees.loc[trees.MAE.idxmin()]
    cols = FEATURE_SETS[bt.feature_set]
    Xte = data.loc[te, cols].sample(min(1000, te.sum()), random_state=SEED)
    sv = shap.TreeExplainer(fitted[(bt.model, bt.feature_set)]).shap_values(Xte.values.astype(np.float32))
    plt.figure()
    shap.summary_plot(sv, Xte, max_display=15, show=False)
    plt.title(f"SHAP feature impact: {bt.model} (set {bt.feature_set})")
    plt.gcf().set_size_inches(9, 7); plt.tight_layout(); plt.savefig(FIG / "shap_summary.png", dpi=300); plt.close()
    pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False).round(3) \
        .to_csv(RES / "shap_importance.csv", header=["mean_abs_shap"])
    print("Saved results/ and figures/.")


if __name__ == "__main__":
    main()
