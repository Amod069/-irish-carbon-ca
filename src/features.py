"""Feature engineering for 24 h-ahead CO2 intensity forecasting.

No leakage rule: for a target hour t, grid features come only from t-24 h or earlier.
Weather features are Open-Meteo `_previous_day1` forecasts for hour t (issued ~24 h before t).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "clean_hourly.csv"

TARGET = "co2_intensity"
LOCS = ["donegal", "mayo", "kerry", "cork"]
GRID_VARS = ["co2_intensity", "wind_mw", "demand_mw", "wind_share"]
LAGS = [24, 48, 168]  # hours: same hour yesterday, 2 days ago, last week

CALENDAR = ["hour", "weekday", "month", "is_holiday", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
LAGGED = [f"{v}_lag{l}" for v in GRID_VARS for l in LAGS] + ["co2_intensity_roll24_lag24"]
WEATHER = [f"wind100_fc_{loc}" for loc in LOCS] + ["wind100_fc_mean"]

FEATURE_SETS = {
    "A": CALENDAR + LAGGED,             # calendar + lagged grid features
    "B": CALENDAR + LAGGED + WEATHER,   # A + 24 h-ahead wind forecasts
}


def load_features():
    """Return a complete hourly DataFrame with target + all features (rows with gaps keep NaN)."""
    df = pd.read_csv(DATA, parse_dates=["time_utc"]).set_index("time_utc")
    df = df[df.index >= "2024-06-01"]
    df = df.asfreq("1h")  # make the hourly index complete so shift(24) really means 24 h

    local = df.index.tz_convert("Europe/Dublin")
    df["hour"], df["weekday"], df["month"] = local.hour, local.weekday, local.month
    df["is_holiday"] = df["is_holiday"].fillna(0)
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * local.hour / 24), np.cos(2 * np.pi * local.hour / 24)
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * local.dayofyear / 365.25), np.cos(2 * np.pi * local.dayofyear / 365.25)

    for v in GRID_VARS:
        for l in LAGS:
            df[f"{v}_lag{l}"] = df[v].shift(l)
    # mean CO2 intensity of the 24 h window ending 24 h before t
    df["co2_intensity_roll24_lag24"] = df["co2_intensity"].rolling(24, min_periods=18).mean().shift(24)
    return df


def model_table(df):
    """Rows usable by every model: target and all set-B features present (same rows for A and B -> fair comparison)."""
    cols = [TARGET] + FEATURE_SETS["B"]
    return df[cols].dropna()


if __name__ == "__main__":
    t = model_table(load_features())
    print(f"Usable rows: {len(t)}  ({t.index.min()} -> {t.index.max()})")
    for k, v in FEATURE_SETS.items():
        print(f"Set {k}: {len(v)} features")
