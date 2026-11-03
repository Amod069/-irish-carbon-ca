"""Build data/processed/clean_hourly.csv from the raw EirGrid + Open-Meteo JSON files.

Steps: parse -> Dublin local time to UTC -> 15-min to hourly mean -> fill missing CO2 intensity
(emissions / demand * 1000, flagged) -> merge Open-Meteo wind -> calendar features -> save + quality report.
Usage: py src/build_dataset.py
"""
import json
from pathlib import Path

import holidays
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "clean_hourly.csv"

FIELD_NAMES = {
    "CO2_EMISSIONS": "co2_emissions_t_per_h",
    "CO2_INTENSITY": "co2_intensity",
    "WIND_ACTUAL": "wind_mw",
    "SYSTEM_DEMAND": "demand_mw",
    "SOLAR_ACTUAL": "solar_mw",
}


def dublin_to_utc(local_times):
    """Convert naive Europe/Dublin times (in file order) to UTC.
    October: the repeated hour is resolved by order ('infer'). March: the missing hour does not occur."""
    idx = pd.DatetimeIndex(local_times)
    try:
        return idx.tz_localize("Europe/Dublin", ambiguous="infer", nonexistent="NaT").tz_convert("UTC")
    except Exception:  # repeated hour not present twice -> drop ambiguous rows
        return idx.tz_localize("Europe/Dublin", ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")


def load_eirgrid():
    frames = []
    for f in sorted((RAW / "eirgrid").glob("*.json")):
        rows = pd.DataFrame(json.loads(f.read_text(encoding="utf-8"))["Rows"])
        if rows.empty:
            continue
        for field, g in rows.groupby("FieldName", sort=False):
            if field not in FIELD_NAMES:
                continue
            times = pd.to_datetime(g["EffectiveTime"], format="%d-%b-%Y %H:%M:%S")
            frames.append(pd.DataFrame({
                "time_utc": dublin_to_utc(times),
                "field": FIELD_NAMES[field],
                "value": pd.to_numeric(g["Value"], errors="coerce").values,
            }))
    df = pd.concat(frames).dropna(subset=["time_utc"])
    wide = df.pivot_table(index="time_utc", columns="field", values="value", aggfunc="mean")
    hourly = wide.resample("1h").mean()
    return hourly.reindex(columns=list(FIELD_NAMES.values()))


def load_meteo():
    frames = []
    for f in sorted((RAW / "openmeteo").glob("*.json")):
        loc = f.stem.split("_")[0]
        h = json.loads(f.read_text(encoding="utf-8"))["hourly"]
        frames.append(pd.DataFrame({
            "time_utc": pd.to_datetime(h["time"], utc=True),
            "loc": loc,
            # previous_day1 = forecast issued 24 h earlier -> the only weather MODEL input allowed
            "wind100_fc": h["wind_speed_100m_previous_day1"],
            # near-actual wind -> for analysis only, never a model input
            "wind100_act": h["wind_speed_100m"],
        }))
    df = pd.concat(frames).drop_duplicates(["time_utc", "loc"], keep="last")
    wide = df.pivot(index="time_utc", columns="loc", values=["wind100_fc", "wind100_act"])
    wide.columns = [f"{var}_{loc}" for var, loc in wide.columns]
    for var in ["wind100_fc", "wind100_act"]:
        wide[f"{var}_mean"] = wide.filter(like=f"{var}_").mean(axis=1)
    return wide


def main():
    grid = load_eirgrid()

    # Fill missing CO2 intensity: tCO2/h / MW = tCO2/MWh = 1000 g/kWh
    missing = grid["co2_intensity"].isna() & grid["co2_emissions_t_per_h"].notna() & grid["demand_mw"].notna()
    grid["co2_intensity_filled"] = missing.astype(int)
    grid.loc[missing, "co2_intensity"] = grid.loc[missing, "co2_emissions_t_per_h"] / grid.loc[missing, "demand_mw"] * 1000

    df = grid.join(load_meteo(), how="left")

    # Calendar features (Irish local time)
    local = df.index.tz_convert("Europe/Dublin")
    ie_holidays = holidays.Ireland(years=range(local.year.min(), local.year.max() + 1))
    df["hour"] = local.hour
    df["weekday"] = local.weekday
    df["month"] = local.month
    df["is_holiday"] = [int(d in ie_holidays) for d in local.date]
    df["wind_share"] = df["wind_mw"] / df["demand_mw"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index_label="time_utc")

    # Data-quality report
    print(f"Saved {OUT}")
    print(f"Date range (UTC): {df.index.min()} -> {df.index.max()}")
    print(f"Rows: {len(df)}, columns: {df.shape[1]}")
    print(f"CO2 intensity values filled from emissions/demand: {int(df['co2_intensity_filled'].sum())}")
    print("Missing % per column:")
    print((df.isna().mean() * 100).round(2).to_string())


if __name__ == "__main__":
    main()
