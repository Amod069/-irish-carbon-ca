"""Download raw data: EirGrid (CO2, wind, demand) per day + Open-Meteo 24h-ahead wind forecasts per month.

Usage:
    py src/download.py                                   # full study period 2024-06-01 -> yesterday
    py src/download.py --start 2025-06-14 --end 2025-06-16   # small test
Files already downloaded are skipped, so the script can be re-run after a failure.
"""
import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EIRGRID_DIR = RAW / "eirgrid"
METEO_DIR = RAW / "openmeteo"
FAILED_LOG = RAW / "failed.txt"

EIRGRID_URL = "https://www.smartgriddashboard.com/api/chart/"
METEO_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# chart name -> (chartType, areas)
EIRGRID_CHARTS = {
    "co2": ("co2", "co2emission,co2intensity"),
    "grid": ("default", "windactual,demandactual,solaractual"),
}
LOCATIONS = {
    "donegal": (54.9, -8.0),
    "mayo": (54.1, -9.5),
    "kerry": (52.2, -9.7),
    "cork": (51.9, -8.9),
}


def get_json(url, params, retries=4):
    """GET with retries and exponential backoff. Returns parsed JSON or raises the last error."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # network error, timeout, bad status, bad JSON
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 2)  # 2, 4, 8 s


def log_failure(msg):
    with open(FAILED_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def download_eirgrid(start, end):
    days = [start + timedelta(d) for d in range((end - start).days + 1)]
    jobs = [(d, name) for d in days for name in EIRGRID_CHARTS]
    done = failed = 0
    for d, name in tqdm(jobs, desc="EirGrid"):
        out = EIRGRID_DIR / f"{name}_{d.isoformat()}.json"
        if out.exists():
            continue
        chart_type, areas = EIRGRID_CHARTS[name]
        params = {
            "region": "ROI", "chartType": chart_type, "dateRange": "day",
            "dateFrom": d.strftime("%d-%b-%Y"), "dateTo": d.strftime("%d-%b-%Y"), "areas": areas,
        }
        try:
            data = get_json(EIRGRID_URL, params)
            if not data.get("Rows"):
                raise ValueError("empty Rows")
            out.write_text(json.dumps(data), encoding="utf-8")
            done += 1
        except Exception as e:
            failed += 1
            log_failure(f"eirgrid {name} {d.isoformat()} {e}")
        time.sleep(0.5)
    return done, failed


def backfill_demand():
    """Some days the 'grid' chart has no SYSTEM_DEMAND; the 'demand' chart still has it. Fetch only those days."""
    missing = [f.stem.split("_")[1] for f in sorted(EIRGRID_DIR.glob("grid_*.json"))
               if '"SYSTEM_DEMAND"' not in f.read_text(encoding="utf-8")]
    done = failed = 0
    for d in tqdm(missing, desc="Demand backfill"):
        out = EIRGRID_DIR / f"demand_{d}.json"
        if out.exists():
            continue
        day = date.fromisoformat(d).strftime("%d-%b-%Y")
        params = {"region": "ROI", "chartType": "demand", "dateRange": "day",
                  "dateFrom": day, "dateTo": day, "areas": "demandactual"}
        try:
            data = get_json(EIRGRID_URL, params)
            if not data.get("Rows"):
                raise ValueError("empty Rows")
            out.write_text(json.dumps(data), encoding="utf-8")
            done += 1
        except Exception as e:
            failed += 1
            log_failure(f"eirgrid demand {d} {e}")
        time.sleep(0.5)
    return done, failed


def month_chunks(start, end):
    """Split [start, end] into calendar-month pieces."""
    cur = start
    while cur <= end:
        nxt = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield cur, min(nxt - timedelta(days=1), end)
        cur = nxt


def download_meteo(start, end):
    jobs = [(loc, a, b) for loc in LOCATIONS for a, b in month_chunks(start, end)]
    done = failed = 0
    for loc, a, b in tqdm(jobs, desc="Open-Meteo"):
        out = METEO_DIR / f"{loc}_{a.isoformat()}_{b.isoformat()}.json"
        if out.exists():
            continue
        lat, lon = LOCATIONS[loc]
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "wind_speed_100m,wind_speed_100m_previous_day1",
            "start_date": a.isoformat(), "end_date": b.isoformat(),
            "timezone": "UTC", "models": "best_match", "wind_speed_unit": "ms",
        }
        try:
            data = get_json(METEO_URL, params)
            if "hourly" not in data:
                raise ValueError(f"no hourly data: {str(data)[:200]}")
            out.write_text(json.dumps(data), encoding="utf-8")
            done += 1
        except Exception as e:
            failed += 1
            log_failure(f"openmeteo {loc} {a} {b} {e}")
        time.sleep(0.5)
    return done, failed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2024-06-01")
    p.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat())
    p.add_argument("--skip-eirgrid", action="store_true")
    p.add_argument("--skip-meteo", action="store_true")
    args = p.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    EIRGRID_DIR.mkdir(parents=True, exist_ok=True)
    METEO_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_LOG.unlink(missing_ok=True)  # fresh failure list each run
    print(f"Period: {start} -> {end}")

    if not args.skip_meteo:
        d, f = download_meteo(start, end)
        print(f"Open-Meteo: {d} new files, {f} failed")
    if not args.skip_eirgrid:
        d, f = download_eirgrid(start, end)
        print(f"EirGrid: {d} new files, {f} failed")
        d, f = backfill_demand()
        print(f"Demand backfill: {d} new files, {f} failed")
    if FAILED_LOG.exists():
        print(f"Failures logged in {FAILED_LOG} - just re-run the script to retry them.")


if __name__ == "__main__":
    main()
