# Automation.py
# Stocks-only updater (Polygon). Writes into stocks.csv.
# - Runs only during US market hours unless FORCE_RUN=true/1/yes
# - Overwrites (symbol, date) collisions for yesterday & today
# - Leaves crypto rows untouched

import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytz
import requests

CSV_PATH    = "stocks.csv"
POLYGON_KEY = os.getenv("POLYGON_API_KEY", "").strip()
FORCE_RUN   = os.getenv("FORCE_RUN", "").lower() in {"1", "true", "yes"}
SESSION     = requests.Session()

def log(msg: str) -> None:
    print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] {msg}")

def market_hours_ok_now_et() -> bool:
    """True during 09:30–16:55 ET on Mon–Fri."""
    now_utc = datetime.utcnow().replace(tzinfo=pytz.UTC)
    et = now_utc.astimezone(pytz.timezone("US/Eastern"))
    wd = et.weekday()  # 0=Mon..6=Sun
    mins = et.hour * 60 + et.minute
    return (wd < 5) and (9 * 60 + 30 <= mins <= 16 * 60 + 55)

def fetch_polygon_daily(symbol: str, start: str, end: str,
                        retries: int = 3, backoff: float = 1.0) -> pd.DataFrame:
    if not POLYGON_KEY:
        log("⚠️  POLYGON_API_KEY not set; skipping.")
        return pd.DataFrame()
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_KEY}
    for attempt in range(1, retries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=30)
            if r.status_code == 200:
                rows = r.json().get("results", [])
                if not rows:
                    return pd.DataFrame()
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(None)
                df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})[
                    ["date","open","high","low","close","volume"]
                ]
                for c in ["open","high","low","close","volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])
                df = df[df["close"] > 0]
                return df
            log(f"⚠️  Polygon {symbol}: HTTP {r.status_code} (attempt {attempt}/{retries})")
        except Exception as e:
            log(f"⚠️  Polygon {symbol}: error {e} (attempt {attempt}/{retries})")
        time.sleep(backoff * attempt)
    return pd.DataFrame()

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(CSV_PATH)

    df0 = pd.read_csv(CSV_PATH, parse_dates=["date"])
    if "asset_type" not in df0.columns:
        df0["asset_type"] = "stock"
    if "sector" not in df0.columns:
        df0["sector"] = np.nan
    df0["asset_type"] = df0["asset_type"].astype(str).str.lower()

    # only stocks
    meta = (df0[df0["asset_type"].eq("stock")][["symbol","sector","asset_type"]]
            .drop_duplicates().sort_values("symbol").reset_index(drop=True))
    if meta.empty:
        log("ℹ️  No stock symbols in file. Nothing to do.")
        return

    allow = FORCE_RUN or market_hours_ok_now_et()
    now_utc = datetime.utcnow().replace(tzinfo=pytz.UTC)
    et = now_utc.astimezone(pytz.timezone("US/Eastern"))
    log(f"ET now: {et:%Y-%m-%d %H:%M} | allow_stocks={allow}")
    if not allow:
        log("⏭️  Outside market hours (and FORCE_RUN not set). Exiting.")
        return

    today = datetime.now(timezone.utc).date()
    start = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    end   = today.strftime("%Y-%m-%d")
    log(f"Fetch window: {start} → {end}")

    frames = []
    for i, row in meta.iterrows():
        sym = str(row["symbol"]).strip()
        if not sym:
            continue
        log(f"📡 STOCK {sym} ({i+1}/{len(meta)})")
        df_new = fetch_polygon_daily(sym, start, end)
        if df_new.empty:
            continue
        df_new["symbol"] = sym
        df_new["asset_type"] = "stock"
        df_new["sector"] = row["sector"]
        frames.append(df_new)
        time.sleep(0.2)

    if not frames:
        log("ℹ️  No new data fetched.")
        return

    new_data = pd.concat(frames, ignore_index=True)
    new_data["date"] = pd.to_datetime(new_data["date"]).dt.normalize()
    df0["date"] = pd.to_datetime(df0["date"]).dt.normalize()

    keys = new_data[["symbol","date"]].drop_duplicates()
    before = len(df0)
    existing_filtered = (df0.merge(keys, on=["symbol","date"], how="left", indicator=True)
                           .loc[lambda x: x["_merge"] == "left_only"]
                           .drop(columns="_merge"))
    dropped = before - len(existing_filtered)
    log(f"🗑️  Overwritten rows (stocks): {dropped}")

    combined = (pd.concat([new_data, existing_filtered], ignore_index=True)
                  .drop_duplicates(subset=["symbol","date"], keep="first")
                  .sort_values(["symbol","date"]))

    preferred = ["date","open","high","low","close","volume","symbol","asset_type","sector"]
    cols = [c for c in preferred if c in combined.columns] + [c for c in combined.columns if c not in preferred]
    combined[cols].to_csv(CSV_PATH, index=False)
    log(f"✅ Wrote {CSV_PATH} ({len(combined):,} rows).")

if __name__ == "__main__":
    main()
