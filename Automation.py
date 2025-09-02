# update_stocks_and_crypto.py
import os
import time
import subprocess
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import requests

# ------------------ Config via env ------------------
POLYGON_KEY = os.getenv("POLYGON_API_KEY", "")
if not POLYGON_KEY:
    raise EnvironmentError("POLYGON_API_KEY is not set.")

# Skip crypto by default (your plan doesn’t include it)
INCLUDE_CRYPTO = os.getenv("INCLUDE_CRYPTO", "false").lower() in {"1", "true", "yes"}

# How many calendar days to fetch ending today (ET). Default 2 = yesterday & today.
BACKFILL_DAYS = max(1, int(os.getenv("BACKFILL_DAYS", "2")))

# ------------------ Market-hours guard (ET) ------------------
et_tz = pytz.timezone("US/Eastern")
now_et = datetime.now(et_tz)
weekday = now_et.weekday()             # 0=Mon .. 6=Sun
mins = now_et.hour * 60 + now_et.minute

market_open = 9 * 60 + 30              # 09:30
market_last = 16 * 60 + 55             # 16:55

# Allow manual overrides (set in workflow)
FORCE_RUN = os.getenv("FORCE_RUN", "").lower() in {"1", "true", "yes"}
event_name = os.getenv("GITHUB_EVENT_NAME", "").lower()  # set by Actions

print(f"🕒 ET now: {now_et:%Y-%m-%d %H:%M}  (weekday={weekday}, mins={mins}, event={event_name}, force={FORCE_RUN})")
if not FORCE_RUN and (weekday >= 5 or not (market_open <= mins <= market_last)):
    print("⏳ Market window closed for updater. Skipping.")
    raise SystemExit(0)

# ------------------ Load existing data ------------------
existing_path = "stocks.csv"
if not os.path.exists(existing_path):
    raise FileNotFoundError("stocks.csv not found.")

existing_df = pd.read_csv(existing_path, parse_dates=["date"])

# Ensure key columns exist
if "asset_type" not in existing_df.columns:
    existing_df["asset_type"] = "stock"
if "sector" not in existing_df.columns:
    existing_df["sector"] = np.nan

# ------------------ Symbol meta ------------------
meta_cols = ["symbol", "sector", "asset_type"]
symbol_meta = (
    existing_df[meta_cols]
    .drop_duplicates()
    .sort_values("symbol")
    .reset_index(drop=True)
)

# Normalize asset_type
symbol_meta["asset_type"] = symbol_meta["asset_type"].astype(str).str.lower()

# Default missing crypto sector to "Crypto"
symbol_meta.loc[
    (symbol_meta["asset_type"] == "crypto") & (symbol_meta["sector"].isna()),
    "sector"
] = "Crypto"

# Skip crypto if not included
if not INCLUDE_CRYPTO:
    before = len(symbol_meta)
    symbol_meta = symbol_meta[symbol_meta["asset_type"] != "crypto"]
    print(f"🔧 Skipping crypto: filtered {before - len(symbol_meta)} crypto symbols.")

n_total = len(symbol_meta)
print(f"🧾 Symbols to update: {n_total}")

# ------------------ Date window (ET) ------------------
end_date = now_et.date()
start_date = end_date - timedelta(days=BACKFILL_DAYS - 1)
start_str = start_date.strftime("%Y-%m-%d")
end_str   = end_date.strftime("%Y-%m-%d")
print(f"📅 Fetch window (ET calendar): {start_str} → {end_str}  (days={BACKFILL_DAYS})")

# ------------------ Polygon fetch ------------------
SESSION = requests.Session()

def polygon_ticker(symbol: str, asset_type: str) -> str:
    if asset_type.lower() == "crypto":
        s = symbol.strip()
        return s if s.startswith("X:") else f"X:{s}"
    return symbol.strip()

def fetch_polygon_daily(polygon_symbol: str, start: str, end: str,
                        retries: int = 3, backoff: float = 1.0) -> pd.DataFrame:
    url = f"https://api.polygon.io/v2/aggs/ticker/{polygon_symbol}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_KEY}
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json().get("results", [])
                if not data:
                    return pd.DataFrame()
                df = pd.DataFrame(data)
                if df.empty:
                    return df
                df["date"] = pd.to_datetime(df["t"], unit="ms")  # naive UTC -> we normalize to date below
                df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})[
                    ["date","open","high","low","close","volume"]
                ]
                for c in ["open","high","low","close","volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])
                df = df[df["close"] > 0]
                return df
            else:
                print(f"⚠️ {polygon_symbol}: HTTP {resp.status_code} (attempt {attempt+1}/{retries})")
        except Exception as e:
            print(f"⚠️ {polygon_symbol}: error {e} (attempt {attempt+1}/{retries})")
        time.sleep(backoff * (attempt + 1))
    return pd.DataFrame()

# ------------------ Collect data ------------------
all_frames = []
for i, row in symbol_meta.iterrows():
    symbol = str(row["symbol"]).strip()
    sector = row["sector"]
    a_type = (row["asset_type"] or "stock").lower()
    if not symbol:
        continue
    pt = polygon_ticker(symbol, a_type)
    print(f"📡 Fetching {symbol} ({a_type}) → {pt}  ({i+1}/{len(symbol_meta)})...")
    df_new = fetch_polygon_daily(pt, start_str, end_str)
    if not df_new.empty:
        # Store symbol without 'X:' prefix (consistent with stocks.csv)
        df_new["symbol"] = symbol
        df_new["sector"] = sector if pd.notna(sector) else ( "Crypto" if a_type == "crypto" else None )
        df_new["asset_type"] = a_type
        all_frames.append(df_new)
    time.sleep(0.2)  # polite pacing

# ------------------ Overwrite-first merge (new rows win) ------------------
if not all_frames:
    print("\n⚠️ No new data fetched. File not changed.")
    raise SystemExit(0)

new_data = pd.concat(all_frames, ignore_index=True)
print(f"🟩 New rows fetched: {len(new_data)}")

# Normalize to date (drop intraday time) to match on (symbol,date)
new_data["date"] = pd.to_datetime(new_data["date"]).dt.normalize()
existing_df["date"] = pd.to_datetime(existing_df["date"]).dt.normalize()

# Remove existing rows that collide with new (symbol,date) keys
keys_df = new_data[["symbol", "date"]].drop_duplicates()
before_rows = len(existing_df)
existing_filtered = existing_df.merge(keys_df, on=["symbol", "date"], how="left", indicator=True)
existing_filtered = existing_filtered[existing_filtered["_merge"] == "left_only"].drop(columns="_merge")
dropped = before_rows - len(existing_filtered)
print(f"🗑️ Overwritten existing rows: {dropped}")

# Combine (new first), safety dedupe, sort
combined = pd.concat([new_data, existing_filtered], ignore_index=True)
combined = combined.drop_duplicates(subset=["symbol", "date"], keep="first")
combined = combined.sort_values(["symbol", "date"])
print(f"📊 Final total rows: {len(combined)}")

# Save updated file
combined.to_csv("stocks.csv", index=False)
print("✅ stocks.csv written.")

# ------------------ Push to GitHub (only if changed) ------------------
try:
    subprocess.run(["git", "config", "user.name", "Auto Bot"], check=True)
    subprocess.run(["git", "config", "user.email", "bot@example.com"], check=True)
    subprocess.run(["git", "add", "stocks.csv"], check=True)

    diff = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if diff.returncode != 0:
        msg_scope = "stocks-only" if not INCLUDE_CRYPTO else "stocks+crypto"
        subprocess.run(["git", "commit", "-m", f"🔄 Auto-update {msg_scope} @ {now_et:%Y-%m-%d %H:%M ET}"], check=True)
        subprocess.run(["git", "push"], check=True)
        print("🚀 Pushed update to GitHub.")
    else:
        print("ℹ️ No changes detected after merge; nothing to commit.")
except Exception as e:
    print("⚠️ Git push failed:", e)
