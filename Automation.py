# update_stocks_and_crypto.py
import os
import time
import subprocess
from datetime import datetime, timedelta

import pandas as pd
import pytz
import requests
import numpy as np

# --- API key ---
POLYGON_KEY = os.getenv("POLYGON_API_KEY", "")
if not POLYGON_KEY:
    raise EnvironmentError("POLYGON_API_KEY is not set.")

# --- Market-hours guard: 09:30–16:55 ET, Mon–Fri (kept as-is) ---
now_utc = datetime.utcnow().replace(tzinfo=pytz.UTC)
et = now_utc.astimezone(pytz.timezone("US/Eastern"))
weekday = et.weekday()               # 0=Mon .. 6=Sun
mins = et.hour * 60 + et.minute      # minutes since midnight ET

market_open = 9 * 60 + 30            # 09:30
market_last = 16 * 60 + 55           # 16:55

# Allow manual overrides (set in workflow)
FORCE_RUN = os.getenv("FORCE_RUN", "").lower() in {"1", "true", "yes"}
event_name = os.getenv("GITHUB_EVENT_NAME", "").lower()  # set by Actions

print(f"🕒 ET now: {et:%Y-%m-%d %H:%M}  (weekday={weekday}, mins={mins}, event={event_name}, force={FORCE_RUN})")
if not FORCE_RUN and (weekday >= 5 or not (market_open <= mins <= market_last)):
    print("⏳ Market window closed for updater. Skipping.")
    raise SystemExit(0)

# --- Load existing data ---
existing_path = "stocks.csv"
if not os.path.exists(existing_path):
    raise FileNotFoundError("stocks.csv not found.")

existing_df = pd.read_csv(existing_path, parse_dates=["date"])

# Ensure key columns exist
if "asset_type" not in existing_df.columns:
    existing_df["asset_type"] = "stock"
if "sector" not in existing_df.columns:
    existing_df["sector"] = np.nan

# --- Build symbol → metadata map (symbol, sector, asset_type) ---
meta_cols = ["symbol", "sector", "asset_type"]
symbol_meta = (
    existing_df[meta_cols]
    .drop_duplicates()
    .sort_values("symbol")
    .reset_index(drop=True)
)
# For crypto rows missing sector, default to "Crypto"
symbol_meta.loc[
    (symbol_meta["asset_type"].str.lower() == "crypto")
    & (symbol_meta["sector"].isna()),
    "sector"
] = "Crypto"

n_total = len(symbol_meta)
n_crypto = (symbol_meta["asset_type"].str.lower() == "crypto").sum()
n_stocks = n_total - n_crypto
print(f"🧾 Symbols to update: {n_total}  (stocks={n_stocks}, crypto={n_crypto})")

# --- Date range: yesterday + today (daily bars) ---
today = datetime.now().date()
yesterday = today - timedelta(days=1)
start_date = yesterday.strftime("%Y-%m-%d")
end_date = today.strftime("%Y-%m-%d")
print(f"📅 Fetch window: {start_date} → {end_date}")

# --- Helpers ---
def to_polygon_ticker(symbol: str, asset_type: str) -> str:
    """Stocks use raw ticker (AAPL); crypto uses 'X:BTCUSD' etc."""
    if str(asset_type).lower() == "crypto":
        s = symbol.strip()
        return s if s.startswith("X:") else f"X:{s}"
    return symbol.strip()

SESSION = requests.Session()

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
                df["date"] = pd.to_datetime(df["t"], unit="ms")
                df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})[
                    ["date","open","high","low","close","volume"]
                ]
                # clean
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

# --- Collect data ---
all_frames = []
for i, row in symbol_meta.iterrows():
    symbol = str(row["symbol"]).strip()
    sector = row["sector"]
    asset_type = row["asset_type"]
    if not symbol:
        continue
    poly_ticker = to_polygon_ticker(symbol, asset_type)
    print(f"📡 Fetching {symbol} ({asset_type}) → {poly_ticker}  ({i+1}/{len(symbol_meta)})...")
    df_new = fetch_polygon_daily(poly_ticker, start_date, end_date)
    if not df_new.empty:
        # For storage, we keep symbol *without* 'X:' (consistent with stocks.csv you built)
        df_new["symbol"] = symbol
        df_new["sector"] = sector if pd.notna(sector) else ("Crypto" if str(asset_type).lower()=="crypto" else None)
        df_new["asset_type"] = asset_type if pd.notna(asset_type) else "stock"
        all_frames.append(df_new)
    time.sleep(0.2)  # gentle pacing

# --- Overwrite-first merge (new rows win) ---
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

# --- Push to GitHub (only if changed) ---
try:
    subprocess.run(["git", "config", "user.name", "Auto Bot"], check=True)
    subprocess.run(["git", "config", "user.email", "bot@example.com"], check=True)
    subprocess.run(["git", "add", "stocks.csv"], check=True)

    diff = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if diff.returncode != 0:
        subprocess.run(["git", "commit", "-m", f"🔄 Auto-update stocks+crypto @ {datetime.now()}"], check=True)
        subprocess.run(["git", "push"], check=True)
        print("🚀 Pushed update to GitHub.")
    else:
        print("ℹ️ No changes detected after merge; nothing to commit.")
except Exception as e:
    print("⚠️ Git push failed:", e)
