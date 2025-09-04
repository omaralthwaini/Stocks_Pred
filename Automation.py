# update_stocks_and_crypto.py
import os
import time
import subprocess
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytz
import requests

# ==============================
# Config / env
# ==============================
POLYGON_KEY = os.getenv("POLYGON_API_KEY", "")
FORCE_RUN   = os.getenv("FORCE_RUN", "").lower() in {"1", "true", "yes"}
EVENT_NAME  = os.getenv("GITHUB_EVENT_NAME", "").lower()  # set by Actions

STOCKS_CSV = "stocks.csv"
SESSION = requests.Session()

# ==============================
# Pretty logging
# ==============================
def log(msg: str) -> None:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now} UTC] {msg}")

# ==============================
# Market-hours guard (for STOCKS only)
# ==============================
now_utc = datetime.utcnow().replace(tzinfo=pytz.UTC)
et = now_utc.astimezone(pytz.timezone("US/Eastern"))
weekday = et.weekday()               # 0=Mon .. 6=Sun
mins    = et.hour * 60 + et.minute   # minutes since midnight ET
market_open = 9 * 60 + 30            # 09:30
market_last = 16 * 60 + 55           # 16:55

allow_stocks = FORCE_RUN or (weekday < 5 and market_open <= mins <= market_last)
# Crypto can always update
allow_crypto = True

log(f"ET now: {et:%Y-%m-%d %H:%M} | weekday={weekday} | event={EVENT_NAME} | FORCE_RUN={FORCE_RUN}")
log(f"Stocks update allowed: {allow_stocks} | Crypto update allowed: {allow_crypto}")

# ==============================
# Load existing data
# ==============================
if not os.path.exists(STOCKS_CSV):
    raise FileNotFoundError(f"{STOCKS_CSV} not found.")

df_existing = pd.read_csv(STOCKS_CSV, parse_dates=["date"])
if "asset_type" not in df_existing.columns:
    df_existing["asset_type"] = "stock"
if "sector" not in df_existing.columns:
    df_existing["sector"] = np.nan

# Normalize
df_existing["asset_type"] = df_existing["asset_type"].astype(str).str.lower()
df_existing.loc[
    (df_existing["asset_type"] == "Crypto") & (df_existing["sector"].isna()),
    "sector"
] = "Crypto"

# Build symbol meta
meta_cols = ["symbol", "sector", "asset_type"]
symbol_meta = (
    df_existing[meta_cols]
    .drop_duplicates()
    .sort_values("symbol")
    .reset_index(drop=True)
)

n_total   = len(symbol_meta)
n_crypto  = (symbol_meta["asset_type"] == "Crypto").sum()
n_stocks  = n_total - n_crypto
log(f"Symbols to update: {n_total}  (stocks={n_stocks}, crypto={n_crypto})")

# ==============================
# Date window: yesterday + today
# ==============================
today = datetime.now(timezone.utc).date()
yesterday = today - timedelta(days=1)
start_date_str = yesterday.strftime("%Y-%m-%d")
end_date_str   = today.strftime("%Y-%m-%d")
log(f"Fetch window (daily): {start_date_str} → {end_date_str}")

# Helpers to convert date strings to Binance ms (UTC midnight boundaries)
def _day_start_ms(d_str: str) -> int:
    d = datetime.strptime(d_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(d.timestamp() * 1000)

def _day_end_ms(d_str: str) -> int:
    d = datetime.strptime(d_str, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
    return int(d.timestamp() * 1000) - 1  # inclusive end

# ==============================
# Polygon (stocks) helper
# ==============================
def fetch_polygon_daily_stock(symbol: str, start: str, end: str,
                              retries: int = 3, backoff: float = 1.0) -> pd.DataFrame:
    if not POLYGON_KEY:
        log("⚠️ POLYGON_API_KEY not set; skipping Polygon fetch.")
        return pd.DataFrame()
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_KEY}
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json().get("results", [])
                if not data:
                    return pd.DataFrame()
                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(None)
                df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})[
                    ["date","open","high","low","close","volume"]
                ]
                for c in ["open","high","low","close","volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])
                df = df[df["close"] > 0]
                return df
            else:
                log(f"⚠️ Polygon {symbol}: HTTP {r.status_code} (attempt {attempt+1}/{retries})")
        except Exception as e:
            log(f"⚠️ Polygon {symbol}: error {e} (attempt {attempt+1}/{retries})")
        time.sleep(backoff * (attempt + 1))
    return pd.DataFrame()

# ==============================
# Binance (crypto) helper
# ==============================
def fetch_binance_daily_crypto(symbol: str, start: str, end: str,
                               retries: int = 3, backoff: float = 1.0) -> pd.DataFrame:
    """
    symbol like 'BTCUSDT'. Uses /api/v3/klines interval=1d. Public endpoint—no API key.
    """
    base = "https://api.binance.com"   # spot market
    url = f"{base}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": "1d",
        "startTime": _day_start_ms(start),
        "endTime": _day_end_ms(end),
        "limit": 1000,
    }
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=30)
            if r.status_code == 200:
                rows = r.json()
                if not rows:
                    return pd.DataFrame()
                # columns: [openTime, open, high, low, close, volume, closeTime, ...]
                df = pd.DataFrame(rows, columns=[
                    "open_time","open","high","low","close","volume",
                    "close_time","qav","num_trades","taker_buy_base","taker_buy_quote","ignore"
                ])
                df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
                df = df[["date","open","high","low","close","volume"]].copy()
                for c in ["open","high","low","close","volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])
                df = df[df["close"] > 0]
                return df
            else:
                log(f"⚠️ Binance {symbol}: HTTP {r.status_code} (attempt {attempt+1}/{retries})")
        except Exception as e:
            log(f"⚠️ Binance {symbol}: error {e} (attempt {attempt+1}/{retries})")
        time.sleep(backoff * (attempt + 1))
    return pd.DataFrame()

# ==============================
# Collect data
# ==============================
frames = []

for i, row in symbol_meta.iterrows():
    sym = str(row["symbol"]).strip()
    a_type = str(row["asset_type"]).lower()
    sector = row["sector"]

    if not sym:
        continue

    if a_type == "stock":
        if not allow_stocks:
            continue
        log(f"📡 STOCK  {sym}  ({i+1}/{len(symbol_meta)})")
        df_new = fetch_polygon_daily_stock(sym, start_date_str, end_date_str)
    elif a_type == "Crypto":
        if not allow_crypto:
            continue
        log(f"📡 CRYPTO {sym}  ({i+1}/{len(symbol_meta)})")
        df_new = fetch_binance_daily_crypto(sym, start_date_str, end_date_str)
    else:
        # unknown asset types are skipped
        log(f"➡️  Skip {sym} (unknown asset_type={a_type})")
        continue

    if df_new.empty:
        continue

    # attach metadata for storage
    df_new["symbol"] = sym
    df_new["asset_type"] = a_type
    df_new["sector"] = sector if pd.notna(sector) else ("Crypto" if a_type == "Crypto" else None)
    frames.append(df_new)

    # light pacing
    time.sleep(0.2)

if not frames:
    log("⚠️ No new data fetched. File not changed.")
    raise SystemExit(0)

new_data = pd.concat(frames, ignore_index=True)
log(f"🟩 New rows fetched: {len(new_data)}")

# Normalize dates to calendar day for keying
new_data["date"] = pd.to_datetime(new_data["date"]).dt.normalize()
df_existing["date"] = pd.to_datetime(df_existing["date"]).dt.normalize()

# Overwrite any collisions on (symbol, date)
keys_df = new_data[["symbol", "date"]].drop_duplicates()
before_rows = len(df_existing)
existing_filtered = df_existing.merge(keys_df, on=["symbol", "date"], how="left", indicator=True)
existing_filtered = existing_filtered[existing_filtered["_merge"] == "left_only"].drop(columns="_merge")
dropped = before_rows - len(existing_filtered)
log(f"🗑️ Overwritten existing rows: {dropped}")

# Combine → dedupe → sort
combined = pd.concat([new_data, existing_filtered], ignore_index=True)
combined = combined.drop_duplicates(subset=["symbol","date"], keep="first").sort_values(["symbol","date"])
log(f"📊 Final total rows: {len(combined)}")

# Save
combined.to_csv(STOCKS_CSV, index=False)
log("✅ stocks.csv written.")

# ==============================
# Git commit/push (if changed)
# ==============================
try:
    subprocess.run(["git", "config", "user.name", "Auto Bot"], check=True)
    subprocess.run(["git", "config", "user.email", "bot@example.com"], check=True)
    subprocess.run(["git", "add", STOCKS_CSV], check=True)

    diff = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if diff.returncode != 0:
        subprocess.run(["git", "commit", "-m", f"🔄 Auto-update stocks+crypto @ {datetime.now()}"], check=True)
        subprocess.run(["git", "push"], check=True)
        log("🚀 Pushed update to GitHub.")
    else:
        log("ℹ️ No changes detected after merge; nothing to commit.")
except Exception as e:
    log(f"⚠️ Git push failed: {e}")
