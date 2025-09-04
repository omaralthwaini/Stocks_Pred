# Automation.py
# Updates stocks (Polygon) and crypto (Binance) into stocks.csv.
# For now we only update STOCKS (fast): set UPDATE_CRYPTO=false in the workflow.
# - Stocks: only during US market hours (unless FORCE_RUN=1/true/yes)
# - Crypto: behind a toggle (UPDATE_CRYPTO) for later
# - Overwrites (symbol, date) rows in the file with fresh data

import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytz
import requests

# -------------------------------
# Config / env
# -------------------------------
CSV_PATH      = "stocks.csv"
POLYGON_KEY   = os.getenv("POLYGON_API_KEY", "").strip()
FORCE_RUN     = os.getenv("FORCE_RUN", "").lower() in {"1", "true", "yes"}

# Toggle which assets to update (workflow sets stocks=True, crypto=False for speed)
UPDATE_STOCKS = os.getenv("UPDATE_STOCKS", "true").lower() in {"1", "true", "yes"}
UPDATE_CRYPTO = os.getenv("UPDATE_CRYPTO", "false").lower() in {"1", "true", "yes"}

# How many days back to fetch for stocks (0 = today only)
STOCKS_WINDOW_DAYS = int(os.getenv("STOCKS_WINDOW_DAYS", "0"))

SESSION = requests.Session()

# -------------------------------
# Logging
# -------------------------------
def log(msg: str) -> None:
    print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] {msg}")

# -------------------------------
# Time utilities
# -------------------------------
def market_hours_ok_now_et() -> bool:
    """True during 09:30–16:55 ET on Mon–Fri."""
    now_utc = datetime.utcnow().replace(tzinfo=pytz.UTC)
    et = now_utc.astimezone(pytz.timezone("US/Eastern"))
    weekday = et.weekday()               # 0=Mon..6=Sun
    mins    = et.hour * 60 + et.minute
    open_m  = 9 * 60 + 30
    last_m  = 16 * 60 + 55
    return (weekday < 5) and (open_m <= mins <= last_m)

def day_start_ms(d_str: str) -> int:
    d = datetime.strptime(d_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(d.timestamp() * 1000)

def day_end_ms(d_str: str) -> int:
    d = datetime.strptime(d_str, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
    return int(d.timestamp() * 1000) - 1  # inclusive

# -------------------------------
# Binance symbol helper (kept for later)
# -------------------------------
def to_binance_symbol(symbol: str) -> str:
    """
    Convert unified USD symbols to Binance USDT spot pairs.
    e.g., BTCUSD -> BTCUSDT, SOLUSD -> SOLUSDT
    """
    s = symbol.upper()
    return (s[:-3] + "USDT") if s.endswith("USD") else s

# -------------------------------
# Fetchers
# -------------------------------
def fetch_polygon_daily_stock(symbol: str, start: str, end: str,
                              retries: int = 3, backoff: float = 1.0) -> pd.DataFrame:
    if not POLYGON_KEY:
        log("⚠️  POLYGON_API_KEY not set; skipping stock fetch.")
        return pd.DataFrame()

    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_KEY}

    for attempt in range(1, retries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json().get("results", [])
                if not data:
                    return pd.DataFrame()
                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(None)
                df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})[
                    ["date", "open", "high", "low", "close", "volume"]
                ]
                for c in ["open", "high", "low", "close", "volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])
                df = df[df["close"] > 0]
                return df
            else:
                log(f"⚠️  Polygon {symbol}: HTTP {r.status_code} (attempt {attempt}/{retries})")
        except Exception as e:
            log(f"⚠️  Polygon {symbol}: error {e} (attempt {attempt}/{retries})")
        time.sleep(backoff * attempt)

    return pd.DataFrame()

def fetch_binance_daily_crypto(symbol: str, start: str, end: str,
                               retries: int = 3, backoff: float = 1.0) -> pd.DataFrame:
    """Public Binance spot /api/v3/klines (interval=1d)."""
    base = "https://api.binance.com"
    url = f"{base}/api/v3/klines"
    b_symbol = to_binance_symbol(symbol)

    params = {
        "symbol": b_symbol,
        "interval": "1d",
        "startTime": day_start_ms(start),
        "endTime": day_end_ms(end),
        "limit": 1000,
    }

    for attempt in range(1, retries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=30)
            if r.status_code == 200:
                rows = r.json()
                if not rows:
                    return pd.DataFrame()
                df = pd.DataFrame(rows, columns=[
                    "open_time","open","high","low","close","volume",
                    "close_time","qav","num_trades","taker_buy_base","taker_buy_quote","ignore"
                ])
                df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
                df = df[["date", "open", "high", "low", "close", "volume"]].copy()
                for c in ["open", "high", "low", "close", "volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])
                df = df[df["close"] > 0]
                return df
            else:
                log(f"⚠️  Binance {symbol}→{b_symbol}: HTTP {r.status_code} (attempt {attempt}/{retries})")
        except Exception as e:
            log(f"⚠️  Binance {symbol}→{b_symbol}: error {e} (attempt {attempt}/{retries})")
        time.sleep(backoff * attempt)

    return pd.DataFrame()

# -------------------------------
# Main
# -------------------------------
def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found.")

    # Load current file
    df_existing = pd.read_csv(CSV_PATH, parse_dates=["date"])
    if "asset_type" not in df_existing.columns:
        df_existing["asset_type"] = "stock"
    if "sector" not in df_existing.columns:
        df_existing["sector"] = np.nan

    # Normalize
    df_existing["asset_type"] = df_existing["asset_type"].astype(str).str.lower()
    df_existing.loc[
        (df_existing["asset_type"] == "crypto") & (df_existing["sector"].isna()),
        "sector"
    ] = "Crypto"

    # Symbol meta
    meta = (df_existing[["symbol", "asset_type", "sector"]]
            .drop_duplicates()
            .sort_values("symbol")
            .reset_index(drop=True))

    # Decide allowed
    allow_stocks = UPDATE_STOCKS and (FORCE_RUN or market_hours_ok_now_et())
    allow_crypto = UPDATE_CRYPTO

    now_utc = datetime.utcnow().replace(tzinfo=pytz.UTC)
    et = now_utc.astimezone(pytz.timezone("US/Eastern"))
    log(f"ET now: {et:%Y-%m-%d %H:%M} | allow_stocks={allow_stocks} | allow_crypto={allow_crypto}")

    # Filter meta early for speed
    before = len(meta)
    if not allow_crypto:
        meta = meta[meta["asset_type"] == "stock"].reset_index(drop=True)
    if not allow_stocks and not allow_crypto:
        log("ℹ️ Nothing to do (both assets disabled/out of hours).")
        return
    log(f"Symbols to update: {len(meta)} (filtered from {before})")

    # Windows
    today = datetime.now(timezone.utc).date()
    # Stocks: configurable small window (0 => today only)
    start_stock = (today - timedelta(days=max(0, STOCKS_WINDOW_DAYS))).strftime("%Y-%m-%d")
    end_stock   = today.strftime("%Y-%m-%d")
    # Crypto: yesterday → today (kept for later)
    start_crypto = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    end_crypto   = end_stock

    frames = []
    for i, row in meta.iterrows():
        sym = str(row["symbol"]).strip()
        a_type = str(row["asset_type"]).lower()
        sector = row["sector"]

        if not sym:
            continue

        if a_type == "stock":
            if not allow_stocks:
                continue
            log(f"📡 STOCK  {sym}  ({i+1}/{len(meta)})")
            df_new = fetch_polygon_daily_stock(sym, start_stock, end_stock)

        elif a_type == "crypto":
            if not allow_crypto:
                continue
            log(f"📡 CRYPTO {sym}  ({i+1}/{len(meta)})")
            df_new = fetch_binance_daily_crypto(sym, start_crypto, end_crypto)

        else:
            log(f"➡️  Skip {sym} (unknown asset_type={a_type})")
            continue

        if df_new.empty:
            continue

        df_new["symbol"] = sym
        df_new["asset_type"] = a_type
        df_new["sector"] = sector if pd.notna(sector) else ("Crypto" if a_type == "crypto" else None)
        frames.append(df_new)

        # Gentle pacing (Polygon rate limits)
        time.sleep(0.15 if a_type == "stock" else 0.20)

    if not frames:
        log("ℹ️ Nothing fetched this run.")
        return

    new_data = pd.concat(frames, ignore_index=True)
    log(f"🟩 New rows fetched: {len(new_data)}")

    # Normalize dates and overwrite collisions on (symbol, date)
    new_data["date"] = pd.to_datetime(new_data["date"]).dt.normalize()
    df_existing["date"] = pd.to_datetime(df_existing["date"]).dt.normalize()

    keys = new_data[["symbol", "date"]].drop_duplicates()
    before_rows = len(df_existing)
    existing_filtered = (
        df_existing.merge(keys, on=["symbol", "date"], how="left", indicator=True)
                   .loc[lambda x: x["_merge"] == "left_only"]
                   .drop(columns="_merge")
    )
    dropped = before_rows - len(existing_filtered)
    log(f"🗑️  Overwritten existing rows: {dropped}")

    combined = pd.concat([new_data, existing_filtered], ignore_index=True)
    combined = combined.drop_duplicates(subset=["symbol", "date"], keep="first").sort_values(["symbol", "date"])

    # Friendly column order
    preferred = ["date","open","high","low","close","volume","symbol","asset_type","sector"]
    cols = [c for c in preferred if c in combined.columns] + [c for c in combined.columns if c not in preferred]
    combined = combined[cols]

    combined.to_csv(CSV_PATH, index=False)
    log(f"✅ Wrote {CSV_PATH} with {len(combined):,} rows.")

if __name__ == "__main__":
    main()
