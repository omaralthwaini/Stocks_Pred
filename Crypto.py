# Crypto.py
# Crypto-only updater (CryptoCompare). Writes into stocks.csv.
# - Runs 24/7
# - Overwrites (symbol, date) collisions for yesterday & today (UTC)
# - Expects crypto rows in stocks.csv with asset_type == "crypto"
# - Symbols like BTCUSD, ETHUSDT, SOLUSD, etc.

import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

CSV_PATH   = "stocks.csv"
CC_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY", "").strip()
SESSION    = requests.Session()

def log(msg: str) -> None:
    print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] {msg}")

def parse_symbol(symbol: str) -> tuple[str, str]:
    """Split 'BTCUSD' -> ('BTC', 'USD'), 'ETHUSDT' -> ('ETH','USDT')."""
    s = symbol.upper()
    for q in ("USDT", "USD", "EUR", "GBP", "BTC", "ETH"):
        if s.endswith(q):
            return s[:-len(q)], q
    # fallback
    return s, "USD"

def end_of_today_utc_ts() -> int:
    now = datetime.now(timezone.utc)
    eod = now.replace(hour=23, minute=59, second=59, microsecond=0)
    return int(eod.timestamp())

def fetch_cc_histoday(fsym: str, tsym: str, limit: int = 2, to_ts: int | None = None,
                      retries: int = 3, backoff: float = 1.0) -> pd.DataFrame:
    """
    CryptoCompare /data/v2/histoday
    Returns last `limit+1` days up to `to_ts` (inclusive). We use limit=2 to cover yesterday & today.
    """
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    headers = {}
    if CC_API_KEY:
      headers["authorization"] = f"Apikey {CC_API_KEY}"

    params = {"fsym": fsym, "tsym": tsym, "limit": max(2, int(limit)), "aggregate": 1}
    if to_ts:
        params["toTs"] = int(to_ts)

    for attempt in range(1, retries + 1):
        try:
            r = SESSION.get(url, params=params, headers=headers, timeout=30)
            if r.status_code == 200:
                js = r.json()
                if js.get("Response") == "Error" or js.get("Type") == 99:
                    log(f"⚠️  CC {fsym}{tsym}: API error: {js.get('Message')}")
                    return pd.DataFrame()
                data = (js.get("Data") or {}).get("Data", [])
                if not data:
                    return pd.DataFrame()
                df = pd.DataFrame(data)
                # expected cols: time, open, high, low, close, volumefrom, volumeto
                df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)
                df = df.rename(columns={"volumefrom": "volume"})[
                    ["date", "open", "high", "low", "close", "volume"]
                ]
                for c in ["open","high","low","close","volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])
                df = df[df["close"] > 0]
                return df
            else:
                log(f"⚠️  CC {fsym}{tsym}: HTTP {r.status_code} (attempt {attempt}/3)")
        except Exception as e:
            log(f"⚠️  CC {fsym}{tsym}: error {e} (attempt {attempt}/3)")
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

    meta = (df0[df0["asset_type"].eq("crypto")][["symbol","sector","asset_type"]]
            .drop_duplicates().sort_values("symbol").reset_index(drop=True))
    if meta.empty:
        log("ℹ️  No crypto symbols in file. Nothing to do.")
        return

    to_ts = end_of_today_utc_ts()
    log(f"Fetch window: yesterday & today (UTC) up to {datetime.utcfromtimestamp(to_ts)}Z")
    frames = []

    for i, row in meta.iterrows():
        sym = str(row["symbol"]).strip()
        if not sym:
            continue
        fsym, tsym = parse_symbol(sym)
        log(f"📡 CRYPTO {sym} ({fsym}/{tsym})  ({i+1}/{len(meta)})")
        df_new = fetch_cc_histoday(fsym, tsym, limit=2, to_ts=to_ts)
        if df_new.empty:
            continue
        df_new["symbol"] = sym
        df_new["asset_type"] = "crypto"
        df_new["sector"] = row["sector"] if pd.notna(row["sector"]) else "Crypto"
        frames.append(df_new)
        time.sleep(0.15)  # gentle pacing

    if not frames:
        log("ℹ️  No new crypto data fetched.")
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
    log(f"🗑️  Overwritten rows (crypto): {dropped}")

    combined = (pd.concat([new_data, existing_filtered], ignore_index=True)
                  .drop_duplicates(subset=["symbol","date"], keep="first")
                  .sort_values(["symbol","date"]))

    preferred = ["date","open","high","low","close","volume","symbol","asset_type","sector"]
    cols = [c for c in preferred if c in combined.columns] + [c for c in combined.columns if c not in preferred]
    combined[cols].to_csv(CSV_PATH, index=False)
    log(f"✅ Wrote {CSV_PATH} ({len(combined):,} rows).")

if __name__ == "__main__":
    main()
