# Crypto.py
# Crypto-only updater (CoinGecko). Writes into stocks.csv.
# - Runs 24/7
# - Overwrites (symbol, date) collisions for recent crypto data
# - Handles HTTP 429 Rate Limits with exponential backoff & Retry-After support
#
# Expected CSV structure:
# date,open,high,low,close,volume,symbol,asset_type,sector

import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests


# ============================================================
# SETTINGS
# ============================================================

CSV_PATH = "stocks.csv"

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

# Register a FREE Demo Key at https://www.coingecko.com/en/api/pricing
# If provided, rate limits double from ~10 to 30 req/min!
COINGECKO_API_KEY = ""  # e.g., "CG-xxxxxxxxxxxxxxxxxxxx"

# Delay between processing individual coins (in seconds)
# 3.5s keeps requests under ~15-17 requests/minute on public IP
DELAY_BETWEEN_COINS = 3.5

SESSION = requests.Session()


# ============================================================
# LOGGING
# ============================================================

def log(msg: str) -> None:
    print(
        f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC] {msg}"
    )


# ============================================================
# MANUAL OVERRIDES (FOR AMBIGUOUS TICKERS)
# ============================================================

MANUAL_OVERRIDES = {
    "AVAX": "avalanche-2",
    "AVAXUSD": "avalanche-2",
    "AVAXUSDT": "avalanche-2",
    "BNB": "binancecoin",
    "BNBUSD": "binancecoin",
    "BNBUSDT": "binancecoin",
}


# ============================================================
# HELPER: API REQUEST WRAPPER WITH RATE LIMIT BACKOFF
# ============================================================

def make_coingecko_request(url: str, params: dict = None, max_retries: int = 5) -> requests.Response:
    """
    Executes a GET request to CoinGecko with automatic handling for HTTP 429 rate limits.
    """
    if params is None:
        params = {}

    headers = {}
    if COINGECKO_API_KEY:
        headers["x-cg-demo-api-key"] = COINGECKO_API_KEY

    backoff = 10.0  # Base delay in seconds when rate limited

    for attempt in range(1, max_retries + 1):
        try:
            response = SESSION.get(
                url,
                params=params,
                headers=headers,
                timeout=30
            )

            # Success
            if response.status_code == 200:
                return response

            # Rate Limit Encountered (HTTP 429)
            elif response.status_code == 429:
                # Check for Retry-After header sent by server
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait_time = int(retry_after) + 1
                else:
                    wait_time = backoff * attempt  # 10s, 20s, 30s...

                log(f"⚠️ Rate limited (HTTP 429). Pausing execution for {wait_time:.1f}s... (Attempt {attempt}/{max_retries})")
                time.sleep(wait_time)

            # Server Error (HTTP 5xx)
            elif response.status_code in [500, 502, 503, 504]:
                log(f"⚠️ CoinGecko Server Error (HTTP {response.status_code}). Retrying in 5s...")
                time.sleep(5)

            else:
                return response

        except Exception as e:
            log(f"⚠️ Connection error: {e}. Retrying in 5s... (Attempt {attempt}/{max_retries})")
            time.sleep(5)

    return None


# ============================================================
# DYNAMIC COINGECKO SYMBOL RESOLUTION
# ============================================================

def build_coin_map(symbols: list) -> dict:
    """
    Fetch all coins from CoinGecko and automatically map input symbols 
    (e.g., AEROUSD, ALGOUSDT, BTC) to CoinGecko coin IDs.
    """
    log("🔄 Fetching coin list from CoinGecko...")
    url = f"{COINGECKO_BASE_URL}/coins/list"
    
    response = make_coingecko_request(url)
    if not response or response.status_code != 200:
        log("⚠️ Failed to fetch coin list. Falling back to manual overrides.")
        return MANUAL_OVERRIDES

    try:
        coins_list = response.json()
    except Exception as e:
        log(f"⚠️ Parsing coin list failed: {e}. Falling back to manual overrides.")
        return MANUAL_OVERRIDES

    symbol_to_id = {}
    for coin in coins_list:
        sym = coin.get("symbol", "").lower()
        coin_id = coin.get("id", "")
        if sym and coin_id:
            symbol_to_id[sym] = coin_id

    mapping = {}
    suffixes = ["USDT", "USDC", "BUSD", "USD"]

    for raw_sym in symbols:
        clean_sym = str(raw_sym).strip().upper()
        if not clean_sym:
            continue

        if clean_sym in MANUAL_OVERRIDES:
            mapping[clean_sym] = MANUAL_OVERRIDES[clean_sym]
            continue

        base_sym = clean_sym
        for suffix in suffixes:
            if base_sym.endswith(suffix) and len(base_sym) > len(suffix):
                base_sym = base_sym[:-len(suffix)]
                break

        coin_id = symbol_to_id.get(base_sym.lower())
        if coin_id:
            mapping[clean_sym] = coin_id

    return mapping


# ============================================================
# COINGECKO API: FETCH OHLC
# ============================================================

def fetch_coingecko_ohlc(coin_id: str, days: int = 1) -> pd.DataFrame:
    """
    Fetch OHLC data from CoinGecko endpoint: /api/v3/coins/{id}/ohlc
    """
    url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/ohlc"
    params = {
        "vs_currency": "usd",
        "days": days
    }

    response = make_coingecko_request(url, params=params)

    if not response or response.status_code != 200:
        log(f"⚠️ CoinGecko {coin_id}: failed to retrieve valid data.")
        return pd.DataFrame()

    data = response.json()
    if not data:
        log(f"⚠️ CoinGecko {coin_id}: empty response")
        return pd.DataFrame()

    rows = []
    for item in data:
        if len(item) < 5:
            continue

        timestamp = item[0]
        open_price = item[1]
        high_price = item[2]
        low_price = item[3]
        close_price = item[4]

        date = datetime.fromtimestamp(
            timestamp / 1000,
            tz=timezone.utc
        ).strftime("%Y-%m-%d")

        rows.append({
            "date": date,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": np.nan
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Convert numeric columns
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Clean missing/invalid values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(CSV_PATH)

    df0 = pd.read_csv(
        CSV_PATH,
        parse_dates=["date"]
    )

    if "asset_type" not in df0.columns:
        df0["asset_type"] = "stock"

    if "sector" not in df0.columns:
        df0["sector"] = np.nan

    df0["asset_type"] = df0["asset_type"].astype(str).str.lower()

    meta = (
        df0[df0["asset_type"].eq("crypto")][["symbol", "sector", "asset_type"]]
        .drop_duplicates()
        .sort_values("symbol")
        .reset_index(drop=True)
    )

    if meta.empty:
        log("ℹ️ No crypto symbols in file. Nothing to do.")
        return

    log(f"Found {len(meta)} crypto symbols in {CSV_PATH}.")

    # Dynamically build CoinGecko symbol mapping
    coin_map = build_coin_map(meta["symbol"].unique().tolist())

    frames = []

    for i, row in meta.iterrows():

        sym = str(row["symbol"]).strip().upper()
        if not sym:
            continue

        coin_id = coin_map.get(sym)

        if not coin_id:
            log(f"⚠️ {sym}: no CoinGecko mapping. Skipping.")
            continue

        log(f"📡 CRYPTO {sym} -> CoinGecko {coin_id} ({i + 1}/{len(meta)})")

        df_new = fetch_coingecko_ohlc(coin_id=coin_id, days=1)

        if df_new.empty:
            log(f"⚠️ {sym}: no data returned.")
            continue

        df_new["symbol"] = sym
        df_new["asset_type"] = "crypto"
        df_new["sector"] = row["sector"] if pd.notna(row["sector"]) else "Crypto"

        frames.append(df_new)

        # Gentle pacing between requests to respect public API quotas
        time.sleep(DELAY_BETWEEN_COINS)

    if not frames:
        log("ℹ️ No new crypto data fetched.")
        return

    new_data = pd.concat(frames, ignore_index=True)

    new_data["date"] = pd.to_datetime(new_data["date"]).dt.normalize()
    df0["date"] = pd.to_datetime(df0["date"]).dt.normalize()

    # Identify rows to overwrite
    keys = new_data[["symbol", "date"]].drop_duplicates()

    before = len(df0)

    existing_filtered = (
        df0
        .merge(
            keys,
            on=["symbol", "date"],
            how="left",
            indicator=True
        )
        .loc[lambda x: x["_merge"] == "left_only"]
        .drop(columns="_merge")
    )

    dropped = before - len(existing_filtered)
    log(f"🗑️ Overwritten rows (crypto): {dropped}")

    combined = (
        pd.concat([new_data, existing_filtered], ignore_index=True)
        .drop_duplicates(subset=["symbol", "date"], keep="first")
        .sort_values(["symbol", "date"])
    )

    preferred = [
        "date", "open", "high", "low", "close", "volume",
        "symbol", "asset_type", "sector"
    ]

    cols = [c for c in preferred if c in combined.columns] + \
           [c for c in combined.columns if c not in preferred]

    combined[cols].to_csv(CSV_PATH, index=False)
    log(f"✅ Wrote {CSV_PATH} ({len(combined):,} rows).")


if __name__ == "__main__":
    main()