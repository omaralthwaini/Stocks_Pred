import pandas as pd
import requests
import os
import time
from datetime import datetime

# Polygon API Key
POLYGON_KEY = os.getenv("POLYGON_API_KEY", "ML8KNIhH8hbBS9Cv_w9YcHfwqEpp3IQZ")

# Load existing data
existing_path = "stocks.csv"
if not os.path.exists(existing_path):
    raise FileNotFoundError("stocks.csv not found. Please ensure it exists.")

existing_df = pd.read_csv(existing_path, parse_dates=["date"])

# Symbols and sectors to fetch
symbol_sector_map = (
    existing_df[["symbol", "sector"]]
    .drop_duplicates()
    .sort_values("symbol")
    .reset_index(drop=True)
)

# Define historical range (before existing data)
start_date = "2020-01-01"
end_date = "2024-08-13"

# --- Fetch function ---
def fetch_polygon_daily(symbol, start, end):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": POLYGON_KEY
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        print(f"⚠️ {symbol}: Error {resp.status_code}")
        return pd.DataFrame()
    data = resp.json().get("results", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["t"], unit="ms")
    df = df.rename(columns={
        "o": "open", "h": "high", "l": "low",
        "c": "close", "v": "volume"
    })[["date", "open", "high", "low", "close", "volume"]]
    df["symbol"] = symbol
    return df

# --- Fetch historical data ---
all_frames = []

print(f"📅 Fetching historical data from {start_date} to {end_date}")
for i, row in symbol_sector_map.iterrows():
    symbol, sector = row["symbol"], row["sector"]
    print(f"📡 Fetching {symbol} ({i+1}/{len(symbol_sector_map)})...")
    df_hist = fetch_polygon_daily(symbol, start_date, end_date)
    if not df_hist.empty:
        df_hist["sector"] = sector
        all_frames.append(df_hist)
    time.sleep(0.5)

# --- Merge + save ---
if all_frames:
    new_data = pd.concat(all_frames, ignore_index=True)

    # Combine with existing and deduplicate
    combined = pd.concat([existing_df, new_data], ignore_index=True)
    combined = combined.drop_duplicates(subset=["symbol", "date"])
    combined = combined.sort_values(["symbol", "date"])

    combined.to_csv("stocks.csv", index=False)
    print(f"\n✅ stocks.csv updated with total {len(combined):,} rows.")
else:
    print("\n⚠️ No historical data fetched. stocks.csv unchanged.")
