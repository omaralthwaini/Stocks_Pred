import os
import time
import pandas as pd
import requests
from datetime import datetime, timedelta

# --- CONFIG ---
POLYGON_KEY = os.getenv("POLYGON_API_KEY", "ML8KNIhH8hbBS9Cv_w9YcHfwqEpp3IQZ")
EXISTING_FILE = "stocks.csv"
LATEST_FILE = "stocks_latest.csv"
SAVE_PATH = "stocks.csv"  # Overwrites the original after merging
DAYS_BACK = 6  # Today + 6 previous days = last 7 days
RATE_LIMIT_SECONDS = 12  # For free plan

# --- LOAD EXISTING DATA ---
if not os.path.exists(EXISTING_FILE):
    raise FileNotFoundError(f"{EXISTING_FILE} not found")

df_existing = pd.read_csv(EXISTING_FILE, parse_dates=["date"])
symbols_sectors = df_existing[["symbol", "sector"]].drop_duplicates()

# --- DEFINE FETCH FUNCTION ---
def fetch_polygon_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 5000,
        "apiKey": POLYGON_KEY
    }
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json().get("results", [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df["symbol"] = symbol
        return df
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return pd.DataFrame()

# --- DATE RANGE ---
end_date = datetime.today().strftime("%Y-%m-%d")
start_date = (datetime.today() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")

# --- FETCH ALL SYMBOLS ---
latest_frames = []

for i, row in symbols_sectors.iterrows():
    symbol = row["symbol"]
    sector = row["sector"]
    print(f"[{i+1:>3}] Fetching {symbol}...")

    df = fetch_polygon_daily(symbol, start=start_date, end=end_date)
    if not df.empty:
        df["sector"] = sector
        latest_frames.append(df)

    time.sleep(RATE_LIMIT_SECONDS)  # Rate limit for free users

if not latest_frames:
    print("⚠️ No new data was fetched.")
    exit()

# --- COMBINE + CLEAN ---
df_latest = pd.concat(latest_frames, ignore_index=True)
df_combined = (
    pd.concat([df_existing, df_latest], ignore_index=True)
      .drop_duplicates(subset=["symbol", "date"])
      .sort_values(["symbol", "date"])
      .reset_index(drop=True)
)

# --- SAVE ---
df_combined.to_csv(SAVE_PATH, index=False)
print(f"\n✅ Update complete. Combined file saved to {SAVE_PATH}")
print(f"📊 Total rows: {len(df_combined):,} — Unique symbols: {df_combined['symbol'].nunique()}")
