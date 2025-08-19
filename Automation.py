import pandas as pd
import requests
import os
import time
from datetime import datetime, timedelta
import pytz
import subprocess

# Polygon API Key from environment (in GitHub Actions, use repo secret)
POLYGON_KEY = os.getenv("POLYGON_API_KEY", "")

# Abort if key is missing
if not POLYGON_KEY:
    raise EnvironmentError("POLYGON_API_KEY is not set.")

# --- Market hours check (9:30am to 4pm ET, Mon-Fri) ---
now_utc = datetime.utcnow().replace(tzinfo=pytz.UTC)
now_et = now_utc.astimezone(pytz.timezone("US/Eastern"))
if now_et.weekday() >= 5 or not (9 <= now_et.hour < 16):
    print("⏳ Market is closed. Skipping update.")
    exit()

# --- Load existing data ---
existing_path = "stocks.csv"
if not os.path.exists(existing_path):
    raise FileNotFoundError("stocks.csv not found.")

existing_df = pd.read_csv(existing_path, parse_dates=["date"])

# --- Build symbol → sector map ---
symbol_sector_map = (
    existing_df[["symbol", "sector"]]
    .drop_duplicates()
    .sort_values("symbol")
    .reset_index(drop=True)
)

# --- Date range: today + yesterday only ---
today = datetime.now().date()
yesterday = today - timedelta(days=1)
start_date = yesterday.strftime("%Y-%m-%d")
end_date = today.strftime("%Y-%m-%d")

# --- Fetch from Polygon API ---
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

# --- Fetch and collect data ---
all_frames = []
for i, row in symbol_sector_map.iterrows():
    symbol, sector = row["symbol"], row["sector"]
    print(f"📡 Fetching {symbol} ({i+1}/{len(symbol_sector_map)})...")
    df_new = fetch_polygon_daily(symbol, start_date, end_date)
    if not df_new.empty:
        df_new["sector"] = sector
        all_frames.append(df_new)
    time.sleep(0.3)  # Light delay to be respectful

# --- Update local file ---
if all_frames:
    new_data = pd.concat(all_frames, ignore_index=True)

    # Determine the dates in the new data (usually today and yesterday)
    overwrite_dates = new_data["date"].dt.normalize().unique()

    # Drop old records from those dates
    existing_cleaned = existing_df[~existing_df["date"].dt.normalize().isin(overwrite_dates)]

    # Combine new + old (excluding overwritten), then sort
    combined = pd.concat([existing_cleaned, new_data], ignore_index=True)
    combined = combined.sort_values(["symbol", "date"])

    # Save final output
    combined.to_csv("stocks.csv", index=False)
    print(f"\n✅ stocks.csv updated with {len(new_data)} new rows after overwriting {len(overwrite_dates)} date(s).")

    # --- Optional GitHub Auto Push ---
    try:
        subprocess.run(["git", "config", "user.name", "Auto Bot"], check=True)
        subprocess.run(["git", "config", "user.email", "bot@example.com"], check=True)
        subprocess.run(["git", "add", "stocks.csv"], check=True)
        subprocess.run(["git", "commit", "-m", f"🔄 Auto-update stocks.csv @ {datetime.now()}"], check=True)
        subprocess.run(["git", "push"], check=True)
        print("🚀 Pushed update to GitHub.")
    except Exception as e:
        print("⚠️ Git push failed:", e)
else:
    print("\n⚠️ No new data fetched. File not changed.")
