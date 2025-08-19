import pandas as pd
import requests
import os
import time
from datetime import datetime, timedelta
import pytz
import subprocess

# --- Get API key ---
POLYGON_KEY = os.getenv("POLYGON_API_KEY", "")
if not POLYGON_KEY:
    raise EnvironmentError("POLYGON_API_KEY is not set.")

# --- Market hours check: allow 09:30–16:45 ET, Mon–Fri ---
now_utc = datetime.utcnow().replace(tzinfo=pytz.UTC)
et = now_utc.astimezone(pytz.timezone("US/Eastern"))
weekday = et.weekday()          # 0=Mon, 6=Sun
mins = et.hour * 60 + et.minute # minutes since midnight ET

market_open = 9 * 60 + 30       # 09:30 = 570
market_last = 16 * 60 + 45      # 16:45 = 1005

print(f"🕒 ET now: {et.strftime('%Y-%m-%d %H:%M')}  (weekday={weekday}, mins={mins})")
if weekday >= 5 or not (market_open <= mins <= market_last):
    print("⏳ Market window closed for updater. Skipping.")
    raise SystemExit(0)  # exit cleanly so downstream steps can decide what to do

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
print(f"🧾 Symbols to update: {len(symbol_sector_map)}")

# --- Date range: yesterday + today ---
today = datetime.now().date()
yesterday = today - timedelta(days=1)
start_date = yesterday.strftime("%Y-%m-%d")
end_date = today.strftime("%Y-%m-%d")
print(f"📅 Fetch window: {start_date} → {end_date}")

# --- Fetch from Polygon API ---
def fetch_polygon_daily(symbol, start, end):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": POLYGON_KEY
    }
    resp = requests.get(url, params=params, timeout=30)
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

# --- Collect data ---
all_frames = []
for i, row in symbol_sector_map.iterrows():
    symbol, sector = str(row["symbol"]).strip(), row["sector"]
    if not symbol:
        continue
    print(f"📡 Fetching {symbol}  ({i+1}/{len(symbol_sector_map)})...")
    df_new = fetch_polygon_daily(symbol, start_date, end_date)
    if not df_new.empty:
        df_new["sector"] = sector
        all_frames.append(df_new)
    time.sleep(0.25)  # gentle pacing

# --- Overwrite-first merge (new rows win) ---
if not all_frames:
    print("\n⚠️ No new data fetched. File not changed.")
    raise SystemExit(0)

new_data = pd.concat(all_frames, ignore_index=True)
print(f"🟩 New rows fetched: {len(new_data)}")

# Normalize to date (no time) so (symbol,date) matches are exact
new_data["date"] = new_data["date"].dt.normalize()
existing_df["date"] = existing_df["date"].dt.normalize()

# Anti-join: drop old rows that collide with the new (symbol,date) keys
keys_df = new_data[["symbol", "date"]].drop_duplicates()
before_rows = len(existing_df)
existing_filtered = existing_df.merge(keys_df, on=["symbol","date"], how="left", indicator=True)
existing_filtered = existing_filtered[existing_filtered["_merge"] == "left_only"].drop(columns="_merge")
dropped = before_rows - len(existing_filtered)
print(f"🗑️ Overwritten existing rows: {dropped}")

# Combine (new data first so it “wins”), dedupe for safety, sort
combined = pd.concat([new_data, existing_filtered], ignore_index=True)
combined = combined.drop_duplicates(subset=["symbol","date"], keep="first")
combined = combined.sort_values(["symbol", "date"])
print(f"📊 Final total rows: {len(combined)}")

# Save updated file
combined.to_csv("stocks.csv", index=False)
print("✅ stocks.csv written.")

# --- Push to GitHub ---
try:
    subprocess.run(["git", "config", "user.name", "Auto Bot"], check=True)
    subprocess.run(["git", "config", "user.email", "bot@example.com"], check=True)
    subprocess.run(["git", "add", "stocks.csv"], check=True)

    # Only commit if there is a diff
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if diff.returncode != 0:
        subprocess.run(["git", "commit", "-m", f"🔄 Auto-update stocks.csv @ {datetime.now()}"], check=True)
        subprocess.run(["git", "push"], check=True)
        print("🚀 Pushed update to GitHub.")
    else:
        print("ℹ️ No changes detected after merge; nothing to commit.")
except Exception as e:
    print("⚠️ Git push failed:", e)
