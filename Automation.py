import os
import time
import subprocess
from datetime import datetime, timedelta

import pandas as pd
import pytz
import requests

# --- API key ---
POLYGON_KEY = os.getenv("POLYGON_API_KEY", "")
if not POLYGON_KEY:
    raise EnvironmentError("POLYGON_API_KEY is not set.")

# --- Market-hours guard: 09:30–16:55 ET, Mon–Fri ---
now_utc = datetime.utcnow().replace(tzinfo=pytz.UTC)
et = now_utc.astimezone(pytz.timezone("US/Eastern"))
weekday = et.weekday()               # 0=Mon .. 6=Sun
mins = et.hour * 60 + et.minute      # minutes since midnight ET

market_open = 9 * 60 + 30            # 09:30
market_last = 16 * 60 + 55           # 16:55 (gives your 16:45/16:50 extra pass room)

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

# --- Session with simple retries ---
SESSION = requests.Session()
def fetch_polygon_daily(symbol: str, start: str, end: str, retries: int = 3, backoff: float = 1.0) -> pd.DataFrame:
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_KEY}
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json().get("results", [])
                if not data:
                    return pd.DataFrame()
                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["t"], unit="ms")
                df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})[
                    ["date","open","high","low","close","volume"]
                ]
                df["symbol"] = symbol
                return df
            else:
                print(f"⚠️ {symbol}: HTTP {resp.status_code} (attempt {attempt+1}/{retries})")
        except Exception as e:
            print(f"⚠️ {symbol}: error {e} (attempt {attempt+1}/{retries})")
        time.sleep(backoff * (attempt + 1))
    return pd.DataFrame()

# --- Collect data ---
all_frames = []
for i, row in symbol_sector_map.iterrows():
    symbol = str(row["symbol"]).strip()
    sector = row["sector"]
    if not symbol:
        continue
    print(f"📡 Fetching {symbol}  ({i+1}/{len(symbol_sector_map)})...")
    df_new = fetch_polygon_daily(symbol, start_date, end_date)
    if not df_new.empty:
        df_new["sector"] = sector
        all_frames.append(df_new)
    time.sleep(0.2)  # gentle pacing

# --- Overwrite-first merge (new rows win) ---
if not all_frames:
    print("\n⚠️ No new data fetched. File not changed.")
    raise SystemExit(0)

new_data = pd.concat(all_frames, ignore_index=True)
print(f"🟩 New rows fetched: {len(new_data)}")

# Normalize to date (drop intraday time) to match on (symbol,date)
new_data["date"] = new_data["date"].dt.normalize()
existing_df["date"] = existing_df["date"].dt.normalize()

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
        subprocess.run(["git", "commit", "-m", f"🔄 Auto-update stocks.csv @ {datetime.now()}"], check=True)
        subprocess.run(["git", "push"], check=True)
        print("🚀 Pushed update to GitHub.")
    else:
        print("ℹ️ No changes detected after merge; nothing to commit.")
except Exception as e:
    print("⚠️ Git push failed:", e)
