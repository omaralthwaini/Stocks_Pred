import requests
import pandas as pd
from datetime import datetime, timezone

# ============================================================
# SIMPLE COINGECKO TEST
# ============================================================

COIN_ID = "bitcoin"

url = f"https://api.coingecko.com/api/v3/coins/{COIN_ID}/ohlc"

params = {
    "vs_currency": "usd",
    "days": "1"
}

print("Testing CoinGecko...")
print(f"URL: {url}")

response = requests.get(url, params=params)

print("\nHTTP Status:", response.status_code)

if response.status_code != 200:
    print("ERROR:")
    print(response.text)
    exit()

data = response.json()

print("\nRaw response:")
print(data[:5])

# ============================================================
# Convert to your stocks.csv structure
# ============================================================

rows = []

for item in data:
    timestamp, open_price, high, low, close = item

    date = datetime.fromtimestamp(
        timestamp / 1000,
        tz=timezone.utc
    ).strftime("%Y-%m-%d")

    rows.append({
        "date": date,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": None,
        "symbol": "BTC",
        "asset_type": "crypto",
        "sector": "Crypto"
    })

df = pd.DataFrame(rows)

print("\nFormatted data:")
print(df)

print("\nColumns:")
print(df.columns.tolist())