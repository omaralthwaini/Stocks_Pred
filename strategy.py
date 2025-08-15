import pandas as pd
import numpy as np

def run_strategy(df):
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Compute SMAs
    for w in [10, 20, 50, 200]:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()

    # SMA rising flags
    for w in [10, 20, 50]:
        col = f"sma_{w}"
        df[f"sma_{w}_up"] = df[col] > df[col].shift(1)

    df["above_smas"] = df["close"] > df[[f"sma_{w}" for w in [10, 20, 50, 200]]].max(axis=1)
    df["sma_up_all"] = df[[f"sma_{w}_up" for w in [10, 20, 50]]].all(axis=1)

    # ✅ Make sure trades list is initialized before the loop
    trades = []

    i = 0
    while i < len(df):
        row = df.iloc[i]
        if row["sma_up_all"] and row["above_smas"]:
            entry_date = row["date"]
            entry_price = row["close"]
            symbol = row["symbol"]

            exit_date, exit_price, j_exit = None, None, None

            for j in range(i + 1, len(df)):
                next_row = df.iloc[j]
                below = sum(next_row["close"] < next_row[f"sma_{w}"] for w in [10, 20, 50, 200])
                if below >= 2:
                    exit_date = next_row["date"]
                    exit_price = next_row["close"]
                    j_exit = j
                    break

            trades.append({
                "symbol": symbol,
                "entry_date": entry_date,
                "entry": entry_price,
                "exit_date": exit_date,
                "exit_price": exit_price,
                "outcome": (
                    1 if exit_price and exit_price > entry_price
                    else -1 if exit_price and exit_price < entry_price
                    else 0
                )
            })

            i = j_exit + 1 if j_exit is not None else i + 1
        else:
            i += 1

    # ✅ Even if trades is still empty, this is valid
    return pd.DataFrame(trades)
