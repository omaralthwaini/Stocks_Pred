import pandas as pd
import numpy as np

def run_strategy(df):
    all_trades = []
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True)
        g["sma_10"] = g["close"].rolling(10).mean()
        g["sma_20"] = g["close"].rolling(20).mean()
        g["sma_50"] = g["close"].rolling(50).mean()
        g["sma_200"] = g["close"].rolling(200).mean()
        g["stop_loss"] = g["low"].shift(1)

        i = 0
        while i < len(g):
            row = g.iloc[i]
            if (
                row["close"] > g.loc[i, ["sma_10", "sma_20", "sma_50", "sma_200"]].max()
                and g.loc[i, ["sma_10", "sma_20", "sma_50"]].diff().iloc[-1] > 0
            ):
                entry = row["close"]
                entry_date = row["date"]
                for j in range(i+1, len(g)):
                    below = sum(g.loc[j, "close"] < g.loc[j, [f"sma_{w}" for w in (10,20,50,200)]])
                    if below >= 2:
                        trades.append({
                            "symbol": sym, "entry_date": entry_date, "entry": entry,
                            "exit_date": g.loc[j, "date"], "exit_price": g.loc[j, "close"]
                        })
                        i = j
                        break
                else:
                    # fallback to latest price if no exit found
                    trades.append({
                        "symbol": sym, "entry_date": entry_date, "entry": entry,
                        "exit_date": g["date"].iloc[-1], "exit_price": g["close"].iloc[-1]
                    })
                    i = len(g)
            else:
                i += 1

    return pd.DataFrame(trades)
