import pandas as pd
import numpy as np

def run_strategy(df, k_days_rising=3, eps=1e-6):
    trades = []
    for sym, g in df.groupby("symbol"):
        df_sym = g.sort_values("date").copy().reset_index(drop=True)

        for w in [10, 20, 50, 200]:
            df_sym[f"sma_{w}"] = df_sym["close"].rolling(w, min_periods=w).mean()

        for w in [10, 20, 50]:
            inc = df_sym[f"sma_{w}"].diff() > eps
            df_sym[f"sma_{w}_up"] = inc.rolling(k_days_rising, min_periods=k_days_rising).apply(lambda x: x.all(), raw=False).astype(bool)

        df_sym["above_smas"] = df_sym["close"] > df_sym[[f"sma_{w}" for w in [10, 20, 50, 200]]].max(axis=1) + eps
        df_sym["sma_up_all"] = df_sym[[f"sma_{w}_up" for w in [10, 20, 50]]].all(axis=1)
        df_sym["ready"] = ~df_sym[[f"sma_{w}" for w in [10, 20, 50, 200]]].isna().any(axis=1)

        i = 0
        while i < len(df_sym):
            row = df_sym.iloc[i]
            if row["ready"] and row["above_smas"] and row["sma_up_all"]:
                entry_date = row["date"]
                entry_price = row["close"]
                exit_date, exit_price = None, None

                for j in range(i + 1, len(df_sym)):
                    future = df_sym.iloc[j]
                    below_count = sum(future["close"] < future[f"sma_{w}"] for w in [10, 20, 50, 200])
                    if below_count >= 2:
                        exit_date = future["date"]
                        exit_price = future["close"]
                        break

                trades.append({
                    "symbol": row["symbol"],
                    "entry_date": entry_date,
                    "entry": entry_price,
                    "exit_date": exit_date,
                    "exit_price": exit_price,
                    "exit_reason": "sma_below_2" if exit_date else "force_close",
                    "outcome": (
                        1 if exit_price and exit_price > entry_price
                        else -1 if exit_price and exit_price < entry_price
                        else 0
                    )
                })

                if exit_date is None:
                    break  # stop new trades if last one is still open

                i = j + 1
            else:
                i += 1

    return pd.DataFrame(trades)
