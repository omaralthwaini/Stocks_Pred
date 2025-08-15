import pandas as pd
import numpy as np

def run_strategy(df, k_days_rising=3, eps=1e-6):
    trades = []

    for sym, group in df.groupby("symbol"):
        df_sym = group.sort_values("date").reset_index(drop=True).copy()

        # --- Compute SMAs ---
        for w in [10, 20, 50, 200]:
            df_sym[f"sma_{w}"] = df_sym["close"].rolling(w, min_periods=w).mean()

        # --- Rising flags for 10/20/50 using k-day window ---
        for w in [10, 20, 50]:
            col = f"sma_{w}"
            inc = df_sym[col].diff() > eps
            df_sym[f"sma_{w}_up"] = inc.rolling(k_days_rising, min_periods=k_days_rising).apply(lambda x: x.all(), raw=False).astype(bool)

        # --- Entry & Exit Conditions ---
        df_sym["above_smas"] = df_sym["close"] > df_sym[[f"sma_{w}" for w in [10, 20, 50, 200]]].max(axis=1) + eps
        df_sym["sma_up_all"] = df_sym[[f"sma_{w}_up" for w in [10, 20, 50]]].all(axis=1)
        df_sym["ready"] = ~df_sym[[f"sma_{w}" for w in [10, 20, 50, 200]]].isna().any(axis=1)

        i = 0
        while i < len(df_sym):
            row = df_sym.iloc[i]
            if row["ready"] and row["above_smas"] and row["sma_up_all"]:
                entry_date = row["date"]
                entry_price = row["close"]
                symbol = row["symbol"]

                exit_date = None
                exit_price = None
                exit_reason = "force_close"

                # --- Scan forward for exit ---
                for j in range(i + 1, len(df_sym)):
                    next_row = df_sym.iloc[j]
                    below_count = sum(next_row["close"] < next_row[f"sma_{w}"] for w in [10, 20, 50, 200])
                    if below_count >= 2:
                        exit_date = next_row["date"]
                        exit_price = next_row["close"]
                        exit_reason = "sma_below_2"
                        break

                trades.append({
                    "symbol": symbol,
                    "entry_date": entry_date,
                    "entry": entry_price,
                    "exit_date": exit_date,
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "outcome": (
                        1 if exit_price and exit_price > entry_price
                        else -1 if exit_price and exit_price < entry_price
                        else 0
                    )
                })

                if exit_date is None:
                    break  # Stop if trade is still open

                i = j + 1  # Move to bar after exit
            else:
                i += 1

    return pd.DataFrame(trades)
