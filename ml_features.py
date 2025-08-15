import pandas as pd
import numpy as np

def build_ml_dataset(combined, trades):
    rows = []
    combined = combined.sort_values(["symbol", "date"])

    for _, trade in trades.iterrows():
        sym = trade["symbol"]
        entry_date = trade["entry_date"]
        exit_price = trade.get("exit_price", np.nan)  # ← make sure this is present

        sub = combined[(combined["symbol"] == sym) & (combined["date"] <= entry_date)].copy().tail(8)
        if len(sub) < 8:
            continue

        sub["days_before_entry"] = list(range(-7, 1))
        sub["target"] = exit_price
        sub["entry"] = trade["entry"]
        sub["exit_price"] = exit_price              # ✅ <-- this line is required
        sub["exit_date"] = trade.get("exit_date", pd.NaT)
        sub["trade_id"] = f"{sym}_{entry_date.strftime('%Y%m%d')}"
        sub["outcome"] = np.sign(exit_price - trade["entry"]) if not pd.isna(exit_price) else 0
        sub["pct_return"] = ((exit_price / trade["entry"]) - 1.0) * 100 if not pd.isna(exit_price) else 0

        rows.append(sub)

    return pd.concat(rows, ignore_index=True)
