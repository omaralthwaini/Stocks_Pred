def build_ml_dataset(combined, trades):
    rows = []
    combined = combined.sort_values(["symbol", "date"])

    for _, trade in trades.iterrows():
        sym = trade["symbol"]
        entry_date = trade["entry_date"]
        exit_price = trade["exit_price"]

        sub = combined[(combined["symbol"] == sym) & (combined["date"] <= entry_date)].copy().tail(8)
        if len(sub) < 8:
            continue

        sub["days_before_entry"] = list(range(-7, 1))
        sub["target"] = exit_price
        sub["entry"] = trade["entry"]
        sub["exit_date"] = trade["exit_date"]
        sub["trade_id"] = f"{sym}_{entry_date.strftime('%Y%m%d')}"
        sub["outcome"] = np.sign(exit_price - trade["entry"])
        sub["pct_return"] = (exit_price / trade["entry"] - 1.0) * 100
        rows.append(sub)

    return pd.concat(rows, ignore_index=True)
