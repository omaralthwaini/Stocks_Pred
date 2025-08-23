import pandas as pd
import numpy as np

def _compute_smas(df_sym):
    for w in [10, 20, 50, 200]:
        df_sym[f"sma_{w}"] = df_sym["close"].rolling(w, min_periods=w).mean()
    return df_sym

def _rising_flags_apply(df_sym, windows=(10, 20, 50), k_days_rising=3, eps=1e-6):
    for w in windows:
        inc = df_sym[f"sma_{w}"].diff() > eps
        df_sym[f"sma_{w}_up"] = (
            inc.rolling(k_days_rising, min_periods=k_days_rising)
               .apply(lambda x: bool(x.all()), raw=False)
               .astype(bool)
        )
    return df_sym

def _prev_day_low_as_stop(df_sym, i):
    if i > 0 and pd.notna(df_sym.loc[i-1, "low"]):
        return float(df_sym.loc[i-1, "low"])
    return None

def run_strategy(df, caps=None, k_days_rising=3, eps=1e-6, body_min=0.003):
    """
    Entry:
      • All SMAs (10/20/50/200) available
      • Close above the max of those SMAs
      • SMA10/20/50 each rising for k_days_rising
      • Green candle with minimum body (close-open)/open >= body_min (default 0.3%)

    Exit:
      • Bar-level stop: if a future bar's LOW < prior-day LOW at entry -> exit next day's OPEN (or same day's CLOSE if last row)
      • Or if a future CLOSE is below >= 2 of the 10/20/50/200 SMAs -> exit at that CLOSE
    """
    trades = []

    # Ensure df is sorted per symbol
    for sym, g in df.groupby("symbol"):
        df_sym = g.sort_values("date").reset_index(drop=True).copy()

        # SMAs + rising flags
        _compute_smas(df_sym)
        _rising_flags_apply(df_sym, (10, 20, 50), k_days_rising, eps)

        # Entry readiness
        sma_cols = [f"sma_{w}" for w in [10, 20, 50, 200]]
        df_sym["ready"] = ~df_sym[sma_cols].isna().any(axis=1)
        df_sym["above_smas"] = df_sym["close"] > df_sym[sma_cols].max(axis=1) + eps
        df_sym["sma_up_all"] = df_sym[[f"sma_{w}_up" for w in [10, 20, 50]]].all(axis=1)

        i = 0
        while i < len(df_sym):
            if not df_sym.loc[i, "ready"]:
                i += 1
                continue

            # Entry conditions (trend + price filter)
            if not (df_sym.loc[i, "above_smas"] and df_sym.loc[i, "sma_up_all"]):
                i += 1
                continue

            c, o = float(df_sym.loc[i, "close"]), float(df_sym.loc[i, "open"])
            if not (c > o):  # must be green
                i += 1
                continue

            body = (c - o) / max(o, eps)
            if body < body_min:
                i += 1
                continue

            # Enter
            entry_date  = df_sym.loc[i, "date"]
            entry_price = c
            stop_price  = _prev_day_low_as_stop(df_sym, i)

            exit_date   = None
            exit_price  = None
            exit_reason = "force_close"

            # Manage forward
            for j in range(i + 1, len(df_sym)):
                # Stop logic
                low_j = df_sym.loc[j, "low"]
                if stop_price is not None and pd.notna(low_j) and low_j < stop_price:
                    if (j + 1) < len(df_sym):
                        exit_date   = df_sym.loc[j + 1, "date"]
                        exit_price  = float(df_sym.loc[j + 1, "open"])
                        exit_reason = "stop_next_open"
                    else:
                        exit_date   = df_sym.loc[j, "date"]
                        exit_price  = float(df_sym.loc[j, "close"])
                        exit_reason = "stop_eod"
                    break

                # SMA breakdown: close below >= 2 SMAs -> exit at close
                price = float(df_sym.loc[j, "close"])
                below_count = 0
                for w in [10, 20, 50, 200]:
                    sma_val = df_sym.loc[j, f"sma_{w}"]
                    if pd.notna(sma_val) and price < float(sma_val):
                        below_count += 1
                if below_count >= 2:
                    exit_date   = df_sym.loc[j, "date"]
                    exit_price  = price
                    exit_reason = "sma_below_2"
                    break

            trades.append({
                "symbol": sym,
                "entry_date": entry_date,
                "entry": entry_price,
                "exit_date": exit_date,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "outcome": (
                    1 if pd.notna(exit_price) and exit_price > entry_price
                    else -1 if pd.notna(exit_price) and exit_price < entry_price
                    else 0
                ),
            })

            # If still open, stop scanning this symbol
            if exit_date is None:
                break

            # Jump past the exit bar
            i = j + 1

    return pd.DataFrame(trades)
