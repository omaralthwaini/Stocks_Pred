import pandas as pd
import numpy as np

def _rising_flags_apply(df_sym, windows=(10, 20, 50), k_days_rising=3, eps=1e-6):
    """
    Build 'sma_{w}_up' flags using rolling(...).apply(lambda x: x.all()) as requested.
    """
    for w in windows:
        inc = df_sym[f"sma_{w}"].diff() > eps
        df_sym[f"sma_{w}_up"] = (
            inc.rolling(k_days_rising, min_periods=k_days_rising)
               .apply(lambda x: bool(x.all()), raw=False)
               .astype(bool)
        )
    return df_sym

def _compute_smas(df_sym):
    for w in [10, 20, 50, 200]:
        df_sym[f"sma_{w}"] = df_sym["close"].rolling(w, min_periods=w).mean()
    return df_sym

def _stop_for_entry(df_sym, i):
    """Entry stop = prior day's low of the entry candle."""
    if i > 0 and pd.notna(df_sym.loc[i-1, "low"]):
        return float(df_sym.loc[i-1, "low"])
    return None

def _simulate_trades(df_sym, i_start, entry_filter_fn, exit_stop_next_open=True, eps=1e-6):
    """
    Walk candles and create a single trade whenever entry_filter_fn(i) is True.
    Exit rules:
      1) If day's low < stop => exit next day's open (or same day's close if no next bar)
      2) Else if close is below >=2 of [SMA10,20,50,200] => exit at same day's close
    No overlapping trades per symbol (resume after exit bar).
    """
    trades = []
    i = i_start
    while i < len(df_sym):
        if not entry_filter_fn(i):
            i += 1
            continue

        entry_date  = df_sym.loc[i, "date"]
        entry_price = float(df_sym.loc[i, "close"])
        stop_price  = _stop_for_entry(df_sym, i)

        exit_date   = None
        exit_price  = None
        exit_reason = "force_close"

        for j in range(i + 1, len(df_sym)):
            # 1) Stop breach first (risk fail-safe)
            if stop_price is not None and df_sym.loc[j, "low"] < stop_price:
                # Exit at next day's open if available; else same day's close
                if exit_stop_next_open and (j + 1) < len(df_sym):
                    exit_date  = df_sym.loc[j + 1, "date"]
                    exit_price = float(df_sym.loc[j + 1, "open"])
                    exit_reason = "stop_next_open"
                else:
                    exit_date  = df_sym.loc[j, "date"]
                    exit_price = float(df_sym.loc[j, "close"])
                    exit_reason = "stop_eod"
                break

            # 2) SMA break: close below >=2 SMAs
            price = float(df_sym.loc[j, "close"])
            below_count = sum(price < float(df_sym.loc[j, f"sma_{w}"]) for w in [10, 20, 50, 200])
            if below_count >= 2:
                exit_date   = df_sym.loc[j, "date"]
                exit_price  = price
                exit_reason = "sma_below_2"
                break

        # record the trade
        trades.append({
            "symbol": df_sym.loc[i, "symbol"],
            "entry_date": entry_date,
            "entry": entry_price,
            "exit_date": exit_date,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "outcome": (
                1 if pd.notna(exit_price) and exit_price > entry_price
                else -1 if pd.notna(exit_price) and exit_price < entry_price
                else 0
            )
        })

        if exit_date is None:
            break  # still open; stop scanning
        i = j + 1  # resume after the exit bar

    return trades

# -------------------------
# Baseline strategy (your current logic + min body + stop-next-open)
# -------------------------
def run_strategy(df, caps, k_days_rising=3, eps=1e-6, body_min=0.003):
    """
    Entry:
      - All SMAs present
      - close > max(SMA10,20,50,200)
      - SMA10/20/50 rising for k days (using rolling.apply(...).all())
      - Green candle AND (close - open)/open >= body_min (default 0.3%)
    Exit:
      - If any day’s low < (prior day low of entry candle) => exit at next day open
      - Else first day close is below >=2 of the SMAs => exit at that close
    """
    trades = []
    for sym, g in df.groupby("symbol"):
        df_sym = g.sort_values("date").reset_index(drop=True).copy()
        _compute_smas(df_sym)
        _rising_flags_apply(df_sym, (10, 20, 50), k_days_rising, eps)

        df_sym["ready"] = ~df_sym[[f"sma_{w}" for w in [10, 20, 50, 200]]].isna().any(axis=1)
        df_sym["above_smas"] = df_sym["close"] > df_sym[[f"sma_{w}" for w in [10, 20, 50, 200]]].max(axis=1) + eps
        df_sym["sma_up_all"] = df_sym[[f"sma_{w}_up" for w in [10, 20, 50]]].all(axis=1)

        def entry_ok(i):
            if not df_sym.loc[i, "ready"]:
                return False
            if not (df_sym.loc[i, "above_smas"] and df_sym.loc[i, "sma_up_all"]):
                return False
            # green + minimum body
            c, o = float(df_sym.loc[i, "close"]), float(df_sym.loc[i, "open"])
            if not (c > o):
                return False
            body = (c - o) / max(o, eps)
            return body >= body_min

        trades.extend(_simulate_trades(df_sym, i_start=0, entry_filter_fn=entry_ok, exit_stop_next_open=True, eps=eps))

    return pd.DataFrame(trades)

# -------------------------
# Stacked SMA Strategy (new page)
# -------------------------
def run_strategy_stacked(df, caps, k_days_rising=3, eps=1e-6, body_min=0.003):
    """
    Entry:
      - All SMAs present
      - SMA10 > SMA20 > SMA50 > SMA200  (stacking)
      - price > SMA10
      - SMA10/20/50 rising for k days (rolling.apply(...).all())
      - Green candle with minimum body (default 0.3%)
    Exit:
      - Same as baseline: stop-next-open, else SMA break (>=2)
    """
    trades = []
    for sym, g in df.groupby("symbol"):
        df_sym = g.sort_values("date").reset_index(drop=True).copy()
        _compute_smas(df_sym)
        _rising_flags_apply(df_sym, (10, 20, 50), k_days_rising, eps)

        df_sym["ready"] = ~df_sym[[f"sma_{w}" for w in [10, 20, 50, 200]]].isna().any(axis=1)
        df_sym["stacked"] = (
            (df_sym["sma_10"] > df_sym["sma_20"]) &
            (df_sym["sma_20"] > df_sym["sma_50"]) &
            (df_sym["sma_50"] > df_sym["sma_200"])
        )
        df_sym["price_above_sma10"] = df_sym["close"] > df_sym["sma_10"] + eps
        df_sym["sma_up_all"] = df_sym[[f"sma_{w}_up" for w in [10, 20, 50]]].all(axis=1)

        def entry_ok(i):
            if not df_sym.loc[i, "ready"]:
                return False
            if not (df_sym.loc[i, "stacked"] and df_sym.loc[i, "price_above_sma10"] and df_sym.loc[i, "sma_up_all"]):
                return False
            # green + minimum body
            c, o = float(df_sym.loc[i, "close"]), float(df_sym.loc[i, "open"])
            if not (c > o):
                return False
            body = (c - o) / max(o, eps)
            return body >= body_min

        trades.extend(_simulate_trades(df_sym, i_start=0, entry_filter_fn=entry_ok, exit_stop_next_open=True, eps=eps))

    return pd.DataFrame(trades)
