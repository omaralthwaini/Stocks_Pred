# strategy.py
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

def run_strategy(
    df,
    caps=None,
    k_days_rising=3,
    eps=1e-6,
    body_min=0.003,
    *,
    # maps from your seed run (can be empty)
    avg_win_map=None,        # {"AAPL": 7.2, ...}  (percent)
    avg_loss_map=None,       # {"AAPL": -3.8, ...} (percent; usually negative)
    enhanced_cutoff="2025-01-01",
    # NEW knobs (defaults == your current behavior)
    guard_buffer_pp=0.0,     # extra buffer around thresholds, in percentage *points*, e.g. 0.75
    guard_confirm_bars=1,    # need this many consecutive closes beyond threshold to exit
    min_hold_bars=0,         # don’t allow any guard to fire before this many bars
    profit_trail_peak_dd=None,  # if set (e.g. 3.0), exit when drawdown from peak close >= this %
):
    """
    Entry (unchanged):
      • SMAs (10/20/50/200) available
      • Close > max(SMA10,SMA20,SMA50,SMA200)
      • SMA10/20/50 rising for k_days_rising
      • Green candle with body >= body_min

    Base exits (unchanged):
      • Stop: if a future LOW < prior-day LOW at entry -> next day's OPEN (or same day's CLOSE if last row)
      • SMA breakdown: close below >= 2 of (10/20/50/200) -> exit at CLOSE

    Enhanced guards (only for entries on/after enhanced_cutoff, using maps from all history):
      • Loss guard: exit when return <= avg_loss(symbol) - buffer, with confirmation bars
      • Profit guard A (default): once return has been >= avg_win(symbol), exit when it
        closes back < avg_win(symbol) - buffer, confirmed for N bars
      • Profit guard B (optional): trailing peak giveback — exit when drawdown from max
        close since entry >= profit_trail_peak_dd (in %). If provided, this replaces Profit guard A.
    """
    trades = []

    avg_win_map  = avg_win_map or {}
    avg_loss_map = avg_loss_map or {}
    cutoff_ts = pd.Timestamp(enhanced_cutoff) if enhanced_cutoff else None
    thr_eps = 1e-9

    for sym, g in df.groupby("symbol"):
        df_sym = g.sort_values("date").reset_index(drop=True).copy()

        _compute_smas(df_sym)
        _rising_flags_apply(df_sym, (10, 20, 50), k_days_rising, eps)

        sma_cols = [f"sma_{w}" for w in [10, 20, 50, 200]]
        df_sym["ready"] = ~df_sym[sma_cols].isna().any(axis=1)
        df_sym["above_smas"] = df_sym["close"] > df_sym[sma_cols].max(axis=1) + eps
        df_sym["sma_up_all"] = df_sym[[f"sma_{w}_up" for w in [10, 20, 50]]].all(axis=1)

        i = 0
        while i < len(df_sym):
            if not df_sym.loc[i, "ready"]:
                i += 1
                continue

            if not (df_sym.loc[i, "above_smas"] and df_sym.loc[i, "sma_up_all"]):
                i += 1
                continue

            c, o = float(df_sym.loc[i, "close"]), float(df_sym.loc[i, "open"])
            if not (c > o):
                i += 1
                continue

            body = (c - o) / max(o, eps)
            if body < body_min:
                i += 1
                continue

            entry_date  = pd.to_datetime(df_sym.loc[i, "date"])
            entry_price = c
            stop_price  = _prev_day_low_as_stop(df_sym, i)

            use_guards = cutoff_ts is not None and entry_date >= cutoff_ts
            win_thr  = float(avg_win_map.get(sym))  if sym in avg_win_map  else None
            loss_thr = float(avg_loss_map.get(sym)) if sym in avg_loss_map else None

            crossed_win_once = False
            consec_loss_hit  = 0
            consec_win_recross=0
            peak_close = entry_price  # for optional peak giveback

            exit_date   = None
            exit_price  = None
            exit_reason = "force_close"

            for j in range(i + 1, len(df_sym)):
                price_j = float(df_sym.loc[j, "close"])
                date_j  = pd.to_datetime(df_sym.loc[j, "date"])
                ret_pct = (price_j / entry_price - 1.0) * 100.0
                bars_held = j - i

                # Track peak for optional giveback
                if price_j > peak_close:
                    peak_close = price_j

                # ---------- Enhanced guards ----------
                if use_guards and bars_held >= min_hold_bars:
                    buffer = guard_buffer_pp  # in percentage points

                    # Loss guard with buffer + confirmation
                    if loss_thr is not None:
                        if ret_pct <= (loss_thr - buffer) + thr_eps:
                            consec_loss_hit += 1
                        else:
                            consec_loss_hit = 0
                        if consec_loss_hit >= guard_confirm_bars:
                            exit_date   = date_j
                            exit_price  = price_j
                            exit_reason = "guard_loss_close"
                            break

                    # Profit guard: choose mode
                    if profit_trail_peak_dd is not None:
                        # Peak giveback mode
                        dd_from_peak = (price_j / peak_close - 1.0) * 100.0
                        if dd_from_peak <= -abs(profit_trail_peak_dd) - thr_eps:
                            exit_date   = date_j
                            exit_price  = price_j
                            exit_reason = "guard_peak_giveback"
                            break
                    elif win_thr is not None:
                        # Avg-win re-cross with buffer + confirmation
                        if (not crossed_win_once) and (ret_pct >= (win_thr - buffer) - thr_eps):
                            crossed_win_once = True
                        if crossed_win_once:
                            if ret_pct < (win_thr - buffer) - thr_eps:
                                consec_win_recross += 1
                            else:
                                consec_win_recross = 0
                            if consec_win_recross >= guard_confirm_bars:
                                exit_date   = date_j
                                exit_price  = price_j
                                exit_reason = "guard_win_trail"
                                break

                # ---------- Base stop: prior-day low breach -> next open ----------
                low_j = df_sym.loc[j, "low"]
                if stop_price is not None and pd.notna(low_j) and low_j < stop_price:
                    if (j + 1) < len(df_sym):
                        exit_date   = pd.to_datetime(df_sym.loc[j + 1, "date"])
                        exit_price  = float(df_sym.loc[j + 1, "open"])
                        exit_reason = "stop_next_open"
                    else:
                        exit_date   = date_j
                        exit_price  = price_j
                        exit_reason = "stop_eod"
                    break

                # ---------- Base SMA breakdown ----------
                below_count = 0
                for w in [10, 20, 50, 200]:
                    sma_val = df_sym.loc[j, f"sma_{w}"]
                    if pd.notna(sma_val) and price_j < float(sma_val):
                        below_count += 1
                if below_count >= 2:
                    exit_date   = date_j
                    exit_price  = price_j
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

            if exit_date is None:
                break

            i = j + 1

    return pd.DataFrame(trades)
