# strategy.py
import pandas as pd
import numpy as np

def _compute_smas(df_sym):
    for w in [10, 20, 50, 200]:
        df_sym[f"sma_{w}"] = df_sym["close"].rolling(w, min_periods=w).mean()
    return df_sym

def _compute_rising_flags(df_sym, windows=(10, 20, 50, 200), eps=1e-12):
    """
    Rising = tiny positive slope today vs yesterday (no multi-day streak needed).
    """
    for w in windows:
        col = f"sma_{w}"
        df_sym[f"{col}_rising"] = df_sym[col].diff() > eps
    return df_sym

def _entry_stop_price(df_sym, i):
    """
    Default: yesterday's low.
    If yesterday's low >= today's close, use min(low) of last 3 bars (including today).
    """
    today_close = float(df_sym.loc[i, "close"])
    prev_low = float(df_sym.loc[i-1, "low"]) if i > 0 and pd.notna(df_sym.loc[i-1, "low"]) else None

    if prev_low is not None and prev_low < today_close:
        return prev_low

    # Fallback: min low of the last 3 bars (i-2 .. i), bounded to start of data
    start = max(0, i - 2)
    last3_low = df_sym.loc[start:i, "low"].min()
    return float(last3_low) if pd.notna(last3_low) else None

def run_strategy(
    df,
    caps=None,
    eps=1e-12,
    *,
    # maps from your seed run (can be empty)
    avg_win_map=None,        # {"AAPL": 7.2, ...}  (percent)
    avg_loss_map=None,       # {"AAPL": -3.8, ...} (percent; usually negative)
    enhanced_cutoff="2025-01-01",
    # Guard knobs (unchanged)
    guard_buffer_pp=0.0,     # extra buffer around thresholds, in percentage points
    guard_confirm_bars=1,    # need this many consecutive closes beyond threshold to exit
    min_hold_bars=3,         # NEW default: hold at least 3 bars before ANY exit can fire
    profit_trail_peak_dd=None,  # if set (e.g. 3.0), exit when drawdown from peak close >= this %
):
    """
    Entry (updated):
      • SMAs (10/20/50/200) computed.
      • Today's close is above at least 3 SMAs that are RISING today (diff > 0).
      • Today is a GREEN candle (close > open).
      • Today's volume >= rolling median(21) of volume.

    Base exits (unchanged mechanics, but ALL are blocked until min_hold_bars elapse):
      • Stop: if a future LOW < entry stop (see rule above) -> next day's OPEN (or same day's CLOSE if last row)
      • SMA breakdown: close below >= 2 of (10/20/50/200) -> exit at CLOSE

    Enhanced guards (only for entries on/after enhanced_cutoff):
      • Loss guard vs avg_loss_map (with buffer + confirmation)
      • Profit guard A: avg-win re-cross (with buffer + confirmation), OR
      • Profit guard B: trailing peak giveback (% drawdown from highest close since entry)
    """
    trades = []

    avg_win_map  = avg_win_map or {}
    avg_loss_map = avg_loss_map or {}
    cutoff_ts = pd.Timestamp(enhanced_cutoff) if enhanced_cutoff else None
    tiny = eps

    for sym, g in df.groupby("symbol"):
        df_sym = g.sort_values("date").reset_index(drop=True).copy()

        # Preconditions
        _compute_smas(df_sym)
        _compute_rising_flags(df_sym, (10, 20, 50, 200), eps=tiny)

        # Volume median(21). If no 'volume' column, treat condition as passed.
        if "volume" in df_sym.columns:
            df_sym["vol_med21"] = df_sym["volume"].rolling(21, min_periods=21).median()

        i = 0
        while i < len(df_sym):
            # Need at least 3 SMAs available today AND yesterday (for rising check)
            avail = []
            for w in (10, 20, 50, 200):
                cur = df_sym.loc[i, f"sma_{w}"]
                prev = df_sym.loc[i-1, f"sma_{w}"] if i > 0 else np.nan
                if pd.notna(cur) and pd.notna(prev):
                    avail.append(w)
            if len(avail) < 3:
                i += 1
                continue

            # Candle must be green
            c = float(df_sym.loc[i, "close"])
            o = float(df_sym.loc[i, "open"])
            if not (c > o + tiny):
                i += 1
                continue

            # Volume filter (if available)
            if "volume" in df_sym.columns:
                vol = df_sym.loc[i, "volume"]
                med = df_sym.loc[i, "vol_med21"]
                if not (pd.notna(vol) and pd.notna(med) and float(vol) >= float(med)):
                    i += 1
                    continue

            # Price above at least 3 rising SMAs today
            above_rising_count = 0
            for w in (10, 20, 50, 200):
                sma_val = df_sym.loc[i, f"sma_{w}"]
                rising  = bool(df_sym.loc[i, f"sma_{w}_rising"]) if pd.notna(df_sym.loc[i, f"sma_{w}_rising"]) else False
                if pd.notna(sma_val) and rising and (c > float(sma_val) + tiny):
                    above_rising_count += 1
            if above_rising_count < 3:
                i += 1
                continue

            # ---- ENTRY ACCEPTED ----
            entry_date  = pd.to_datetime(df_sym.loc[i, "date"])
            entry_price = c
            stop_price  = _entry_stop_price(df_sym, i)

            use_guards = (cutoff_ts is not None) and (entry_date >= cutoff_ts)
            win_thr  = float(avg_win_map.get(sym))  if sym in avg_win_map  else None
            loss_thr = float(avg_loss_map.get(sym)) if sym in avg_loss_map else None

            crossed_win_once   = False
            consec_loss_hit    = 0
            consec_win_recross = 0
            peak_close = entry_price

            exit_date   = None
            exit_price  = None
            exit_reason = "force_close"

            for j in range(i + 1, len(df_sym)):
                price_j = float(df_sym.loc[j, "close"])
                date_j  = pd.to_datetime(df_sym.loc[j, "date"])
                ret_pct = (price_j / entry_price - 1.0) * 100.0
                bars_held = j - i

                # Track peak
                if price_j > peak_close:
                    peak_close = price_j

                # --- Respect global min-hold: block ALL exits before this many bars ---
                if bars_held < int(min_hold_bars):
                    continue

                # ---------- Enhanced guards ----------
                if use_guards:
                    buffer = float(guard_buffer_pp)

                    # Loss guard with buffer + confirmation
                    if loss_thr is not None:
                        if ret_pct <= (loss_thr - buffer) + tiny:
                            consec_loss_hit += 1
                        else:
                            consec_loss_hit = 0
                        if consec_loss_hit >= int(guard_confirm_bars):
                            exit_date   = date_j
                            exit_price  = price_j
                            exit_reason = "guard_loss_close"
                            break

                    # Profit guard: trail peak giveback OR avg-win re-cross
                    if profit_trail_peak_dd is not None:
                        dd_from_peak = (price_j / peak_close - 1.0) * 100.0
                        if dd_from_peak <= -abs(float(profit_trail_peak_dd)) - tiny:
                            exit_date   = date_j
                            exit_price  = price_j
                            exit_reason = "guard_peak_giveback"
                            break
                    elif win_thr is not None:
                        if (not crossed_win_once) and (ret_pct >= (win_thr - buffer) - tiny):
                            crossed_win_once = True
                        if crossed_win_once:
                            if ret_pct < (win_thr - buffer) - tiny:
                                consec_win_recross += 1
                            else:
                                consec_win_recross = 0
                            if consec_win_recross >= int(guard_confirm_bars):
                                exit_date   = date_j
                                exit_price  = price_j
                                exit_reason = "guard_win_trail"
                                break

                # ---------- Base stop: entry stop breach -> next open ----------
                low_j = df_sym.loc[j, "low"]
                if stop_price is not None and pd.notna(low_j) and float(low_j) < float(stop_price) - tiny:
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
                for w in (10, 20, 50, 200):
                    sma_val = df_sym.loc[j, f"sma_{w}"]
                    if pd.notna(sma_val) and price_j < float(sma_val) - tiny:
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
                break  # still open; move to next symbol
            i = j + 1

    return pd.DataFrame(trades)
