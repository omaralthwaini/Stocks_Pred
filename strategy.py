# strategy.py
import pandas as pd
import numpy as np

def _compute_smas(df_sym):
    for w in [10, 20, 50, 200]:
        df_sym[f"sma_{w}"] = df_sym["close"].rolling(w, min_periods=w).mean()
    return df_sym

def _rising_flags_apply(df_sym, windows=(10, 20, 50, 200), k_days_rising=3, eps=1e-6):
    """
    'Rising' means each of the last k days the SMA increased (strictly > eps).
    Produces boolean columns: sma_{w}_up
    """
    for w in windows:
        inc = df_sym[f"sma_{w}"].diff() > eps
        df_sym[f"sma_{w}_up"] = (
            inc.rolling(k_days_rising, min_periods=k_days_rising)
               .apply(lambda x: bool(np.all(x)), raw=False)
               .astype(bool)
        )
    return df_sym

def _initial_stop_from_lows(df_sym, i, entry_price):
    """
    Initial stop logic:
      1) yday low as baseline (if exists)
      2) if yday low >= entry close -> fallback to min(low) of last-3 bars
      3) if still >= entry, try the lowest low < entry among those last-3; else None
    """
    ylow = float(df_sym.loc[i-1, "low"]) if i > 0 and pd.notna(df_sym.loc[i-1, "low"]) else None

    # last-3 bars before entry (i-3, i-2, i-1) if available
    if i > 0:
        start = max(0, i-3)
        last3 = df_sym.loc[start:i-1, "low"].dropna()
        last3_min = float(last3.min()) if not last3.empty else None
    else:
        last3 = pd.Series(dtype="float64")
        last3_min = None

    stop = None
    if ylow is not None:
        stop = ylow
        if stop >= entry_price and last3_min is not None:
            stop = last3_min
    elif last3_min is not None:
        stop = last3_min

    # Final guard: never return a stop at/above entry; try pick a sub-entry low if it exists
    if stop is not None and stop >= entry_price and not last3.empty:
        sub_entry = last3[last3 < entry_price]
        if not sub_entry.empty:
            stop = float(sub_entry.min())
        else:
            # give up: no sensible technical stop below entry
            return None

    return float(stop) if stop is not None else None

def run_strategy(
    df,
    caps=None,
    k_days_rising=3,
    eps=1e-6,
    body_min=0.003,              # ≥0.3% body
    *,
    # maps from your seed run (can be empty)
    avg_win_map=None,            # {"AAPL": 7.2, ...}  (percent)
    avg_loss_map=None,           # {"AAPL": -3.8, ...} (percent; usually negative)
    enhanced_cutoff="2025-01-01",
    # Enhanced guard knobs
    guard_buffer_pp=0.0,         # buffer around thresholds (percentage points)
    guard_confirm_bars=1,        # consecutive closes beyond threshold to exit
    min_hold_bars=0,             # guards & SMA breakdown blocked before this many bars (STOP still active)
    profit_trail_peak_dd=None,   # trailing peak giveback % (if set, replaces avg-win re-cross)
    # Entry quality knobs (constants you can expose later if you want)
    vol_mul_vs_med21=1.2,        # today volume >= 1.2× median(21)
    close_top_frac=0.60,         # close in top 40% of day range  (>=0.60)
    max_extend_vs_sma10=0.05,    # |close - SMA10| / SMA10 <= 5%
):
    """
    ENTRY (tightened):
      • Have SMA10/20/50 available (SMA200 optional).
      • At least THREE rising SMAs among {10,20,50,200}, where 'rising' = each of last k_days_rising days SMA up.
      • Close above those rising SMAs it counts.
      • Candle is green AND body >= body_min (default 0.3%).
      • Close in the top 40% of the intraday range.
      • Volume >= 1.2× rolling median(21).
      • Not extended: |close - SMA10| / SMA10 ≤ 5% (if SMA10 available).

    BASE EXITS:
      • STOP: if a future LOW < initial stop -> next day's OPEN (or same day's CLOSE if last row).
        (STOP is allowed to fire immediately; it ignores min_hold_bars.)
      • SMA breakdown: close below ≥ 2 of {10,20,50,200} -> exit at CLOSE
        (now gated by min_hold_bars to avoid super-early breakdowns).

    ENHANCED GUARDS (entries on/after enhanced_cutoff):
      • Loss guard: exit when return <= avg_loss(symbol) - buffer, with confirmation bars.
      • Profit guard A: once return ≥ avg_win(symbol), exit when it re-crosses below
        avg_win(symbol) - buffer, confirmed N bars.
      • Profit guard B (if profit_trail_peak_dd is set): trailing peak giveback replaces A.
    """
    trades = []

    avg_win_map  = avg_win_map or {}
    avg_loss_map = avg_loss_map or {}
    cutoff_ts = pd.Timestamp(enhanced_cutoff) if enhanced_cutoff else None
    thr_eps = 1e-9

    # Precompute per-symbol volume median (21) once
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True).copy()
    if "volume" in df.columns:
        df["vol_med21"] = df.groupby("symbol")["volume"].transform(lambda s: s.rolling(21, min_periods=21).median())
    else:
        df["vol_med21"] = np.nan  # volume filter will be skipped if NaN

    for sym, g in df.groupby("symbol", sort=False):
        df_sym = g.reset_index(drop=True).copy()

        # SMAs + rising flags
        _compute_smas(df_sym)
        _rising_flags_apply(df_sym, (10, 20, 50, 200), k_days_rising, eps)

        # We only *require* 10/20/50 to exist for entry decisions; 200 is optional
        sma_basic = [f"sma_{w}" for w in [10, 20, 50]]
        df_sym["ready"] = ~df_sym[sma_basic].isna().any(axis=1)

        i = 0
        while i < len(df_sym):
            if not df_sym.loc[i, "ready"]:
                i += 1
                continue

            c = float(df_sym.loc[i, "close"])
            o = float(df_sym.loc[i, "open"])
            h = float(df_sym.loc[i, "high"]) if pd.notna(df_sym.loc[i, "high"]) else c
            l = float(df_sym.loc[i, "low"])  if pd.notna(df_sym.loc[i, "low"])  else c

            # --- Entry filters ---

            # 1) Rising SMAs and price above at least THREE of them (among 10/20/50/200)
            rising_and_above = 0
            for w in [10, 20, 50, 200]:
                sma_val = df_sym.loc[i, f"sma_{w}"] if f"sma_{w}" in df_sym.columns else np.nan
                sma_up  = bool(df_sym.loc[i, f"sma_{w}_up"]) if f"sma_{w}_up" in df_sym.columns else False
                if pd.notna(sma_val) and sma_up and c > float(sma_val) + eps:
                    rising_and_above += 1
            if rising_and_above < 3:
                i += 1
                continue

            # 2) Green candle + body >= body_min
            if not (c > o):
                i += 1
                continue
            body = (c - o) / max(o, eps)
            if body < body_min:
                i += 1
                continue

            # 3) Close in top 40% of the day's range
            rng = max(h - l, eps)
            close_pos = (c - l) / rng  # 0..1
            if close_pos < close_top_frac:   # e.g., 0.60
                i += 1
                continue

            # 4) Volume >= 1.2 × median(21)  (skip if volume data not available yet)
            vol_ok = True
            if "volume" in df_sym.columns and pd.notna(df_sym.loc[i, "vol_med21"]):
                v = float(df_sym.loc[i, "volume"])
                vmed = float(df_sym.loc[i, "vol_med21"])
                vol_ok = (v >= vol_mul_vs_med21 * vmed)
            if not vol_ok:
                i += 1
                continue

            # 5) Not extended vs SMA10
            sma10 = df_sym.loc[i, "sma_10"]
            if pd.notna(sma10) and sma10 > 0:
                if abs(c - float(sma10)) / float(sma10) > max_extend_vs_sma10:
                    i += 1
                    continue

            # --- Entry accepted ---
            entry_date  = pd.to_datetime(df_sym.loc[i, "date"])
            entry_price = c

            # Initial stop from lows (yday, fallback last-3)
            stop_price  = _initial_stop_from_lows(df_sym, i, entry_price)

            use_guards = (cutoff_ts is not None) and (entry_date >= cutoff_ts)
            win_thr  = float(avg_win_map.get(sym))  if sym in avg_win_map  else None
            loss_thr = float(avg_loss_map.get(sym)) if sym in avg_loss_map else None

            crossed_win_once   = False
            consec_loss_hit    = 0
            consec_win_recross = 0
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

                # ---------- STOP (always active, even before min_hold_bars) ----------
                low_j = df_sym.loc[j, "low"]
                if (stop_price is not None) and pd.notna(low_j) and (low_j < stop_price - 1e-12):
                    if (j + 1) < len(df_sym):
                        exit_date   = pd.to_datetime(df_sym.loc[j + 1, "date"])
                        exit_price  = float(df_sym.loc[j + 1, "open"])
                        exit_reason = "stop_next_open"
                    else:
                        exit_date   = date_j
                        exit_price  = price_j
                        exit_reason = "stop_eod"
                    break

                # ---------- Enhanced guards (gated by min_hold_bars) ----------
                if use_guards and (bars_held >= min_hold_bars):
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
                        # Trailing peak giveback (replaces avg-win re-cross)
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

                # ---------- SMA breakdown (now also gated by min_hold_bars) ----------
                if bars_held >= min_hold_bars:
                    below_count = 0
                    for w in [10, 20, 50, 200]:
                        sma_val = df_sym.loc[j, f"sma_{w}"]
                        if pd.notna(sma_val) and price_j < float(sma_val) - 1e-12:
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
                # still open; advance one bar to avoid re-entering immediately on the same setup
                i = j if 'j' in locals() else (i + 1)
            else:
                i = j + 1

    return pd.DataFrame(trades)
