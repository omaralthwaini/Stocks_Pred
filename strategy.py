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
    avg_win_map=None,          # dict: symbol -> avg win return (%)  (can be computed from ALL history)
    avg_loss_map=None,         # dict: symbol -> avg loss return (%) (negative number, from ALL history)
    enhanced_cutoff="2025-01-01"  # only entries on/after this date use the new guards
):
    """
    Entry (unchanged):
      • All SMAs (10/20/50/200) available
      • Close above the max of those SMAs
      • SMA10/20/50 each rising for k_days_rising
      • Green candle with minimum body (close-open)/open >= body_min (default 0.3%)

    Exit (base, unchanged):
      • Bar-level stop: if a future bar's LOW < prior-day LOW at entry -> exit next day's OPEN
        (or same day's CLOSE if last row)  => exit_reason 'stop_next_open' / 'stop_eod'
      • Or if a future CLOSE is below >= 2 of the 10/20/50/200 SMAs -> exit at that CLOSE
        => exit_reason 'sma_below_2'

    NEW 2025+ overlays (use stats from avg_*_map; stats can be computed from ALL history):
      • Loss guard: if today's CLOSE <= entry * (1 + avg_loss%) -> exit at today's CLOSE
        => exit_reason 'avg_loss_breach'
      • Giveback guard: once any day's HIGH >= entry * (1 + avg_win%), if a later CLOSE < that same
        win level -> exit at that CLOSE  => exit_reason 'giveback_below_avg_win'

      Notes:
      - If a symbol has no avg win/loss in the maps, that specific guard is skipped.
      - Guards apply only for entries with entry_date >= enhanced_cutoff.
      - Order inside the daily loop (2025+): loss guard -> (update win reached) -> giveback -> SMA breakdown -> stop-low.
        For pre-2025 entries, original order is used: stop-low -> SMA breakdown.
    """
    cutoff_ts = pd.Timestamp(enhanced_cutoff)
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

            # Enhanced guards only for 2025+ entries
            use_enhanced = pd.to_datetime(entry_date) >= cutoff_ts
            avg_win = (avg_win_map or {}).get(sym, np.nan)
            avg_loss = (avg_loss_map or {}).get(sym, np.nan)

            win_line = float(entry_price * (1.0 + avg_win/100.0)) if use_enhanced and pd.notna(avg_win) and avg_win > 0 else None
            loss_floor = float(entry_price * (1.0 + avg_loss/100.0)) if use_enhanced and pd.notna(avg_loss) and avg_loss < 0 else None
            reached_win = False  # becomes True once any future HIGH crosses win_line

            exit_date   = None
            exit_price  = None
            exit_reason = "force_close"

            # Manage forward
            for j in range(i + 1, len(df_sym)):
                price_close = float(df_sym.loc[j, "close"]) if pd.notna(df_sym.loc[j, "close"]) else np.nan
                price_high  = float(df_sym.loc[j, "high"])  if pd.notna(df_sym.loc[j, "high"])  else np.nan
                low_j       = df_sym.loc[j, "low"]

                if use_enhanced:
                    # 1) Loss guard: immediate exit at CLOSE if we fall to avg-loss level
                    if (loss_floor is not None) and pd.notna(price_close) and (price_close <= loss_floor):
                        exit_date   = df_sym.loc[j, "date"]
                        exit_price  = price_close
                        exit_reason = "avg_loss_breach"
                        break

                    # 2) Track whether we've reached the avg-win line
                    if (win_line is not None) and pd.notna(price_high) and (price_high >= win_line):
                        reached_win = True

                    # 3) Giveback guard: once reached win_line, exit if CLOSE < win_line
                    if reached_win and (win_line is not None) and pd.notna(price_close) and (price_close < win_line):
                        exit_date   = df_sym.loc[j, "date"]
                        exit_price  = price_close
                        exit_reason = "giveback_below_avg_win"
                        break

                    # 4) SMA breakdown: close below >= 2 SMAs -> exit at close
                    below_count = 0
                    for w in [10, 20, 50, 200]:
                        sma_val = df_sym.loc[j, f"sma_{w}"]
                        if pd.notna(sma_val) and price_close < float(sma_val):
                            below_count += 1
                    if below_count >= 2:
                        exit_date   = df_sym.loc[j, "date"]
                        exit_price  = price_close
                        exit_reason = "sma_below_2"
                        break

                    # 5) Prev-day-low stop (as a fallback; typically trumped by loss guard)
                    if stop_price is not None and pd.notna(low_j) and (low_j < stop_price):
                        if (j + 1) < len(df_sym):
                            exit_date   = df_sym.loc[j + 1, "date"]
                            exit_price  = float(df_sym.loc[j + 1, "open"])
                            exit_reason = "stop_next_open"
                        else:
                            exit_date   = df_sym.loc[j, "date"]
                            exit_price  = price_close
                            exit_reason = "stop_eod"
                        break

                else:
                    # ---- Pre-2025 behavior (original order) ----
                    # Stop logic first
                    if stop_price is not None and pd.notna(low_j) and (low_j < stop_price):
                        if (j + 1) < len(df_sym):
                            exit_date   = df_sym.loc[j + 1, "date"]
                            exit_price  = float(df_sym.loc[j + 1, "open"])
                            exit_reason = "stop_next_open"
                        else:
                            exit_date   = df_sym.loc[j, "date"]
                            exit_price  = price_close
                            exit_reason = "stop_eod"
                        break

                    # SMA breakdown
                    below_count = 0
                    for w in [10, 20, 50, 200]:
                        sma_val = df_sym.loc[j, f"sma_{w}"]
                        if pd.notna(sma_val) and price_close < float(sma_val):
                            below_count += 1
                    if below_count >= 2:
                        exit_date   = df_sym.loc[j, "date"]
                        exit_price  = price_close
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
