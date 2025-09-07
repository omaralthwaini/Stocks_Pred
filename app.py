# app.py
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st
from arch import arch_model  # for GARCH
from strategy import run_strategy
import requests

# --------------------------------------------------------------------------------------
# App config
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Smart Backtester", layout="wide")
st.title("📈 Smart Backtester")

# Toggle crypto usage in the app (kept False since your subscription excludes it)
INCLUDE_CRYPTO = True

# Auto threshold settings (global, always on)
AUTO_MIN_TRADES = 3
AUTO_GRID_STEP  = 0.05

# --------------------------------------------------------------------------------------
# Secrets/env helper
# --------------------------------------------------------------------------------------
def get_polygon_key() -> str:
    """Get POLYGON_API_KEY from st.secrets or env, trimmed. Empty string if missing."""
    key = ""
    try:
        key = st.secrets["POLYGON_API_KEY"]
    except Exception:
        key = os.getenv("POLYGON_API_KEY", "")
    return (key or "").strip()

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def add_rownum(df_in):
    df = df_in.copy()
    df.insert(0, "#", range(1, len(df) + 1))
    return df

def pct_str(x, digits=2, signed=True):
    if pd.isna(x): return "—"
    fmt = f"{{:{'+' if signed else ''}.{digits}f}}%"
    return fmt.format(x)

def money_str(x):
    return "—" if pd.isna(x) else f"${x:,.2f}"

def date_only_cols(df_in, cols=("entry_date","exit_date")):
    df = df_in.copy()
    for c in cols:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            df[c] = s.dt.strftime("%Y-%m-%d").where(s.notna(), df[c])
    return df

def _file_sig(path: str) -> str:
    """Tiny fingerprint of a file to use as a cache key (mtime + size)."""
    try:
        s = os.stat(path)
        return f"{s.st_mtime_ns}-{s.st_size}"
    except FileNotFoundError:
        return f"missing-{time.time()}"

# --------------------------------------------------------------------------------------
# Light Polygon client (daily OHLCV)
# --------------------------------------------------------------------------------------
def _log(msg: str):
    st.write(msg)

def _polygon_get(url: str, params=None, timeout=30, max_retries=4):
    api_key = get_polygon_key()
    if not api_key:
        _log("⚠️ POLYGON_API_KEY not set; skipping request.")
        return None
    params = (params or {}).copy()
    params["apiKey"] = api_key

    backoff = 2
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                _log(f"⚠️ Polygon {r.status_code} (attempt {attempt}/{max_retries}) → retrying after {backoff}s…")
                time.sleep(backoff)
                backoff *= 2
                continue
            _log(f"❌ Polygon HTTP {r.status_code}: {r.text[:200]}")
            return None
        except requests.RequestException as e:
            _log(f"❌ Polygon request error: {e} (attempt {attempt}/{max_retries})")
            time.sleep(backoff)
            backoff *= 2
    return None

def fetch_polygon_daily(symbol: str, start: str, end: str, asset_type: str) -> pd.DataFrame:
    """
    Fetch daily bars from Polygon.
    - Stocks: uses Polygon.
    - Crypto: SKIPPED here (you refresh it via the Binance/Crypto script).
    """
    at = (asset_type or "stock").lower()
    if at == "crypto":
        return pd.DataFrame()

    ticker = symbol  # stocks only
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    r = _polygon_get(url, params={"adjusted": "true", "sort": "asc", "limit": 50000})
    if r is None or r.status_code != 200:
        return pd.DataFrame()

    results = r.json().get("results", [])
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})[
        ["date","open","high","low","close","volume"]
    ]
    df["symbol"] = symbol
    return df

# --------------------------------------------------------------------------------------
# GARCH: risk index
# --------------------------------------------------------------------------------------
def garch_volatility_forecast(series: pd.Series) -> float | None:
    """
    GARCH(1,1) on decimal log-returns (mean='Zero', dist='t').
    Returns 1D-ahead annualized volatility in DECIMAL units (e.g., 0.22 = 22%).
    """
    px = pd.Series(series).astype(float)
    px = px.replace([np.inf, -np.inf], np.nan).dropna()
    px = px[px > 0]
    rets = np.log(px / px.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(rets) < 50:
        return None
    try:
        am = arch_model(rets, vol="GARCH", p=1, q=1, mean="Zero", dist="t")
        res = am.fit(disp='off')
        fc = res.forecast(horizon=1, reindex=False)
        var1 = float(fc.variance.iloc[-1, 0])
        ann_vol = float(np.sqrt(var1) * np.sqrt(252))
        return ann_vol if np.isfinite(ann_vol) and ann_vol > 0 else None
    except Exception:
        return None

# --------------------------------------------------------------------------------------
# OAAT simulator (with GARCH threshold)
# --------------------------------------------------------------------------------------
def simulate_oaat(
    trades_in: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    starting_capital: float = 10000.0,
    alloc_pct: float = 100.0,
    *,
    threshold: float = 0.50,
    require_garch: bool = True
):
    t = trades_in.copy().sort_values("entry_date")
    if require_garch:
        t = t[t["garch_risk_index"].notna()]
    else:
        t["garch_risk_index"] = t["garch_risk_index"].fillna(1.0)

    t = t[t["garch_risk_index"].ge(threshold)]

    capital = float(starting_capital)
    available_from = start_ts
    taken = []

    for _, r in t.iterrows():
        entry_d = pd.to_datetime(r["entry_date"])
        if entry_d < available_from:
            continue

        entry = float(r["entry"])
        ex_d  = pd.to_datetime(r["exit_date"]) if pd.notna(r["exit_date"]) else pd.NaT
        realized = pd.notna(ex_d) and (ex_d <= end_ts)

        if realized:
            exit_px = float(r["exit_price"])
            ret_pct = (exit_px / entry - 1.0) * 100.0
            available_from = ex_d + pd.Timedelta(days=1)
            exit_date = ex_d
            status = "Realized"
        else:
            exit_px = float(r["latest_close"])
            ret_pct = (exit_px / entry - 1.0) * 100.0
            available_from = end_ts + pd.Timedelta(days=1)
            exit_date = pd.NaT
            status = "Unrealized"

        invest_amt = capital * (alloc_pct / 100.0)
        capital = capital - invest_amt + invest_amt * (1.0 + ret_pct / 100.0)

        taken.append({
            "symbol": r["symbol"],
            "entry_date": entry_d,
            "exit_date": exit_date,
            "entry": entry,
            "exit_or_last": exit_px,
            "ret_pct": ret_pct,
            "status": status,
            "capital_after": capital
        })

    if not taken:
        res = pd.DataFrame(columns=[
            "symbol","entry_date","exit_date","entry","exit_or_last","ret_pct","status","capital_after"
        ])
        metrics = dict(
            final_capital=starting_capital,
            total_return=0.0,
            trades=0,
            win_rate=float("nan"),
            avg_win_ret=float("nan"),
            max_win=float("nan"),
            max_loss=float("nan"),
        )
        return metrics, res

    res = pd.DataFrame(taken).sort_values("entry_date").reset_index(drop=True)

    n_trades = len(res)
    n_real = (res["status"] == "Realized").sum()
    win_rate = (res.loc[res["status"] == "Realized", "ret_pct"] > 0).mean() if n_real else float("nan")
    avg_win_ret = res.loc[(res["status"] == "Realized") & (res["ret_pct"] > 0), "ret_pct"].mean()
    max_win = res["ret_pct"].max()
    max_loss = res["ret_pct"].min()
    total_ret = (capital / starting_capital - 1.0) * 100.0

    metrics = dict(
        final_capital=capital,
        total_return=total_ret,
        trades=n_trades,
        win_rate=win_rate,
        avg_win_ret=avg_win_ret,
        max_win=max_win,
        max_loss=max_loss,
    )
    return metrics, res

# --------------------------------------------------------------------------------------
# Attach GARCH risk index to trades
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def attach_garch_risk_index(df_prices: pd.DataFrame,
                            trades_df: pd.DataFrame,
                            base_vol_dec: float = 0.20,
                            lookback: int | None = 252) -> pd.DataFrame:
    """
    Adds garch_risk_index ∈ (0,1], min(1, base_vol / ann_vol), using price history strictly
    before each entry_date, optionally capped by lookback.
    """
    df_sorted = df_prices.sort_values(["symbol", "date"])
    groups = {sym: g[["date","close"]].reset_index(drop=True)
              for sym, g in df_sorted.groupby("symbol")}

    out = []
    for _, r in trades_df.iterrows():
        sym = r["symbol"]
        entry_date = pd.to_datetime(r["entry_date"])
        if sym not in groups:
            out.append(np.nan); continue
        g = groups[sym]
        hist = g[g["date"] < entry_date]["close"]
        if lookback and lookback > 0:
            hist = hist.tail(lookback)
        if len(hist) < 51:
            out.append(np.nan); continue
        ann_vol = garch_volatility_forecast(hist)
        if ann_vol is None or not np.isfinite(ann_vol) or ann_vol <= 1e-9:
            out.append(np.nan); continue
        out.append(float(min(1.0, base_vol_dec / ann_vol)))

    trades2 = trades_df.copy()
    trades2["garch_risk_index"] = out
    return trades2

# --------------------------------------------------------------------------------------
# Auto-learn per-symbol thresholds from CLOSED trades only
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def optimize_thresholds_per_symbol_closed(
    trades_all: pd.DataFrame,
    *,
    step: float = AUTO_GRID_STEP,
    min_trades: int = AUTO_MIN_TRADES,
) -> tuple[dict, pd.DataFrame]:
    """
    For each symbol, sweep thresholds in [0,1] (step=step) on CLOSED trades only,
    simulate OAAT, and select the threshold that maximizes total return,
    with at least min_trades at that threshold. Tie-breakers: more trades, then lower thr.
    """
    closed = trades_all[trades_all["exit_date"].notna()].copy()
    if closed.empty:
        sym_set = sorted(trades_all["symbol"].unique())
        return ({s: 0.00 for s in sym_set},
                pd.DataFrame(columns=["symbol","threshold","trades","total_return_%","win_rate_%"]))

    start_ts = closed["entry_date"].min()
    end_ts   = closed["exit_date"].max()

    thr_values = np.round(np.arange(0.0, 1.0 + 1e-9, step), 2)
    rows, best_map = [], {}

    for sym, cand in closed.groupby("symbol"):
        best = None  # tuple(score, trades, -thr, thr)
        for thr in thr_values:
            m, _ = simulate_oaat(
                cand, start_ts, end_ts,
                starting_capital=10000.0, alloc_pct=100.0,
                threshold=thr, require_garch=True
            )
            tr = m["trades"]
            score = m["total_return"]
            rows.append({
                "symbol": sym,
                "threshold": thr,
                "trades": tr,
                "total_return_%": score,
                "win_rate_%": (m["win_rate"]*100 if pd.notna(m["win_rate"]) else np.nan),
            })
            if tr < min_trades:
                continue
            candidate = (score, tr, -thr, thr)
            if (best is None) or (candidate > best):
                best = candidate
        best_map[sym] = (best[3] if best is not None else 0.00)

    return best_map, pd.DataFrame(rows)

# --------------------------------------------------------------------------------------
# Load prices & caps with file-signature cache keys
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw_prices(_sig: str) -> pd.DataFrame:
    df = pd.read_csv("stocks.csv", parse_dates=["date"])

    # Ensure expected columns exist
    if "asset_type" not in df.columns:
        df["asset_type"] = "stock"
    if "sector" not in df.columns:
        df["sector"] = None

    # Normalize and **filter first** → keep only stocks
    df["asset_type"] = df["asset_type"].astype(str).str.lower()
    df = df[df["asset_type"] == "stock"].copy()

    # Tidy sector so UI doesn’t show “None” chips
    df["sector"] = df["sector"].astype("object")
    df.loc[df["sector"].isna() | (df["sector"].astype(str).str.strip() == ""), "sector"] = "Stocks"

    # Optional: de-dup + sort (harmless and keeps things clean)
    df = (df
          .drop_duplicates(subset=["symbol", "date"])
          .sort_values(["symbol", "date"])
          .reset_index(drop=True))
    return df


@st.cache_data(show_spinner=False)
def load_caps(_sig: str) -> pd.DataFrame:
    return pd.read_csv("market_cap.csv")

# --------------------------------------------------------------------------------------
# Manual "update today" (overwrite today's rows)
# --------------------------------------------------------------------------------------
def _today_str():
    # Use calendar date (naive). If you prefer ET: from datetime import timezone; convert.
    return datetime.now().strftime("%Y-%m-%d")

def update_prices_today(df_existing: pd.DataFrame,
                        include_crypto: bool = INCLUDE_CRYPTO) -> pd.DataFrame:
    """
    Fetch today's daily bar for each symbol (stocks; crypto optional),
    overwrite any existing (symbol, today) rows in stocks.csv, and return the combined df.
    """
    if df_existing.empty or "symbol" not in df_existing.columns or "date" not in df_existing.columns:
        st.error("stocks.csv is missing required columns.")
        return df_existing

    if not get_polygon_key():
        st.error("POLYGON_API_KEY not set; cannot update.")
        return df_existing

    # Build symbol meta (symbol, asset_type, sector)
    meta = (df_existing[["symbol","asset_type","sector"]]
            .drop_duplicates()
            .reset_index(drop=True))
    if not include_crypto:
        meta = meta[meta["asset_type"].str.lower() != "crypto"]

    if meta.empty:
        st.warning("No symbols to update.")
        return df_existing

    start = end = _today_str()
    frames = []

    with st.spinner(f"🔄 Fetching today's bars ({start})…"):
        for _, row in meta.iterrows():
            sym = str(row["symbol"]).strip()
            a_type = (row["asset_type"] or "stock").lower()
            if not sym:
                continue
            df_new = fetch_polygon_daily(sym, start, end, a_type)
            if df_new.empty:
                continue
            # Attach metadata for storage
            df_new["symbol"] = sym
            df_new["asset_type"] = a_type
            df_new["sector"] = row.get("sector", None)
            frames.append(df_new)

    if not frames:
        st.info("No new data returned by Polygon for today. File not changed.")
        return df_existing

    new_data = pd.concat(frames, ignore_index=True)

    # Normalize datetimes to date for keying
    new_data["date"] = pd.to_datetime(new_data["date"]).dt.normalize()
    df_existing["date"] = pd.to_datetime(df_existing["date"]).dt.normalize()

    # Remove collisions for (symbol, date) present in new_data (i.e., overwrite today's rows)
    keys = new_data[["symbol","date"]].drop_duplicates()
    before = len(df_existing)
    merged = df_existing.merge(keys, on=["symbol","date"], how="left", indicator=True)
    existing_filtered = merged[merged["_merge"] == "left_only"].drop(columns="_merge")
    dropped = before - len(existing_filtered)

    combined = pd.concat([new_data, existing_filtered], ignore_index=True)
    combined = (combined
                .drop_duplicates(subset=["symbol","date"], keep="first")
                .sort_values(["symbol","date"])
                .reset_index(drop=True))
    combined.to_csv("stocks.csv", index=False)

    st.success(f"✅ Updated stocks.csv for today. Overwrote {dropped} rows; added {len(new_data)} new rows.")
    return combined

# --------------------------------------------------------------------------------------
# Sidebar: manual update; auto-reload latest files via signatures
# --------------------------------------------------------------------------------------
st.sidebar.header("Data")
st.sidebar.caption(f"Polygon key: {'✅ found' if get_polygon_key() else '❌ missing'}")

# Create cache keys from on-disk state; clear caches only when signatures change
stocks_sig_now = _file_sig("stocks.csv")
caps_sig_now   = _file_sig("market_cap.csv")

if (st.session_state.get("_stocks_sig") != stocks_sig_now) or (st.session_state.get("_caps_sig") != caps_sig_now):
    st.session_state["_stocks_sig"] = stocks_sig_now
    st.session_state["_caps_sig"]   = caps_sig_now
    try: st.cache_data.clear()
    except Exception: pass
    try: st.cache_resource.clear()
    except Exception: pass

# Load current files using signatures (so cache invalidates when files change)
df0  = load_raw_prices(stocks_sig_now)
caps = load_caps(caps_sig_now)

if st.sidebar.button("🔄 Update data now"):
    try:
        _ = update_prices_today(df0, include_crypto=INCLUDE_CRYPTO)
        # After writing stocks.csv, update signature → clear caches → rerun
        st.session_state["_stocks_sig"] = _file_sig("stocks.csv")
        try: st.cache_data.clear()
        except Exception: pass
        try: st.cache_resource.clear()
        except Exception: pass
        if hasattr(st, "rerun"): st.rerun()
        else: st.experimental_rerun()
    except Exception as e:
        st.error(f"Update failed: {e}")
        
# Universe selection
#   - Stocks: filter by market_cap (remove cap 3 & 4, then take top 100 by cap_score)
#   - Crypto: skipped unless INCLUDE_CRYPTO=True
# --------------------------------------------------------------------------------------
stocks_only = df0[df0["asset_type"] == "stock"].copy()
stocks_caps = caps[~caps["cap_score"].isin([3, 4])].sort_values("cap_score")
top_stock_symbols = stocks_caps.head(100)["symbol"].unique().tolist()

if INCLUDE_CRYPTO:
    crypto_only = df0[df0["asset_type"] == "crypto"].copy()
    crypto_symbols = crypto_only["symbol"].unique().tolist()
else:
    crypto_symbols = []

final_symbols = set(top_stock_symbols) | set(crypto_symbols)
df = df0[df0["symbol"].isin(final_symbols)].copy().sort_values(["symbol","date"])

# cap maps (won’t exist for crypto; that’s fine)
cap_score_map = caps.set_index("symbol")["cap_score"]
cap_emoji_map = caps.set_index("symbol")["cap_emoji"]

# --------------------------------------------------------------------------------------
# Strategy → Trades
# --------------------------------------------------------------------------------------
with st.spinner("⏳ Detecting trades..."):
    trades_raw = run_strategy(df)

if trades_raw.empty:
    st.warning("No trades detected.")
    st.stop()

# Attach GARCH & auto-apply per-symbol thresholds
with st.spinner("🔎 Computing GARCH risk index..."):
    trades_all = attach_garch_risk_index(df, trades_raw, base_vol_dec=0.20, lookback=252)

before_n = len(trades_all)

with st.spinner("🧠 Learning per-ticker thresholds from CLOSED trades..."):
    thr_map, sweep_long = optimize_thresholds_per_symbol_closed(
        trades_all, step=AUTO_GRID_STEP, min_trades=AUTO_MIN_TRADES
    )

def _accept_row(r):
    thr = thr_map.get(r["symbol"], 0.00)  # default if symbol had no closed trades
    return pd.notna(r["garch_risk_index"]) and (r["garch_risk_index"] >= thr)

trades = trades_all[trades_all.apply(_accept_row, axis=1)].copy()
after_n = len(trades)

with st.expander("Per-symbol auto thresholds (learned on CLOSED trades)"):
    thr_df = pd.DataFrame(
        [{"symbol": s, "auto_threshold": thr_map[s]} for s in sorted(thr_map.keys())]
    ).sort_values(["auto_threshold","symbol"])
    st.dataframe(thr_df, use_container_width=True, hide_index=True)

with st.expander("Threshold sweep results (per symbol)"):\

    if not sweep_long.empty:
        st.dataframe(sweep_long.sort_values(["symbol","threshold"]),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No closed trades available to train thresholds.")

st.caption(
    f"Auto per-ticker thresholds applied (trained on CLOSED trades; min_trades={AUTO_MIN_TRADES}, step={AUTO_GRID_STEP:.2f}). "
    f"Kept {after_n}/{before_n} trades."
)

if trades.empty:
    st.warning("All trades were filtered out by auto thresholds. Likely too few closed trades yet.")
    st.stop()

# --------------------------------------------------------------------------------------
# Postprocess & enrich
# --------------------------------------------------------------------------------------
sector_map = df[["symbol", "sector"]].drop_duplicates().set_index("symbol")["sector"]
trades["sector"] = trades["symbol"].map(sector_map)
trades["cap_score"] = trades["symbol"].map(cap_score_map)
trades["cap_emoji"] = trades["symbol"].map(cap_emoji_map)
trades["symbol_display"] = trades.apply(
    lambda r: f"{r['cap_emoji']} {r['symbol']}" if pd.notna(r["cap_emoji"]) else r["symbol"], axis=1
)

# Stop loss from yesterday's low
df["stop_loss"] = df.groupby("symbol")["low"].shift(1)
entry_lows = df[["symbol", "date", "stop_loss"]].rename(columns={"date": "entry_date"})
trades = trades.merge(entry_lows, on=["symbol", "entry_date"], how="left")

# Final % return
latest_prices = df.groupby("symbol", as_index=False).agg(latest_close=("close", "last"))
trades = trades.merge(latest_prices, on="symbol", how="left")
trades["pct_return"] = (trades["exit_price"] / trades["entry"] - 1) * 100
trades["unrealized_pct_return"] = (trades["latest_close"] / trades["entry"] - 1) * 100
trades["final_pct"] = trades.apply(
    lambda r: r["pct_return"] if pd.notna(r["exit_price"]) else r["unrealized_pct_return"], axis=1
)

# Min/Max since entry (for open trades)
# --- Per-trade min/max (works for open + closed) ---
def add_minmax_for_trades(df_prices: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    prices = df_prices[["symbol", "date", "low", "high"]].copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices = prices.dropna(subset=["date"])

    # latest available date per symbol (used for open trades)
    last_date_by_sym = prices.groupby("symbol")["date"].max()

    rows = []
    for _, tr in trades_df.iterrows():
        sym = tr["symbol"]
        start = pd.to_datetime(tr["entry_date"], errors="coerce")
        # closed → bound by exit_date; open → bound by latest price date
        end = (pd.to_datetime(tr["exit_date"], errors="coerce")
               if pd.notna(tr["exit_date"])
               else last_date_by_sym.get(sym, pd.NaT))

        if pd.isna(start) or pd.isna(end):
            continue

        sl = prices[(prices["symbol"] == sym) &
                    (prices["date"] >= start) &
                    (prices["date"] <= end)]
        if sl.empty:
            continue

        rows.append({
            "symbol": sym,
            "entry_date": start,
            "min_low": float(sl["low"].min()),
            "max_high": float(sl["high"].max()),
        })

    mm = pd.DataFrame(rows)
    return trades_df.merge(mm, on=["symbol", "entry_date"], how="left")

# apply
trades = add_minmax_for_trades(df, trades)


# Closed perf aggregates for display (avg_return / avg_win / win_rate)
closed = trades[trades["exit_date"].notna()].copy()
if not closed.empty:
    closed["pct_return"] = (closed["exit_price"] / closed["entry"] - 1) * 100
    closed["win"] = closed["pct_return"] > 0

    avg_return = closed.groupby("symbol")["pct_return"].mean().rename("avg_return")
    avg_win_return = (closed[closed["pct_return"] > 0]
                      .groupby("symbol")["pct_return"].mean()
                      .rename("avg_win_return"))
    win_rate = closed.groupby("symbol")["win"].mean().rename("win_rate")

    trades = (trades
              .merge(avg_return, on="symbol", how="left")
              .merge(avg_win_return, on="symbol", how="left")
              .merge(win_rate, on="symbol", how="left"))
else:
    trades["avg_return"] = None
    trades["avg_win_return"] = None
    trades["win_rate"] = None

# --------------------------------------------------------------------------------------
# PAGES
# --------------------------------------------------------------------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose page", ["Home", "Insights", "Tester", "Compare"], index=0)

# ---------- HOME ----------
if page == "Home":
    st.subheader("🔓 Open Trades")

    open_trades = trades[trades["exit_date"].isna()].copy()
    if open_trades.empty:
        st.info("No open trades.")
    else:
        unrealized_total = open_trades["unrealized_pct_return"].mean()
        earliest_entry = open_trades["entry_date"].min()
        avg_hold = (pd.Timestamp.today().normalize() - open_trades["entry_date"]).dt.days.mean()
        best_row = open_trades.loc[open_trades["unrealized_pct_return"].idxmax()]
        worst_row = open_trades.loc[open_trades["unrealized_pct_return"].idxmin()]
        best_symbol = f"{best_row['symbol']} ({pct_str(best_row['unrealized_pct_return'])}) ✅"
        worst_symbol = f"{worst_row['symbol']} ({pct_str(worst_row['unrealized_pct_return'])}) ❌"

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Unrealized P/L", pct_str(unrealized_total))
        c2.metric("Open Trades", f"{len(open_trades)}")
        c3.metric("Earliest Entry", earliest_entry.strftime("%Y-%m-%d"))

        c4, c5, c6 = st.columns(3)
        c4.metric("Avg Holding (days)", f"{avg_hold:.1f}")
        c5.metric("Best Performer", best_symbol)
        c6.metric("Worst Performer", worst_symbol)

        open_trades = open_trades.sort_values(["entry_date", "avg_return"], ascending=[False, False])

        def emoji_unrealized(x):
            if pd.isna(x): return "—"
            return f"✅ {pct_str(x)}" if x > 0 else f"❌ {pct_str(x)}"

        table = open_trades[[
            "symbol_display", "sector", "entry_date", "entry", "stop_loss",
            "avg_return", "avg_win_return", "win_rate", "unrealized_pct_return"
        ]].copy()

        table = date_only_cols(table, ["entry_date"])
        table["entry"] = table["entry"].map(money_str)
        table["stop_loss"] = table["stop_loss"].map(money_str)
        table["avg_return"] = table["avg_return"].map(lambda x: pct_str(x))
        table["avg_win_return"] = table["avg_win_return"].map(lambda x: pct_str(x))
        table["win_rate"] = table["win_rate"].map(lambda x: "—" if pd.isna(x) else f"{x:.0%}")
        table["unrealized_pct_return"] = table["unrealized_pct_return"].map(emoji_unrealized)

        table = table.rename(columns={
            "symbol_display": "Symbol",
            "sector": "Sector",
            "entry_date": "Entry Date",
            "entry": "Entry",
            "stop_loss": "Stop Loss",
            "avg_return": "Avg Return",
            "avg_win_return": "Avg Win",
            "win_rate": "Win Rate",
            "unrealized_pct_return": "Unrealized",
        })

        st.dataframe(add_rownum(table), use_container_width=True, hide_index=True)
        # ---------------- Alarm list: 1-day drop alert ----------------
        # Build per-symbol 1D % change using the latest row vs previous close
        df_tmp = df[["symbol", "date", "close"]].copy().sort_values(["symbol", "date"])
        df_tmp["prev_close"] = df_tmp.groupby("symbol")["close"].shift(1)
        df_tmp["day_change_pct"] = (df_tmp["close"] / df_tmp["prev_close"] - 1.0) * 100.0

        # Take the most recent row per symbol
        latest_per_sym = (
            df_tmp.groupby("symbol", as_index=False)
                 .tail(1)[["symbol", "day_change_pct"]]
        )

        # Join onto current OPEN trades
        alarms = (
            open_trades[["symbol", "symbol_display", "sector", "entry_date",
                         "entry", "latest_close", "unrealized_pct_return"]]
            .merge(latest_per_sym, on="symbol", how="left")
        )

        # Trigger when 1D change <= -ALARM_DAILY_DROP_PCT (defaults to -2.0%)
        threshold = -float(globals().get("ALARM_DAILY_DROP_PCT", 2.0))
        alarms = alarms[alarms["day_change_pct"].le(threshold)].copy()

        if not alarms.empty:
            alarms = alarms.sort_values("day_change_pct")  # worst first
            # Format columns
            alarms_fmt = alarms.copy()
            alarms_fmt = date_only_cols(alarms_fmt, ["entry_date"])
            alarms_fmt["P&L Now"] = alarms_fmt["unrealized_pct_return"].map(lambda x: pct_str(x))
            alarms_fmt["1D Change"] = alarms_fmt["day_change_pct"].map(lambda x: pct_str(x))
            alarms_fmt["Entry"] = alarms_fmt["entry"].map(money_str)

            show_cols = ["symbol_display", "sector", "entry_date", "Entry", "1D Change", "P&L Now"]
            alarms_fmt = alarms_fmt.rename(columns={
            "symbol_display": "Symbol",
            "sector": "Sector",
            "entry_date": "Entry Date",
            })

            show_cols = ["Symbol", "Sector", "Entry Date", "Entry", "1D Change", "P&L Now"]
            alarms_fmt = alarms_fmt[show_cols]


            st.subheader(f"🚨 Alarm List — 1-Day Drop ≥ {globals().get('ALARM_DAILY_DROP_PCT', 2.0):.0f}%")
            st.dataframe(add_rownum(alarms_fmt), use_container_width=True, hide_index=True)
        else:
            st.subheader(f"🚨 Alarm List — 1-Day Drop ≥ {globals().get('ALARM_DAILY_DROP_PCT', 2.0):.0f}%")
            st.info("No alerts right now.")

# ---------- INSIGHTS ----------
if page == "Insights":
    st.subheader("📊 Closed Trades Insights")

    closed = trades[trades["exit_date"].notna()].copy()
    if closed.empty:
        st.info("No closed trades to analyze.")
    else:
        closed["pct_return"] = (closed["exit_price"] / closed["entry"] - 1) * 100
        closed["win"] = closed["pct_return"] > 0
        closed["days_held"] = (closed["exit_date"] - closed["entry_date"]).dt.days

        agg = closed.groupby("symbol").agg(
            n_trades=("pct_return", "size"),
            avg_return=("pct_return", "mean"),
            avg_days=("days_held", "mean"),
            win_rate=("win", "mean"),
        ).reset_index()

        def total_ret_percent(s: pd.Series) -> float:
            return (np.prod(1.0 + s.values/100.0) - 1.0) * 100.0

        tot = (
            closed.groupby("symbol")["pct_return"]
                  .apply(total_ret_percent)
                  .rename("total_return_%")
                  .reset_index()
        )

        win_mean = (closed[closed["pct_return"] > 0]
                    .groupby("symbol")["pct_return"].mean()
                    .rename("avg_win_return"))
        loss_mean = (closed[closed["pct_return"] < 0]
                     .groupby("symbol")["pct_return"].mean()
                     .rename("avg_loss_return"))

        best = (agg
                .merge(win_mean, on="symbol", how="left")
                .merge(loss_mean, on="symbol", how="left")
                .merge(tot, on="symbol", how="left"))

        best["sector"] = best["symbol"].map(sector_map)
        best["symbol_display"] = best["symbol"].map(lambda s: f"{cap_emoji_map.get(s,'')} {s}")

        best = best.sort_values("total_return_%", ascending=False)

        disp = best.copy()
        disp["win_rate_str"]         = disp["win_rate"].map(lambda x: "—" if pd.isna(x) else f"{x:.0%}")
        disp["total_return_str"]     = disp["total_return_%"].map(lambda x: pct_str(x))
        disp["avg_return_str"]       = disp["avg_return"].map(lambda x: pct_str(x))
        disp["avg_win_return_str"]   = disp["avg_win_return"].map(lambda x: pct_str(x))
        disp["avg_loss_return_str"]  = disp["avg_loss_return"].map(lambda x: pct_str(x))
        disp["avg_days_str"]         = disp["avg_days"].map(lambda x: "—" if pd.isna(x) else f"{x:.1f}")

        show_df = add_rownum(disp[[
            "symbol_display", "sector", "n_trades",
            "win_rate_str", "total_return_str", "avg_return_str",
            "avg_win_return_str", "avg_loss_return_str", "avg_days_str"
        ]])

        st.dataframe(show_df.rename(columns={
            "symbol_display": "Symbol",
            "sector": "Sector",
            "n_trades": "Trades",
            "win_rate_str": "Win Rate",
            "total_return_str": "Total Return",
            "avg_return_str": "Avg/Trade",
            "avg_win_return_str": "Avg Win",
            "avg_loss_return_str": "Avg Loss",
            "avg_days_str": "Avg Days"
        }), use_container_width=True, hide_index=True)

        st.subheader("📦 All Trades (Open + Closed)")
        all_trades = trades.copy().sort_values("entry_date", ascending=False)
        all_trades = date_only_cols(all_trades, ["entry_date", "exit_date"])

        display = all_trades[[
            "symbol_display", "sector", "entry_date", "exit_date",
            "entry", "exit_price", "final_pct", "stop_loss", "min_low", "max_high"
        ]].copy()

        display["entry"] = display["entry"].map(money_str)
        display["exit_price"] = display["exit_price"].map(money_str)
        display["final_pct"] = display["final_pct"].map(lambda x: pct_str(x))

        st.dataframe(add_rownum(display), use_container_width=True, hide_index=True)

        st.download_button(
            label="📥 Download All Trades",
            data=display.to_csv(index=False).encode("utf-8"),
            file_name="all_trades.csv",
            mime="text/csv"
        )

# ---------- TESTER ----------
if page == "Tester":
    st.subheader("🧪 One-Position-At-A-Time Backtest")

    start_date = st.sidebar.date_input("Start date", value=trades["entry_date"].min().date())
    end_date = st.sidebar.date_input("End date", value=trades["entry_date"].max().date())
    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)

    tickers_in_window = trades[
        (trades["entry_date"] >= start_ts) &
        (trades["entry_date"] <= end_ts)
    ]["symbol"].unique().tolist()

    selected_tickers = st.sidebar.multiselect(
        "Optional ticker filter (within window)", options=sorted(tickers_in_window),
        default=tickers_in_window
    )

    starting_capital = st.sidebar.number_input("Starting Capital ($)", min_value=1000.0, step=100.0, value=10000.0)
    alloc_pct = st.sidebar.number_input("Allocation per trade (%)", min_value=1.0, max_value=100.0, step=1.0, value=100.0)

    candidates = trades[
        (trades["entry_date"] >= start_ts) &
        (trades["entry_date"] <= end_ts) &
        (trades["symbol"].isin(selected_tickers))
    ].copy().sort_values("entry_date")

    if candidates.empty:
        st.info("No trades match the selected window and tickers.")
        st.stop()

    capital = float(starting_capital)
    available_from = start_ts
    ledger = []

    for _, r in candidates.iterrows():
        entry_d = pd.to_datetime(r["entry_date"])
        if entry_d < available_from:
            continue

        sym = r["symbol"]
        entry = float(r["entry"])
        ex_d = pd.to_datetime(r["exit_date"]) if pd.notna(r["exit_date"]) else pd.NaT
        realized = pd.notna(ex_d) and (ex_d <= end_ts)

        if realized:
            exit_px = float(r["exit_price"])
            ret_pct = (exit_px / entry - 1.0) * 100.0
            exit_d = ex_d
            status = "Realized"
            available_from = ex_d + pd.Timedelta(days=1)
        else:
            exit_px = float(r["latest_close"])
            ret_pct = (exit_px / entry - 1.0) * 100.0
            exit_d = pd.NaT
            status = "Unrealized"
            available_from = end_ts + pd.Timedelta(days=1)

        invest_amt = capital * (alloc_pct / 100.0)
        capital = capital - invest_amt + invest_amt * (1.0 + ret_pct / 100.0)

        ledger.append({
            "symbol": sym,
            "entry_date": entry_d,
            "exit_date": exit_d,
            "entry": entry,
            "exit_or_last": exit_px,
            "ret_pct": ret_pct,
            "status": status,
            "capital_after": capital
        })

    res = pd.DataFrame(ledger).sort_values("entry_date").reset_index(drop=True)

    n_trades = len(res)
    n_real = (res["status"] == "Realized").sum()
    win_rate = (res.loc[res["status"] == "Realized", "ret_pct"] > 0).mean() if n_real else float("nan")
    avg_win_ret = res.loc[(res["status"] == "Realized") & (res["ret_pct"] > 0), "ret_pct"].mean()
    max_win = res["ret_pct"].max()
    max_loss = res["ret_pct"].min()
    total_ret = (capital / starting_capital - 1.0) * 100.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Final capital", money_str(capital))
    c2.metric("Total return", pct_str(total_ret))
    c3.metric("Trades taken", f"{n_trades}")

    c4, c5, c6, c7 = st.columns(4)
    c4.metric("Max Win %", pct_str(max_win))
    c5.metric("Max Loss %", pct_str(max_loss))
    c6.metric("Avg Win %", pct_str(avg_win_ret))
    c7.metric("Win Rate", "—" if pd.isna(win_rate) else f"{win_rate:.0%}")

    res["ret_pct"] = res["ret_pct"].map(lambda x: pct_str(x))
    res["entry"] = res["entry"].map(money_str)
    res["exit_or_last"] = res["exit_or_last"].map(money_str)
    res["capital_after"] = res["capital_after"].map(money_str)
    res = date_only_cols(res, ["entry_date","exit_date"])

    st.dataframe(add_rownum(res), use_container_width=True, hide_index=True)

# ---------- COMPARE ----------
if page == "Compare":
    st.subheader("🆚 GARCH Threshold A/B Compare (manual)")

    with st.spinner("⏳ Preparing trades and GARCH index..."):
        trades_base = run_strategy(df)
        if trades_base.empty:
            st.info("No trades detected.")
            st.stop()
        trades_all_cmp = attach_garch_risk_index(df, trades_base, base_vol_dec=0.20, lookback=252)

        latest_prices = df.groupby("symbol", as_index=False).agg(latest_close=("close", "last"))
        trades_all_cmp = trades_all_cmp.merge(latest_prices, on="symbol", how="left")

    start_date = st.sidebar.date_input("Start date", value=trades_all_cmp["entry_date"].min().date())
    end_date = st.sidebar.date_input("End date", value=trades_all_cmp["entry_date"].max().date())
    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)

    tickers_in_window = trades_all_cmp[
        (trades_all_cmp["entry_date"] >= start_ts) &
        (trades_all_cmp["entry_date"] <= end_ts)
    ]["symbol"].unique().tolist()
    selected_tickers = st.sidebar.multiselect(
        "Tickers (within window)", options=sorted(tickers_in_window), default=tickers_in_window
    )

    starting_capital = st.sidebar.number_input("Starting Capital ($)", min_value=1000.0, step=100.0, value=10000.0)
    alloc_pct = st.sidebar.number_input("Allocation per trade (%)", min_value=1.0, max_value=100.0, step=1.0, value=100.0)

    cA, cB = st.columns(2)
    thrA = cA.number_input("Threshold A", min_value=0.0, max_value=1.0, step=0.05, value=0.00)
    thrB = cB.number_input("Threshold B", min_value=0.0, max_value=1.0, step=0.05, value=0.50)

    candidates = trades_all_cmp[
        (trades_all_cmp["entry_date"] >= start_ts) &
        (trades_all_cmp["entry_date"] <= end_ts) &
        (trades_all_cmp["symbol"].isin(selected_tickers))
    ].copy()

    if candidates.empty:
        st.info("No trades match the selected window and tickers.")
        st.stop()

    metricsA, resA = simulate_oaat(
        candidates, start_ts, end_ts,
        starting_capital=starting_capital, alloc_pct=alloc_pct,
        threshold=thrA, require_garch=True
    )
    metricsB, resB = simulate_oaat(
        candidates, start_ts, end_ts,
        starting_capital=starting_capital, alloc_pct=alloc_pct,
        threshold=thrB, require_garch=True
    )

    st.write("### Results")
    colA, colB, colΔ = st.columns(3)
    colA.metric("Final capital (A)", money_str(metricsA["final_capital"]))
    colB.metric("Final capital (B)", money_str(metricsB["final_capital"]),
                delta=pct_str((metricsB["final_capital"]/metricsA["final_capital"]-1)*100))
    colΔ.metric("Δ Total return (B−A)", pct_str(metricsB["total_return"]-metricsA["total_return"]))

    c1, c2, c3 = st.columns(3)
    c1.metric("Total return A", pct_str(metricsA["total_return"]))
    c2.metric("Total return B", pct_str(metricsB["total_return"]))
    c3.metric("Δ trades (B−A)", f"{metricsB['trades']-metricsA['trades']}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Win Rate A", "—" if pd.isna(metricsA["win_rate"]) else f"{metricsA['win_rate']:.0%}")
    c5.metric("Win Rate B", "—" if pd.isna(metricsB["win_rate"]) else f"{metricsB['win_rate']:.0%}")
    c6.metric("Δ Win Rate", "—" if (pd.isna(metricsA['win_rate']) or pd.isna(metricsB['win_rate']))
              else f"{(metricsB['win_rate']-metricsA['win_rate']):.0%}")

    with st.expander("Show taken trades (A)"):
        tmp = resA.copy()
        tmp["ret_pct"] = tmp["ret_pct"].map(lambda x: f"{x:+.2f}%")
        tmp = date_only_cols(tmp, ["entry_date","exit_date"])
        st.dataframe(add_rownum(tmp), use_container_width=True, hide_index=True)

    with st.expander("Show taken trades (B)"):
        tmp = resB.copy()
        tmp["ret_pct"] = tmp["ret_pct"].map(lambda x: f"{x:+.2f}%")
        tmp = date_only_cols(tmp, ["entry_date","exit_date"])
        st.dataframe(add_rownum(tmp), use_container_width=True, hide_index=True)

    st.write("### Threshold sweep")
    step = st.slider("Step", 0.01, 0.25, 0.05, 0.01)
    thr_values = np.round(np.arange(0.0, 1.0 + 1e-9, step), 2)
    rows = []
    for thr in thr_values:
        m, _ = simulate_oaat(
            candidates, start_ts, end_ts,
            starting_capital=starting_capital, alloc_pct=alloc_pct,
            threshold=thr, require_garch=True
        )
        rows.append({"threshold": thr, "trades": m["trades"], "total_return_%": m["total_return"],
                     "win_rate_%": (m["win_rate"]*100 if pd.notna(m["win_rate"]) else np.nan)})
    sweep_df = pd.DataFrame(rows)
    st.dataframe(add_rownum(sweep_df), use_container_width=True, hide_index=True)