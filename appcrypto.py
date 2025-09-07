# app_crypto.py — Crypto-only Smart Backtester (safe sort + no None)
import os, time
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from arch import arch_model  # for GARCH
from strategy import run_strategy

st.set_page_config(page_title="Smart Backtester — Crypto", layout="wide")
st.title("🪙 Smart Backtester — Crypto")

AUTO_MIN_TRADES = 3
AUTO_GRID_STEP  = 0.05

# ---------------------------- Helpers ----------------------------
def add_rownum(df_in):
    df = df_in.copy()
    df.insert(0, "#", range(1, len(df) + 1))
    return df

def pct_str(x, digits=2, signed=True):
    if pd.isna(x): return "—"
    return f"{x:+.{digits}f}%" if signed else f"{x:.{digits}f}%"

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
    try:
        s = os.stat(path); return f"{s.st_mtime_ns}-{s.st_size}"
    except FileNotFoundError:
        return f"missing-{time.time()}"

def ensure_cols(df: pd.DataFrame, specs: dict):
    """Ensure columns exist with given default values (np.nan for numbers, NaT for dates)."""
    out = df.copy()
    for col, default in specs.items():
        if col not in out.columns:
            out[col] = default
    return out

# ---------------------------- GARCH ----------------------------
def garch_volatility_forecast(series: pd.Series) -> float | None:
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

@st.cache_data(show_spinner=False)
def attach_garch_risk_index(df_prices: pd.DataFrame,
                            trades_df: pd.DataFrame,
                            base_vol_dec: float = 0.20,
                            lookback: int | None = 252) -> pd.DataFrame:
    df_sorted = df_prices.sort_values(["symbol", "date"])
    groups = {sym: g[["date","close"]].reset_index(drop=True)
              for sym, g in df_sorted.groupby("symbol")}
    out = []
    for _, r in trades_df.iterrows():
        sym = r["symbol"]
        entry_date = pd.to_datetime(r["entry_date"])
        if sym not in groups:
            out.append(np.nan); continue
        hist = groups[sym]
        hist = hist[hist["date"] < entry_date]["close"]
        if lookback and lookback > 0:
            hist = hist.tail(lookback)
        if len(hist) < 51:
            out.append(np.nan); continue
        ann_vol = garch_volatility_forecast(hist)
        if ann_vol is None or not np.isfinite(ann_vol) or ann_vol <= 1e-9:
            out.append(np.nan); continue
        out.append(float(min(1.0, base_vol_dec / ann_vol)))
    t2 = trades_df.copy()
    t2["garch_risk_index"] = out
    return t2

# ---------------------------- OAAT simulator ----------------------------
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
            exit_date = ex_d; status = "Realized"
        else:
            exit_px = float(r["latest_close"])
            ret_pct = (exit_px / entry - 1.0) * 100.0
            available_from = end_ts + pd.Timedelta(days=1)
            exit_date = pd.NaT; status = "Unrealized"

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
        res = pd.DataFrame(columns=["symbol","entry_date","exit_date","entry","exit_or_last","ret_pct","status","capital_after"])
        metrics = dict(final_capital=starting_capital, total_return=0.0, trades=0,
                       win_rate=float("nan"), avg_win_ret=float("nan"),
                       max_win=float("nan"), max_loss=float("nan"))
        return metrics, res

    res = pd.DataFrame(taken).sort_values("entry_date").reset_index(drop=True)
    n_trades = len(res); n_real = (res["status"] == "Realized").sum()
    win_rate = (res.loc[res["status"]=="Realized","ret_pct"] > 0).mean() if n_real else float("nan")
    avg_win_ret = res.loc[(res["status"]=="Realized") & (res["ret_pct"]>0), "ret_pct"].mean()
    max_win = res["ret_pct"].max(); max_loss = res["ret_pct"].min()
    total_ret = (res["capital_after"].iloc[-1] / starting_capital - 1.0) * 100.0

    metrics = dict(final_capital=res["capital_after"].iloc[-1],
                   total_return=total_ret, trades=n_trades, win_rate=win_rate,
                   avg_win_ret=avg_win_ret, max_win=max_win, max_loss=max_loss)
    return metrics, res

# ---------------------------- Load crypto prices (FILTER FIRST) ----------------------------
@st.cache_data(show_spinner=False)
def load_raw_prices(_sig: str) -> pd.DataFrame:
    df = pd.read_csv("stocks.csv", parse_dates=["date"])

    # Ensure expected columns exist
    if "asset_type" not in df.columns:
        df["asset_type"] = "stock"
    if "sector" not in df.columns:
        df["sector"] = None

    # Normalize and filter to crypto
    df["asset_type"] = df["asset_type"].astype(str).str.lower()
    df = df[df["asset_type"] == "crypto"].copy()

    # Tidy sector to avoid 'None' in UI
    df["sector"] = df["sector"].astype("object")
    df.loc[df["sector"].isna() | (df["sector"].astype(str).str.strip() == ""), "sector"] = "Crypto"

    # Clean dupes & sort (harmless)
    df = (df
          .drop_duplicates(subset=["symbol","date"])
          .sort_values(["symbol","date"])
          .reset_index(drop=True))
    return df

# ---------------------------- Sidebar / reload ----------------------------
st.sidebar.header("Data")
st.sidebar.caption("Reading crypto rows from stocks.csv")
stocks_sig_now = _file_sig("stocks.csv")

# Cache bust on file change
if st.session_state.get("_stocks_sig") != stocks_sig_now:
    st.session_state["_stocks_sig"] = stocks_sig_now
    try: st.cache_data.clear()
    except Exception: pass

df0 = load_raw_prices(stocks_sig_now)

if st.sidebar.button("🔁 Reload file"):
    try: st.cache_data.clear()
    except Exception: pass
    st.rerun()

# Universe = all crypto symbols
crypto_symbols = sorted(df0["symbol"].unique().tolist())
df = df0[df0["symbol"].isin(crypto_symbols)].copy().sort_values(["symbol","date"])

# ---------------------------- Strategy → Trades ----------------------------
with st.spinner("⏳ Detecting trades..."):
    trades_raw = run_strategy(df)
if trades_raw.empty:
    st.warning("No trades detected."); st.stop()

with st.spinner("🔎 Computing GARCH risk index..."):
    trades_all = attach_garch_risk_index(df, trades_raw, base_vol_dec=0.20, lookback=252)

# Ensure perf columns always exist (use NaN, not None → UI shows '—')
for col in ("avg_return", "avg_win_return", "win_rate"):
    if col not in trades_all.columns:
        trades_all[col] = np.nan

with st.spinner("🧠 Learning per-ticker thresholds from CLOSED trades..."):
    # Learn on CLOSED trades only
    closed = trades_all[trades_all["exit_date"].notna()].copy()
    if closed.empty:
        thr_map = {s: 0.00 for s in sorted(trades_all["symbol"].unique())}
        sweep_long = pd.DataFrame(columns=["symbol","threshold","trades","total_return_%","win_rate_%"])
    else:
        # Simple grid search using OAAT on closed window
        start_ts = closed["entry_date"].min()
        end_ts   = closed["exit_date"].max()
        thr_values = np.round(np.arange(0.0, 1.0 + 1e-9, AUTO_GRID_STEP), 2)
        rows, best_map = [], {}
        for sym, cand in closed.groupby("symbol"):
            best = None
            for thr in thr_values:
                m, _ = simulate_oaat(cand, start_ts, end_ts,
                                     starting_capital=10000.0, alloc_pct=100.0,
                                     threshold=thr, require_garch=True)
                tr = m["trades"]; score = m["total_return"]
                rows.append({"symbol": sym, "threshold": thr, "trades": tr,
                             "total_return_%": score,
                             "win_rate_%": (m["win_rate"]*100 if pd.notna(m["win_rate"]) else np.nan)})
                if tr < AUTO_MIN_TRADES:  # enforce minimum trades
                    continue
                candidate = (score, tr, -thr, thr)
                if (best is None) or (candidate > best):
                    best = candidate
            best_map[sym] = (best[3] if best is not None else 0.00)
        thr_map = best_map
        sweep_long = pd.DataFrame(rows)

def _accept_row(r):
    thr = thr_map.get(r["symbol"], 0.00)
    return pd.notna(r["garch_risk_index"]) and (r["garch_risk_index"] >= thr)

trades = trades_all[trades_all.apply(_accept_row, axis=1)].copy()
if trades.empty:
    st.warning("All trades filtered by auto thresholds."); st.stop()

# ---------------------------- Enrich / postprocess ----------------------------
sector_map = df[["symbol","sector"]].drop_duplicates().set_index("symbol")["sector"]
trades["sector"] = trades["symbol"].map(sector_map)
trades["symbol_display"] = trades["symbol"]

# Stop loss from yesterday's low
df["stop_loss"] = df.groupby("symbol")["low"].shift(1)
entry_lows = df[["symbol","date","stop_loss"]].rename(columns={"date":"entry_date"})
trades = trades.merge(entry_lows, on=["symbol","entry_date"], how="left")

# Final % return
latest_prices = df.groupby("symbol", as_index=False).agg(latest_close=("close","last"))
trades = trades.merge(latest_prices, on="symbol", how="left")
trades["pct_return"] = (trades["exit_price"]/trades["entry"] - 1)*100
trades["unrealized_pct_return"] = (trades["latest_close"]/trades["entry"] - 1)*100
trades["final_pct"] = trades.apply(
    lambda r: r["pct_return"] if pd.notna(r["exit_price"]) else r["unrealized_pct_return"], axis=1
)

# Min/Max since entry
def add_minmax_for_trades(df_prices: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    prices = df_prices[["symbol","date","low","high"]].copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices = prices.dropna(subset=["date"])
    last_date_by_sym = prices.groupby("symbol")["date"].max()
    rows = []
    for _, tr in trades_df.iterrows():
        sym = tr["symbol"]
        start = pd.to_datetime(tr["entry_date"], errors="coerce")
        end = (pd.to_datetime(tr["exit_date"], errors="coerce")
               if pd.notna(tr["exit_date"]) else last_date_by_sym.get(sym, pd.NaT))
        if pd.isna(start) or pd.isna(end): continue
        sl = prices[(prices["symbol"]==sym) & (prices["date"]>=start) & (prices["date"]<=end)]
        if sl.empty: continue
        rows.append({"symbol": sym, "entry_date": start,
                     "min_low": float(sl["low"].min()), "max_high": float(sl["high"].max())})
    mm = pd.DataFrame(rows)
    return trades_df.merge(mm, on=["symbol","entry_date"], how="left")

trades = add_minmax_for_trades(df, trades)

# Ensure any display columns are present and numeric columns are NaN (not None)
trades = ensure_cols(trades, {
    "avg_return": np.nan, "avg_win_return": np.nan, "win_rate": np.nan,
    "entry": np.nan, "stop_loss": np.nan, "unrealized_pct_return": np.nan
})

# ---------------------------- Navigation ----------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose page", ["Home","Insights","Tester","Compare"], index=0)

# ---------------------------- HOME ----------------------------
if page == "Home":
    st.subheader("🔓 Open Trades")
    open_trades = trades[trades["exit_date"].isna()].copy()

    # Guarantee required columns exist (avoid KeyError on sort)
    open_trades = ensure_cols(open_trades, {"entry_date": pd.NaT, "avg_return": np.nan})

    # SAFE sort: only use columns that actually exist
    sort_cols = [c for c in ["entry_date","avg_return"] if c in open_trades.columns]
    if sort_cols:
        open_trades = open_trades.sort_values(by=sort_cols,
                                              ascending=[False]*len(sort_cols),
                                              na_position="last")

    if open_trades.empty:
        st.info("No open trades.")
    else:
        unrealized_total = open_trades["unrealized_pct_return"].mean()
        ed = pd.to_datetime(open_trades["entry_date"], errors="coerce")
        earliest_entry = ed.min()
        earliest_entry_str = "—" if pd.isna(earliest_entry) else earliest_entry.strftime("%Y-%m-%d")

        # Holding days (safe)
        avg_hold_days = (pd.Timestamp.today().normalize() - ed).dt.days
        avg_hold_val = float(avg_hold_days.mean()) if avg_hold_days.notna().any() else float("nan")

        # Best/worst (safe)
        if open_trades["unrealized_pct_return"].notna().any():
            best_row = open_trades.loc[open_trades["unrealized_pct_return"].idxmax()]
            worst_row = open_trades.loc[open_trades["unrealized_pct_return"].idxmin()]
            best_symbol  = f"{best_row['symbol']} ({pct_str(best_row['unrealized_pct_return'])}) ✅"
            worst_symbol = f"{worst_row['symbol']} ({pct_str(worst_row['unrealized_pct_return'])}) ❌"
        else:
            best_symbol = worst_symbol = "—"

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Unrealized P/L", pct_str(unrealized_total))
        c2.metric("Open Trades", f"{len(open_trades)}")
        c3.metric("Earliest Entry", earliest_entry_str)

        c4, c5, c6 = st.columns(3)
        c4.metric("Avg Holding (days)", "—" if pd.isna(avg_hold_val) else f"{avg_hold_val:.1f}")
        c5.metric("Best Performer", best_symbol)
        c6.metric("Worst Performer", worst_symbol)

        # Display table (ensure columns exist)
        needed = ["symbol_display","sector","entry_date","entry","stop_loss",
                  "avg_return","avg_win_return","win_rate","unrealized_pct_return"]
        open_trades = ensure_cols(open_trades, {c: np.nan for c in needed})

        def emoji_unrealized(x):
            if pd.isna(x): return "—"
            return f"✅ {pct_str(x)}" if x > 0 else f"❌ {pct_str(x)}"

        table = open_trades[needed].copy()
        table = date_only_cols(table, ["entry_date"])
        table["entry"] = table["entry"].map(money_str)
        table["stop_loss"] = table["stop_loss"].map(money_str)
        table["avg_return"] = table["avg_return"].map(lambda x: pct_str(x))
        table["avg_win_return"] = table["avg_win_return"].map(lambda x: pct_str(x))
        table["win_rate"] = table["win_rate"].map(lambda x: "—" if pd.isna(x) else f"{x:.0%}")
        table["unrealized_pct_return"] = table["unrealized_pct_return"].map(emoji_unrealized)

        table = table.rename(columns={
            "symbol_display":"Symbol","sector":"Sector","entry_date":"Entry Date",
            "entry":"Entry","stop_loss":"Stop Loss","avg_return":"Avg Return",
            "avg_win_return":"Avg Win","win_rate":"Win Rate","unrealized_pct_return":"Unrealized",
        })
        st.dataframe(add_rownum(table), use_container_width=True, hide_index=True)

        # -------- Alarm list: 1-day drop alert --------
        df_tmp = df[["symbol","date","close"]].copy().sort_values(["symbol","date"])
        df_tmp["prev_close"] = df_tmp.groupby("symbol")["close"].shift(1)
        df_tmp["day_change_pct"] = (df_tmp["close"]/df_tmp["prev_close"] - 1.0) * 100.0

        latest_per_sym = df_tmp.groupby("symbol", as_index=False).tail(1)[["symbol","day_change_pct"]]
        alarms = (open_trades[["symbol","symbol_display","sector","entry_date","entry","latest_close","unrealized_pct_return"]]
                  .merge(latest_per_sym, on="symbol", how="left"))

        threshold = -float(globals().get("ALARM_DAILY_DROP_PCT", 2.0))  # e.g., -2%
        alarms = alarms[alarms["day_change_pct"].le(threshold)].copy()

        st.subheader(f"🚨 Alarm List — 1-Day Drop ≥ {globals().get('ALARM_DAILY_DROP_PCT', 2.0):.0f}%")
        if not alarms.empty:
            alarms = alarms.sort_values("day_change_pct")
            alarms_fmt = alarms.copy()
            alarms_fmt = date_only_cols(alarms_fmt, ["entry_date"])
            alarms_fmt["P&L Now"] = alarms_fmt["unrealized_pct_return"].map(lambda x: pct_str(x))
            alarms_fmt["1D Change"] = alarms_fmt["day_change_pct"].map(lambda x: pct_str(x))
            alarms_fmt["Entry"] = alarms_fmt["entry"].map(money_str)
            alarms_fmt = alarms_fmt.rename(columns={"symbol_display":"Symbol","sector":"Sector","entry_date":"Entry Date"})
            alarms_fmt = alarms_fmt[["Symbol","Sector","Entry Date","Entry","1D Change","P&L Now"]]
            st.dataframe(add_rownum(alarms_fmt), use_container_width=True, hide_index=True)
        else:
            st.info("No alerts right now.")

# ---------------------------- INSIGHTS ----------------------------
if page == "Insights":
    st.subheader("📊 Closed Trades Insights")
    closed = trades[trades["exit_date"].notna()].copy()
    if closed.empty:
        st.info("No closed trades to analyze.")
    else:
        closed["pct_return"] = (closed["exit_price"]/closed["entry"] - 1) * 100
        closed["win"] = closed["pct_return"] > 0
        closed["days_held"] = (closed["exit_date"] - closed["entry_date"]).dt.days

        agg = closed.groupby("symbol").agg(
            n_trades=("pct_return","size"),
            avg_return=("pct_return","mean"),
            avg_days=("days_held","mean"),
            win_rate=("win","mean"),
        ).reset_index()

        def total_ret_percent(s: pd.Series) -> float:
            return (np.prod(1.0 + s.values/100.0) - 1.0) * 100.0

        tot = (closed.groupby("symbol")["pct_return"]
               .apply(total_ret_percent)
               .rename("total_return_%").reset_index())

        win_mean  = (closed[closed["pct_return"]>0].groupby("symbol")["pct_return"].mean().rename("avg_win_return"))
        loss_mean = (closed[closed["pct_return"]<0].groupby("symbol")["pct_return"].mean().rename("avg_loss_return"))

        best = (agg.merge(win_mean, on="symbol", how="left")
                    .merge(loss_mean, on="symbol", how="left")
                    .merge(tot, on="symbol", how="left"))

        best["sector"] = best["symbol"].map(sector_map)
        best["symbol_display"] = best["symbol"]
        best = best.sort_values("total_return_%", ascending=False)

        disp = best.copy()
        disp["win_rate_str"]        = disp["win_rate"].map(lambda x: "—" if pd.isna(x) else f"{x:.0%}")
        disp["total_return_str"]    = disp["total_return_%"].map(lambda x: pct_str(x))
        disp["avg_return_str"]      = disp["avg_return"].map(lambda x: pct_str(x))
        disp["avg_win_return_str"]  = disp["avg_win_return"].map(lambda x: pct_str(x))
        disp["avg_loss_return_str"] = disp["avg_loss_return"].map(lambda x: pct_str(x))
        disp["avg_days_str"]        = disp["avg_days"].map(lambda x: "—" if pd.isna(x) else f"{x:.1f}")

        show_df = add_rownum(disp[[
            "symbol_display","sector","n_trades",
            "win_rate_str","total_return_str","avg_return_str",
            "avg_win_return_str","avg_loss_return_str","avg_days_str"
        ]])

        st.dataframe(show_df.rename(columns={
            "symbol_display":"Symbol","sector":"Sector","n_trades":"Trades",
            "win_rate_str":"Win Rate","total_return_str":"Total Return",
            "avg_return_str":"Avg/Trade","avg_win_return_str":"Avg Win",
            "avg_loss_return_str":"Avg Loss","avg_days_str":"Avg Days"
        }), use_container_width=True, hide_index=True)

        st.subheader("📦 All Trades (Open + Closed)")
        all_trades = trades.copy().sort_values("entry_date", ascending=False)
        all_trades = date_only_cols(all_trades, ["entry_date","exit_date"])
        display = all_trades[[
            "symbol_display","sector","entry_date","exit_date",
            "entry","exit_price","final_pct","stop_loss","min_low","max_high"
        ]].copy()
        display["entry"] = display["entry"].map(money_str)
        display["exit_price"] = display["exit_price"].map(money_str)
        display["final_pct"] = display["final_pct"].map(lambda x: pct_str(x))
        st.dataframe(add_rownum(display), use_container_width=True, hide_index=True)
        st.download_button(
            label="📥 Download All Trades",
            data=display.to_csv(index=False).encode("utf-8"),
            file_name="all_trades_crypto.csv", mime="text/csv"
        )

# ---------------------------- TESTER ----------------------------
if page == "Tester":
    st.subheader("🧪 One-Position-At-A-Time Backtest")
    start_date = st.sidebar.date_input("Start date", value=trades["entry_date"].min().date())
    end_date = st.sidebar.date_input("End date", value=trades["entry_date"].max().date())
    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)

    tickers_in_window = trades[(trades["entry_date"]>=start_ts) & (trades["entry_date"]<=end_ts)]["symbol"].unique().tolist()
    selected_tickers = st.sidebar.multiselect("Optional ticker filter (within window)",
                                              options=sorted(tickers_in_window), default=tickers_in_window)
    starting_capital = st.sidebar.number_input("Starting Capital ($)", min_value=1000.0, step=100.0, value=10000.0)
    alloc_pct = st.sidebar.number_input("Allocation per trade (%)", min_value=1.0, max_value=100.0, step=1.0, value=100.0)

    candidates = trades[(trades["entry_date"]>=start_ts) & (trades["entry_date"]<=end_ts) &
                        (trades["symbol"].isin(selected_tickers))].copy().sort_values("entry_date")
    if candidates.empty:
        st.info("No trades match the selected window and tickers."); st.stop()

    capital = float(starting_capital); available_from = start_ts; ledger = []
    for _, r in candidates.iterrows():
        entry_d = pd.to_datetime(r["entry_date"])
        if entry_d < available_from: continue
        sym = r["symbol"]; entry = float(r["entry"])
        ex_d = pd.to_datetime(r["exit_date"]) if pd.notna(r["exit_date"]) else pd.NaT
        realized = pd.notna(ex_d) and (ex_d <= end_ts)
        if realized:
            exit_px = float(r["exit_price"]); ret_pct = (exit_px/entry - 1.0)*100.0
            exit_d = ex_d; status = "Realized"; available_from = ex_d + pd.Timedelta(days=1)
        else:
            exit_px = float(r["latest_close"]); ret_pct = (exit_px/entry - 1.0)*100.0
            exit_d = pd.NaT; status = "Unrealized"; available_from = end_ts + pd.Timedelta(days=1)
        invest_amt = capital * (alloc_pct/100.0)
        capital = capital - invest_amt + invest_amt*(1.0 + ret_pct/100.0)
        ledger.append({"symbol": sym, "entry_date": entry_d, "exit_date": exit_d,
                       "entry": entry, "exit_or_last": exit_px, "ret_pct": ret_pct,
                       "status": status, "capital_after": capital})

    res = pd.DataFrame(ledger).sort_values("entry_date").reset_index(drop=True)
    n_trades = len(res); n_real = (res["status"]=="Realized").sum()
    win_rate = (res.loc[res["status"]=="Realized","ret_pct"]>0).mean() if n_real else float("nan")
    avg_win_ret = res.loc[(res["status"]=="Realized") & (res["ret_pct"]>0),"ret_pct"].mean()
    max_win = res["ret_pct"].max(); max_loss = res["ret_pct"].min()
    total_ret = (capital/starting_capital - 1.0)*100.0

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

# ---------------------------- COMPARE ----------------------------
if page == "Compare":
    st.subheader("🆚 GARCH Threshold A/B Compare (manual)")
    with st.spinner("⏳ Preparing trades and GARCH index..."):
        trades_base = run_strategy(df)
        if trades_base.empty:
            st.info("No trades detected."); st.stop()
        trades_all_cmp = attach_garch_risk_index(df, trades_base, base_vol_dec=0.20, lookback=252)
        latest_prices = df.groupby("symbol", as_index=False).agg(latest_close=("close","last"))
        trades_all_cmp = trades_all_cmp.merge(latest_prices, on="symbol", how="left")

    start_date = st.sidebar.date_input("Start date", value=trades_all_cmp["entry_date"].min().date())
    end_date   = st.sidebar.date_input("End date",   value=trades_all_cmp["entry_date"].max().date())
    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)

    tickers_in_window = trades_all_cmp[(trades_all_cmp["entry_date"]>=start_ts) &
                                       (trades_all_cmp["entry_date"]<=end_ts)]["symbol"].unique().tolist()
    selected_tickers = st.sidebar.multiselect("Tickers (within window)",
                                              options=sorted(tickers_in_window),
                                              default=tickers_in_window)

    starting_capital = st.sidebar.number_input("Starting Capital ($)", min_value=1000.0, step=100.0, value=10000.0)
    alloc_pct = st.sidebar.number_input("Allocation per trade (%)", min_value=1.0, max_value=100.0, step=1.0, value=100.0)

    cA, cB = st.columns(2)
    thrA = cA.number_input("Threshold A", min_value=0.0, max_value=1.0, step=0.05, value=0.00)
    thrB = cB.number_input("Threshold B", min_value=0.0, max_value=1.0, step=0.05, value=0.50)

    candidates = trades_all_cmp[(trades_all_cmp["entry_date"]>=start_ts) &
                                (trades_all_cmp["entry_date"]<=end_ts) &
                                (trades_all_cmp["symbol"].isin(selected_tickers))].copy()
    if candidates.empty:
        st.info("No trades match the selected window and tickers."); st.stop()

    metricsA, resA = simulate_oaat(candidates, start_ts, end_ts,
                                   starting_capital=starting_capital, alloc_pct=alloc_pct,
                                   threshold=thrA, require_garch=True)
    metricsB, resB = simulate_oaat(candidates, start_ts, end_ts,
                                   starting_capital=starting_capital, alloc_pct=alloc_pct,
                                   threshold=thrB, require_garch=True)

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
        m, _ = simulate_oaat(candidates, start_ts, end_ts,
                             starting_capital=starting_capital, alloc_pct=alloc_pct,
                             threshold=thr, require_garch=True)
        rows.append({"threshold": thr, "trades": m["trades"], "total_return_%": m["total_return"],
                     "win_rate_%": (m["win_rate"]*100 if pd.notna(m["win_rate"]) else np.nan)})
    sweep_df = pd.DataFrame(rows)
    st.dataframe(add_rownum(sweep_df), use_container_width=True, hide_index=True)
