# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from strategy import run_strategy
from arch import arch_model  # NEW: for GARCH

st.set_page_config(page_title="Smart Backtester", layout="wide")
st.title("📈 Smart Backtester")

# ---------- Helpers ----------
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

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("stocks.csv", parse_dates=["date"])
    caps = pd.read_csv("market_cap.csv")
    top_symbols = (
        caps[~caps["cap_score"].isin([3, 4])]
        .sort_values("cap_score")  # low cap_score = higher quality
        .head(100)["symbol"]
        .tolist()
    )
    df = df[df["symbol"].isin(top_symbols)].copy()
    caps = caps[caps["symbol"].isin(top_symbols)].copy()
    return df.sort_values(["symbol", "date"]), caps

# ---------- GARCH: stable risk index ----------
def garch_volatility_forecast(series: pd.Series) -> float | None:
    """
    GARCH(1,1) on decimal log-returns (mean='Zero', dist='t').
    Returns 1-day-ahead **annualized** vol in DECIMAL units (e.g., 0.22 = 22%).
    """
    px = pd.Series(series).astype(float)
    px = px.replace([np.inf, -np.inf], np.nan).dropna()
    px = px[px > 0]
    rets = np.log(px / px.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(rets) < 50:
        return None
    try:
        am = arch_model(rets, vol="GARCH", p=1, q=1, mean="Zero", dist="t")  # rescale=True by default
        res = am.fit(disp="off")
        fc = res.forecast(horizon=1, reindex=False)
        var1 = float(fc.variance.iloc[-1, 0])
        ann_vol = float(np.sqrt(var1) * np.sqrt(252))  # decimal
        return ann_vol if np.isfinite(ann_vol) and ann_vol > 0 else None
    except Exception:
        return None
    
# --- OAAT simulator that accepts a GARCH hard filter threshold ---
def simulate_oaat(
    trades_in: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    starting_capital: float = 10000.0,
    alloc_pct: float = 100.0,
    *,
    threshold: float = 0.50,       # hard filter: keep only trades with garch_risk_index >= threshold
    require_garch: bool = True     # if True, drop NaN risk; if False, treat NaN as 1.0
):
    t = trades_in.copy().sort_values("entry_date")
    if require_garch:
        t = t[t["garch_risk_index"].notna()]
    else:
        # if not requiring garch, treat NaN as 1.0 for the filter
        t["garch_risk_index"] = t["garch_risk_index"].fillna(1.0)

    # hard filter by threshold
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
            # needs a latest_close column upstream (you already add it)
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

    # ----- handle “no trades taken” safely -----
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
    # -------------------------------------------

    res = pd.DataFrame(taken).sort_values("entry_date").reset_index(drop=True)

    # KPIs
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



@st.cache_data(show_spinner=False)
def attach_garch_risk_index(df_prices: pd.DataFrame,
                            trades_df: pd.DataFrame,
                            base_vol_dec: float = 0.20,
                            lookback: int | None = 252) -> pd.DataFrame:
    """
    Adds one column per trade:
      • garch_risk_index ∈ (0,1], where LOWER = riskier; computed as min(1, base_vol/ann_vol).
    Uses price history strictly before each entry_date, optionally capped by lookback.
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

# ---------- Sidebar ----------
st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose page", ["Home", "Insights", "Tester", "Compare"], index=0)
near_band_pp = st.sidebar.number_input("Zone band (± %)", min_value=0.1, max_value=10.0, step=0.1, value=1.0)

# GARCH filter threshold (default 0.50 as requested)
risk_cutoff = st.sidebar.slider("Min GARCH risk index (lower=riskier)", 0.0, 1.0, 0.50, 0.05)

# ---------- Load and Prepare ----------
df, caps = load_data()
cap_score_map = caps.set_index("symbol")["cap_score"]
cap_emoji_map = caps.set_index("symbol")["cap_emoji"]
symbols_to_keep = cap_score_map[~cap_score_map.isin([3, 4])].index.tolist()
df = df[df["symbol"].isin(symbols_to_keep)].copy()

# ---------- Run Strategy ----------
with st.spinner("⏳ Detecting trades..."):
    trades = run_strategy(df)

if trades.empty:
    st.warning("No trades detected.")
    st.stop()

# ---------- NEW: attach GARCH & FILTER (< 0.50 removed) ----------
with st.spinner("🔎 Computing GARCH risk index and filtering..."):
    trades = attach_garch_risk_index(df, trades, base_vol_dec=0.20, lookback=252)
    before_n = len(trades)
    trades = trades[trades["garch_risk_index"].ge(risk_cutoff)].copy()
    after_n = len(trades)

st.caption(f"Applied GARCH filter: kept {after_n}/{before_n} trades with risk index ≥ {risk_cutoff:.2f}.")

if trades.empty:
    st.warning("All trades were filtered out by the GARCH threshold. Try lowering the threshold in the sidebar.")
    st.stop()

# ---------- Postprocess ----------
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
minmax = []
for _, r in trades[trades["exit_date"].isna()].iterrows():
    sym, entry_date = r["symbol"], r["entry_date"]
    sl = df[(df["symbol"] == sym) & (df["date"] >= entry_date)]
    if not sl.empty:
        minmax.append((sym, entry_date, sl["low"].min(), sl["high"].max()))
minmax_df = pd.DataFrame(minmax, columns=["symbol", "entry_date", "min_low", "max_high"]) if minmax else pd.DataFrame(columns=["symbol","entry_date","min_low","max_high"])
trades = trades.merge(minmax_df, on=["symbol", "entry_date"], how="left")

# Closed perf metrics
closed = trades[trades["exit_date"].notna()].copy()
if not closed.empty:
    closed["pct_return"] = (closed["exit_price"] / closed["entry"] - 1) * 100
    win_mean = closed[closed["pct_return"] > 0].groupby("symbol")["pct_return"].mean().rename("avg_win_return")
    avg_return = closed.groupby("symbol")["pct_return"].mean().rename("avg_return")
    trades = trades.merge(avg_return, on="symbol", how="left").merge(win_mean, on="symbol", how="left")
else:
    trades["avg_return"] = None
    trades["avg_win_return"] = None

# ---------- HOME ----------
if page == "Home":
    st.subheader("🔓 Open Trades")

    open_trades = trades[trades["exit_date"].isna()].copy()
    if open_trades.empty:
        st.info("No open trades.")
    else:
        # KPIs
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

        # Display table
        open_trades = open_trades.sort_values(["entry_date", "avg_return"], ascending=[False, False])

        def emoji_unrealized(x):
            if pd.isna(x): return "—"
            return f"✅ {pct_str(x)}" if x > 0 else f"❌ {pct_str(x)}"

        table = open_trades[[
            "symbol_display", "sector", "entry_date", "entry", "stop_loss",
            "avg_return", "avg_win_return", "unrealized_pct_return"
        ]].copy()

        table = date_only_cols(table, ["entry_date"])
        table["entry"] = table["entry"].map(money_str)
        table["stop_loss"] = table["stop_loss"].map(money_str)
        table["avg_return"] = table["avg_return"].map(lambda x: pct_str(x))
        table["avg_win_return"] = table["avg_win_return"].map(lambda x: pct_str(x))
        table["unrealized_pct_return"] = table["unrealized_pct_return"].map(emoji_unrealized)

        table = table.rename(columns={
            "symbol_display": "Symbol",
            "sector": "Sector",
            "entry_date": "Entry Date",
            "entry": "Entry",
            "stop_loss": "Stop Loss",
            "avg_return": "Avg Return",
            "avg_win_return": "Avg Win",
            "unrealized_pct_return": "Unrealized"
        })

        st.dataframe(add_rownum(table), use_container_width=True, hide_index=True)

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

        base = closed.groupby("symbol").agg(
            n_trades=("pct_return", "size"),
            avg_return=("pct_return", "mean"),
            avg_days=("days_held", "mean")
        ).reset_index()

        win_mean = closed[closed["pct_return"] > 0].groupby("symbol")["pct_return"].mean().rename("avg_win_return")
        loss_mean = closed[closed["pct_return"] < 0].groupby("symbol")["pct_return"].mean().rename("avg_loss_return")

        best = base.merge(win_mean, on="symbol", how="left").merge(loss_mean, on="symbol", how="left")
        best["sector"] = best["symbol"].map(sector_map)
        best["symbol_display"] = best["symbol"].map(lambda s: f"{cap_emoji_map.get(s,'')} {s}")
        best = best.sort_values("avg_return", ascending=False)

        disp = best.copy()
        disp["avg_return_str"] = disp["avg_return"].map(lambda x: pct_str(x))
        disp["avg_win_return_str"] = disp["avg_win_return"].map(lambda x: pct_str(x))
        disp["avg_loss_return_str"] = disp["avg_loss_return"].map(lambda x: pct_str(x))
        disp["avg_days_str"] = disp["avg_days"].map(lambda x: f"{x:.1f}")
        show_df = add_rownum(disp[[
            "symbol_display","sector","n_trades","avg_return_str",
            "avg_win_return_str","avg_loss_return_str","avg_days_str"
        ]])
        st.dataframe(show_df.rename(columns={
            "symbol_display": "Symbol",
            "n_trades": "Trades",
            "avg_return_str": "Avg Return",
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

    # Optional ticker filter within selected window
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

    # Apply filter BEFORE sim
    candidates = trades[
        (trades["entry_date"] >= start_ts) &
        (trades["entry_date"] <= end_ts) &
        (trades["symbol"].isin(selected_tickers))
    ].copy().sort_values("entry_date")

    if candidates.empty:
        st.info("No trades match the selected window and tickers.")
        st.stop()

    # Simulation
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

    # Summary KPIs
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
    st.subheader("🆚 GARCH Threshold A/B Compare")

    # Recompute raw trades (unfiltered) and attach GARCH once (cached)
    with st.spinner("⏳ Preparing trades and GARCH index..."):
        trades_raw = run_strategy(df)
        if trades_raw.empty:
            st.info("No trades detected.")
            st.stop()
        trades_all = attach_garch_risk_index(df, trades_raw, base_vol_dec=0.20, lookback=252)

        # enrich with latest prices (like elsewhere)
        latest_prices = df.groupby("symbol", as_index=False).agg(latest_close=("close", "last"))
        trades_all = trades_all.merge(latest_prices, on="symbol", how="left")

    # Time window & ticker filter
    start_date = st.sidebar.date_input("Start date", value=trades_all["entry_date"].min().date())
    end_date = st.sidebar.date_input("End date", value=trades_all["entry_date"].max().date())
    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)

    tickers_in_window = trades_all[
        (trades_all["entry_date"] >= start_ts) &
        (trades_all["entry_date"] <= end_ts)
    ]["symbol"].unique().tolist()
    selected_tickers = st.sidebar.multiselect(
        "Tickers (within window)", options=sorted(tickers_in_window), default=tickers_in_window
    )

    starting_capital = st.sidebar.number_input("Starting Capital ($)", min_value=1000.0, step=100.0, value=10000.0)
    alloc_pct = st.sidebar.number_input("Allocation per trade (%)", min_value=1.0, max_value=100.0, step=1.0, value=100.0)

    # Thresholds to compare
    cA, cB = st.columns(2)
    thrA = cA.number_input("Threshold A", min_value=0.0, max_value=1.0, step=0.05, value=0.00)
    thrB = cB.number_input("Threshold B", min_value=0.0, max_value=1.0, step=0.05, value=0.50)

    # Candidate trades (only from selected tickers and window)
    candidates = trades_all[
        (trades_all["entry_date"] >= start_ts) &
        (trades_all["entry_date"] <= end_ts) &
        (trades_all["symbol"].isin(selected_tickers))
    ].copy()

    if candidates.empty:
        st.info("No trades match the selected window and tickers.")
        st.stop()

    # Simulate A and B
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

    # KPIs side-by-side
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

    # Optional: show taken trades for each
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

    # Threshold sweep (quick grid search)
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
