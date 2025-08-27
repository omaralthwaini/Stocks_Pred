import streamlit as st
import pandas as pd
from datetime import timedelta
from strategy import run_strategy

# ---------- Config ----------
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
    
    # Sort caps by descending cap_score (you may adjust this based on your actual scoring logic)
    top_symbols = (
        caps[~caps["cap_score"].isin([3, 4])]
        .sort_values("cap_score")  # lower = higher priority, if that's how your scoring works
        .head(100)["symbol"]
        .tolist()
    )

    # Filter df to keep only those top 100
    df = df[df["symbol"].isin(top_symbols)].copy()
    caps = caps[caps["symbol"].isin(top_symbols)].copy()
    
    return df.sort_values(["symbol", "date"]), caps


# ---------- Sidebar ----------
st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose page", ["Home", "Insights", "Tester"], index=0)

near_band_pp = st.sidebar.number_input(
    "Zone band (± %)", min_value=0.1, max_value=10.0, step=0.1, value=1.0
)

# ---------- Load and Filter Data ----------
df, caps = load_data()
cap_score_map = caps.set_index("symbol")["cap_score"]
cap_emoji_map = caps.set_index("symbol")["cap_emoji"]

# Filter out cap 3 & 4 before running strategy
symbols_to_keep = cap_score_map[~cap_score_map.isin([3, 4])].index.tolist()
df = df[df["symbol"].isin(symbols_to_keep)].copy()

# ---------- Run Strategy ----------
with st.spinner("⏳ Detecting trades..."):
    trades = run_strategy(df)

if trades.empty:
    st.warning("No trades detected.")
    st.stop()

# ---------- Postprocessing ----------
sector_map = df[["symbol", "sector"]].drop_duplicates().set_index("symbol")["sector"]

trades["sector"] = trades["symbol"].map(sector_map)
trades["cap_score"] = trades["symbol"].map(cap_score_map)
trades["cap_emoji"] = trades["symbol"].map(cap_emoji_map)
trades["symbol_display"] = trades.apply(
    lambda r: f"{r['cap_emoji']} {r['symbol']}" if pd.notna(r["cap_emoji"]) else r["symbol"], axis=1
)

# ---- Stop loss (yesterday low)
df["stop_loss"] = df.groupby("symbol")["low"].shift(1)
entry_lows = df[["symbol", "date", "stop_loss"]].rename(columns={"date": "entry_date"})
trades = trades.merge(entry_lows, on=["symbol", "entry_date"], how="left")

# ---- Final % return
latest_prices = (
    df.sort_values("date").groupby("symbol", as_index=False)
      .agg(latest_close=("close", "last"))
)
trades = trades.merge(latest_prices, on="symbol", how="left")

trades["pct_return"] = (trades["exit_price"] / trades["entry"] - 1) * 100
trades["unrealized_pct_return"] = (trades["latest_close"] / trades["entry"] - 1) * 100
trades["final_pct"] = trades.apply(
    lambda r: r["pct_return"] if pd.notna(r["exit_price"]) else r["unrealized_pct_return"], axis=1
)

# ---- Min/Max since entry
minmax = []
for _, r in trades[trades["exit_date"].isna()].iterrows():
    sym, entry_date = r["symbol"], r["entry_date"]
    sl = df[(df["symbol"] == sym) & (df["date"] >= entry_date)]
    if not sl.empty:
        minmax.append((sym, entry_date, sl["low"].min(), sl["high"].max()))
minmax_df = pd.DataFrame(minmax, columns=["symbol", "entry_date", "min_low", "max_high"]) if minmax else pd.DataFrame(columns=["symbol","entry_date","min_low","max_high"])
trades = trades.merge(minmax_df, on=["symbol", "entry_date"], how="left")

# ---------- HOME ----------
if page == "Home":
    st.subheader("🔓 Open Trades")

    open_trades = trades[trades["exit_date"].isna()].copy()
    if open_trades.empty:
        st.info("No open trades.")
    else:
        open_trades["zone"] = "◻️"
        open_trades = open_trades.sort_values("entry_date", ascending=False)
        show = open_trades[["symbol_display", "sector", "entry_date", "entry", "latest_close"]].copy()
        show = date_only_cols(show, ["entry_date"])
        show["entry"] = show["entry"].map(money_str)
        show["latest_close"] = show["latest_close"].map(money_str)
        show = show.rename(columns={
            "symbol_display": "Symbol",
            "sector": "Sector",
            "entry_date": "Entry date",
            "entry": "Entry",
            "latest_close": "Latest close"
        })
        st.dataframe(show, use_container_width=True, hide_index=True)

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

    # ✅ NEW: optional ticker filter based on tickers within selected window
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

    # ✅ Apply filter BEFORE sim
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

    # ✅ Summary KPIs + Enhanced Callouts
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