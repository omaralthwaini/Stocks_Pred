import streamlit as st
import pandas as pd
from datetime import timedelta
from strategy import run_strategy

# ---- UI setup ----
st.set_page_config(page_title="Smart Backtester", layout="wide")
st.title("📈 Smart Backtester")

# ---- Helpers ----
def add_rownum(df):
    df = df.copy()
    df.insert(0, "#", range(1, len(df) + 1))
    return df

def pct_str(x, digits=2, signed=True):
    if pd.isna(x): return "—"
    fmt = f"{{:{'+' if signed else ''}.{digits}f}}%%"
    return fmt.format(x)

def money_str(x):
    return "—" if pd.isna(x) else f"${x:,.2f}"

def date_only(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            df[c] = s.dt.strftime("%Y-%m-%d").where(s.notna(), df[c])
    return df

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("stocks.csv", parse_dates=["date"])
    return df.sort_values(["symbol", "date"]).copy()

@st.cache_data(show_spinner=True)
def detect_trades(df_subset):
    return run_strategy(df_subset)

# ---- Sidebar ----
df_full = load_data()
all_syms = sorted(df_full["symbol"].unique())
min_date = df_full["date"].min()
max_date = df_full["date"].max()

st.sidebar.header("Universe & Window")
use_all = st.sidebar.checkbox("Use ALL tickers", value=False)

selected_syms = all_syms if use_all else st.sidebar.multiselect(
    "Select tickers", all_syms, default=all_syms[:20])

if not selected_syms:
    st.stop()

start_date = st.sidebar.date_input("Start date", value=max(min_date.date(), pd.to_datetime("2024-01-01").date()))
end_date = st.sidebar.date_input("End date", value=max_date.date(), min_value=start_date)

run_btn = st.sidebar.button("🚀 Detect trades")

page = st.sidebar.radio("Page", ["Home", "Insights", "Tester"], index=0)

# ---- Run strategy once ----
if not run_btn and "trades" not in st.session_state:
    st.info("Click **Detect trades** to begin.")
    st.stop()

if run_btn:
    lb_start = pd.Timestamp(start_date) - pd.Timedelta(days=400)
    df_sub = df_full[
        (df_full["symbol"].isin(selected_syms)) &
        (df_full["date"].between(lb_start, pd.Timestamp(end_date)))
    ].copy()
    with st.spinner("⏳ Detecting entries..."):
        trades = detect_trades(df_sub)
    st.session_state["trades"] = trades
    st.session_state["start"] = pd.Timestamp(start_date)
    st.session_state["end"] = pd.Timestamp(end_date)

trades = st.session_state["trades"]
start_ts, end_ts = st.session_state["start"], st.session_state["end"]

if trades.empty:
    st.warning("No trades found.")
    st.stop()

# ---- Common prep ----
closed = trades[trades["exit_date"].notna()].copy()
trades["unrealized_pct"] = (
    (trades["exit_price"] / trades["entry"] - 1.0) * 100.0
    .where(trades["exit_price"].notna(), None)
)

# --------------------------------
# HOME
# --------------------------------
if page == "Home":
    st.subheader("🔓 Open Trades")
    open_trades = trades[trades["exit_date"].isna()].copy()
    if open_trades.empty:
        st.info("No open trades.")
    else:
        show = open_trades[[
            "symbol", "entry_date", "entry", "unrealized_pct"
        ]].copy()
        show["entry_date"] = pd.to_datetime(show["entry_date"]).dt.date
        show["entry"] = show["entry"].map(money_str)
        show["unrealized_pct"] = show["unrealized_pct"].map(lambda x: pct_str(x))
        st.dataframe(add_rownum(show), use_container_width=True, hide_index=True)

# --------------------------------
# INSIGHTS
# --------------------------------
if page == "Insights":
    st.subheader("📊 Closed Trades Performance")

    if closed.empty:
        st.info("No closed trades yet.")
    else:
        closed["pct_return"] = (closed["exit_price"] / closed["entry"] - 1.0) * 100
        closed["days_held"] = (closed["exit_date"] - closed["entry_date"]).dt.days
        closed["win"] = closed["pct_return"] > 0

        stats = (
            closed.groupby("symbol")
                .agg(
                    n_closed=("symbol", "count"),
                    avg_return=("pct_return", "mean"),
                    win_rate=("win", "mean"),
                    avg_days=("days_held", "mean"),
                )
                .reset_index()
        )

        win_mean = closed[closed["win"]].groupby("symbol")["pct_return"].mean().rename("avg_win_return")
        loss_mean = closed[~closed["win"]].groupby("symbol")["pct_return"].mean().rename("avg_loss_return")
        stats = stats.merge(win_mean, on="symbol", how="left").merge(loss_mean, on="symbol", how="left")

        stats["avg_return"] = stats["avg_return"].map(lambda x: pct_str(x))
        stats["avg_win_return"] = stats["avg_win_return"].map(lambda x: pct_str(x))
        stats["avg_loss_return"] = stats["avg_loss_return"].map(lambda x: pct_str(x))
        stats["win_rate"] = stats["win_rate"].map(lambda x: "—" if pd.isna(x) else f"{x:.0%}")
        stats["avg_days"] = stats["avg_days"].map(lambda x: f"{x:.1f}")

        st.dataframe(add_rownum(stats), use_container_width=True, hide_index=True)

        st.download_button("📥 Download All Closed Trades", date_only(closed, ["entry_date","exit_date"]).to_csv(index=False).encode("utf-8"), "closed_trades.csv", "text/csv")

# --------------------------------
# TESTER
# --------------------------------
if page == "Tester":
    st.subheader("🧪 Strategy Tester (one position at a time)")

    trades_in_window = trades[
        (trades["entry_date"] >= start_ts) & (trades["entry_date"] <= end_ts)
    ].copy().sort_values("entry_date")

    if trades_in_window.empty:
        st.warning("No trades in selected window.")
        st.stop()

    starting_cap = st.sidebar.number_input("Starting capital", min_value=1000.0, value=10000.0)
    alloc_pct = st.sidebar.slider("Allocation per trade (%)", 1, 100, 100)

    capital = starting_cap
    ledger = []
    available_from = start_ts

    for _, r in trades_in_window.iterrows():
        entry_d = pd.to_datetime(r["entry_date"])
        if entry_d < available_from:
            continue

        entry = r["entry"]
        ex_d = r["exit_date"]
        ex_price = r["exit_price"]
        sym = r["symbol"]

        if pd.notna(ex_d) and pd.Timestamp(ex_d) <= end_ts:
            status = "Realized"
            ret_pct = (ex_price / entry - 1.0) * 100.0
            exit_d = pd.Timestamp(ex_d)
            exit_px = ex_price
            available_from = exit_d + timedelta(days=1)
        else:
            # unrealized
            lc = df_full[(df_full["symbol"] == sym) & (df_full["date"] <= end_ts)]["close"].iloc[-1]
            exit_px = lc
            ret_pct = (exit_px / entry - 1.0) * 100.0
            status = "Unrealized"
            exit_d = pd.NaT
            available_from = end_ts + timedelta(days=1)

        invest_amt = capital * (alloc_pct / 100.0)
        cap_before = capital
        invest_after = invest_amt * (1.0 + ret_pct / 100.0)
        capital = (capital - invest_amt) + invest_after
        days_held = (exit_d - entry_d).days if pd.notna(exit_d) else (end_ts - entry_d).days

        ledger.append({
            "symbol": sym,
            "entry_date": entry_d,
            "exit_date": exit_d,
            "entry": entry,
            "exit_or_last": exit_px,
            "ret_pct": ret_pct,
            "days_held": days_held,
            "status": status,
            "capital_before": cap_before,
            "capital_after": capital,
        })

    res = pd.DataFrame(ledger)
    res = date_only(res, ["entry_date", "exit_date"])
    res["entry"] = res["entry"].map(money_str)
    res["exit_or_last"] = res["exit_or_last"].map(money_str)
    res["ret_pct"] = res["ret_pct"].map(lambda x: pct_str(x))
    res["capital_before"] = res["capital_before"].map(money_str)
    res["capital_after"] = res["capital_after"].map(money_str)

    n_real = res[res["status"] == "Realized"].shape[0]
    win_rate = (res[res["status"] == "Realized"]["ret_pct"].str.replace("%","").astype(float) > 0).mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades", len(res))
    c2.metric("Final capital", money_str(capital))
    c3.metric("Win rate", "—" if pd.isna(win_rate) else f"{win_rate:.0%}")
    c4.metric("Avg return", pct_str(
        pd.to_numeric(res["ret_pct"].str.replace("%", ""), errors='coerce').mean()
    ))

    st.dataframe(add_rownum(res), use_container_width=True, hide_index=True)
