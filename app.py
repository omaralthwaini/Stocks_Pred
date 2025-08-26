# app.py (Fast Tester-only)
import streamlit as st
import pandas as pd
from datetime import timedelta
from strategy import run_strategy

st.set_page_config(page_title="Smart Backtester — Fast Tester", layout="wide")
st.title("⚡ Smart Backtester — Fast Tester")

# -------------------- Helpers --------------------
def add_rownum(df_in):
    df = df_in.copy()
    df.insert(0, "#", range(1, len(df) + 1))
    return df

def pct_str(x, digits=2, signed=True):
    if pd.isna(x):
        return "—"
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
    df = df.sort_values(["symbol","date"]).copy()
    return df

@st.cache_data(show_spinner=True)
def detect_seed_entries(df_subset, k_days_rising=3, body_min=0.003):
    # Seed run only; no enhanced guards/maps
    # NOTE: run_strategy ignores `caps` so we pass None for speed.
    return run_strategy(df_subset, caps=None, k_days_rising=k_days_rising, body_min=body_min)

def build_perf(trades_df):
    closed = trades_df[trades_df["exit_date"].notna()].copy()
    if closed.empty:
        return pd.DataFrame(columns=[
            "symbol","win_rate","avg_return","n_closed","avg_days",
            "avg_win_return","avg_loss_return"
        ])

    closed["pct_return"] = (closed["exit_price"] / closed["entry"] - 1) * 100
    closed["win"] = closed["pct_return"] > 0
    closed["days_held"] = (closed["exit_date"] - closed["entry_date"]).dt.days

    base = (
        closed.groupby("symbol")
              .agg(
                  win_rate=("win", "mean"),
                  avg_return=("pct_return", "mean"),
                  n_closed=("pct_return", "size"),
                  avg_days=("days_held", "mean"),
              )
              .reset_index()
    )
    win_mean  = (closed.loc[closed["pct_return"] > 0]
                        .groupby("symbol")["pct_return"]
                        .mean().rename("avg_win_return"))
    loss_mean = (closed.loc[closed["pct_return"] < 0]
                        .groupby("symbol")["pct_return"]
                        .mean().rename("avg_loss_return"))
    perf = base.merge(win_mean, on="symbol", how="left") \
               .merge(loss_mean, on="symbol", how="left")
    return perf

# -------------------- UI: Universe & Window --------------------
df_full = load_data()
all_syms = sorted(df_full["symbol"].unique().tolist())
min_date = df_full["date"].min()
max_date = df_full["date"].max()

st.sidebar.header("Universe & Window")
use_all = st.sidebar.checkbox("Use ALL tickers", value=False)
if use_all:
    selected_syms = all_syms
else:
    # Default to first ~20 symbols for speed; tweak as you like
    default_syms = all_syms[:20]
    selected_syms = st.sidebar.multiselect(
        "Select tickers (universe)", all_syms, default=default_syms
    )
    if not selected_syms:
        st.info("Pick at least one ticker to proceed.")
        st.stop()

start_date = st.sidebar.date_input(
    "Start date",
    value=max(min_date.date(), pd.to_datetime("2024-01-01").date())
)
end_date = st.sidebar.date_input(
    "End date",
    value=max_date.date(), min_value=start_date
)

# Only needed for display (optional)
st.sidebar.header("Display")
near_band_pp = st.sidebar.number_input(
    "Near-band (± percentage points)", min_value=0.1, max_value=10.0, step=0.1, value=1.0
)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Detect entries (seed only)")

# -------------------- Run seed detection on demand --------------------
if not run_btn and "trades_seed" not in st.session_state:
    st.info("Set universe + window, then click **Detect entries (seed only)**.")
    st.stop()

if run_btn:
    # Smart lookback so 200-SMA is valid (≈ 400 calendar days)
    LOOKBACK_CAL_DAYS = 400
    start_ts = pd.Timestamp(start_date)
    end_ts   = pd.Timestamp(end_date)
    lb_start = start_ts - pd.Timedelta(days=LOOKBACK_CAL_DAYS)

    df_sub = df_full[
        (df_full["symbol"].isin(selected_syms)) &
        (df_full["date"].between(lb_start, end_ts))
    ].copy()

    with st.spinner("⏳ Detecting entries…"):
        trades_seed = detect_seed_entries(df_sub)

    st.session_state["trades_seed"] = trades_seed
    st.session_state["universe_syms"] = selected_syms
    st.session_state["window"] = (start_ts, end_ts)

# From cache/session
trades_seed = st.session_state["trades_seed"]
start_ts, end_ts = st.session_state["window"]
universe_syms = st.session_state["universe_syms"]

if trades_seed.empty:
    st.warning("No trades detected in the chosen subset.")
    st.stop()

# -------------------- Tester controls (selection mode) --------------------
st.subheader("🧪 Strategy Tester (one position at a time)")

st.sidebar.header("Tester Settings")
starting_capital = st.sidebar.number_input(
    "Starting capital ($)", min_value=1000.0, step=500.0, value=10000.0
)
alloc_pct = st.sidebar.number_input(
    "Allocation per trade (% of capital)", min_value=1.0, max_value=100.0, step=1.0, value=100.0
)

st.sidebar.subheader("Selection mode")
selection_mode = st.sidebar.radio(
    "Trade selection mode",
    ["By entries", "By tickers", "Top-N (avg return)", "Top-N (avg win return)"],
    index=0
)

# Candidates = entries within the window
candidates = trades_seed[
    (trades_seed["entry_date"] >= start_ts) &
    (trades_seed["entry_date"] <= end_ts) &
    (trades_seed["symbol"].isin(universe_syms))
].copy().sort_values("entry_date")

if candidates.empty:
    st.info("No entries within the selected window for the chosen universe.")
    st.stop()

candidates["label"] = candidates.apply(
    lambda r: f"{r['symbol']} — {pd.to_datetime(r['entry_date']).date()}",
    axis=1
)

# -------------------- Build chosen set --------------------
chosen = pd.DataFrame(columns=candidates.columns)

if selection_mode == "By entries":
    default_selection = candidates["label"].tolist()
    chosen_labels = st.sidebar.multiselect(
        "Choose entries (chronological; overlapping skipped automatically)",
        default_selection, default=default_selection
    )
    chosen = candidates[candidates["label"].isin(chosen_labels)].copy()

elif selection_mode == "By tickers":
    syms_avail = sorted(candidates["symbol"].unique().tolist())
    chosen_syms = st.sidebar.multiselect(
        "Choose tickers (all entries in window)",
        syms_avail, default=syms_avail
    )
    chosen = candidates[candidates["symbol"].isin(chosen_syms)].copy()

else:
    # Build perf only when needed (from ALL detected trades in current universe subset)
    trades_for_perf = trades_seed[trades_seed["symbol"].isin(universe_syms)].copy()
    perf = build_perf(trades_for_perf)

    if perf.empty:
        st.warning("No closed trades available to rank symbols.")
        st.stop()

    if selection_mode == "Top-N (avg return)":
        N = st.sidebar.number_input("Top-N by avg return", min_value=1, max_value=50, step=1, value=5)
        min_closed = st.sidebar.number_input("Min # closed trades", min_value=1, max_value=50, step=1, value=1)
        perf_rank = (perf.loc[perf["n_closed"] >= min_closed]
                        .dropna(subset=["avg_return"])
                        .sort_values("avg_return", ascending=False))
        ranked_syms = [s for s in perf_rank["symbol"].tolist() if s in set(candidates["symbol"].unique())][:int(N)]

        st.caption("Selected top performers (by historical **avg return**):")
        preview = perf_rank.loc[perf_rank["symbol"].isin(ranked_syms),
                                ["symbol","avg_return","win_rate","n_closed"]].copy()
        preview["avg_return"] = preview["avg_return"].map(lambda x: pct_str(x))
        preview["win_rate"]   = preview["win_rate"].map(lambda x: "—" if pd.isna(x) else f"{x:.0%}")
        st.dataframe(add_rownum(preview).rename(columns={
            "symbol":"Ticker","avg_return":"Avg return","win_rate":"Win rate","#":"#","n_closed":"Closed"
        }), use_container_width=True, hide_index=True)

        chosen = candidates[candidates["symbol"].isin(ranked_syms)].copy()

    else:  # Top-N (avg win return)
        N = st.sidebar.number_input("Top-N by avg **win** return", min_value=1, max_value=50, step=1, value=5)
        min_closed = st.sidebar.number_input("Min # closed trades", min_value=1, max_value=50, step=1, value=1)
        perf_win_rank = (perf.loc[perf["n_closed"] >= min_closed]
                            .dropna(subset=["avg_win_return"])
                            .sort_values("avg_win_return", ascending=False))
        ranked_syms = [s for s in perf_win_rank["symbol"].tolist() if s in set(candidates["symbol"].unique())][:int(N)]

        st.caption("Selected top performers (by historical **avg win return**):")
        preview = perf_win_rank.loc[perf_win_rank["symbol"].isin(ranked_syms),
                                    ["symbol","avg_win_return","win_rate","n_closed"]].copy()
        preview["avg_win_return"] = preview["avg_win_return"].map(lambda x: pct_str(x))
        preview["win_rate"]       = preview["win_rate"].map(lambda x: "—" if pd.isna(x) else f"{x:.0%}")
        st.dataframe(add_rownum(preview).rename(columns={
            "symbol":"Ticker","avg_win_return":"Avg win return","win_rate":"Win rate","#":"#","n_closed":"Closed"
        }), use_container_width=True, hide_index=True)

        chosen = candidates[candidates["symbol"].isin(ranked_syms)].copy()

# -------------------- Last close lookups for unrealized --------------------
last_close_upto = (
    df_full[df_full["date"] <= end_ts]
      .sort_values("date")
      .groupby("symbol", as_index=False)
      .agg(last_close_upto=("close", "last"))
      .set_index("symbol")["last_close_upto"].to_dict()
)
latest_fallback = (
    df_full.sort_values("date")
      .groupby("symbol", as_index=False)
      .agg(latest_close=("close","last"))
      .set_index("symbol")["latest_close"].to_dict()
)
def _last_close_for(symbol):
    v = last_close_upto.get(symbol)
    if pd.isna(v) or v is None:
        return latest_fallback.get(symbol)
    return v

# -------------------- One-position-at-a-time simulation --------------------
capital = float(starting_capital)
available_from = start_ts
ledger = []

for _, r in chosen.sort_values("entry_date").iterrows():
    entry_d = pd.to_datetime(r["entry_date"])
    if entry_d < available_from:
        continue

    sym   = r["symbol"]
    entry = float(r["entry"])
    ex_d  = pd.to_datetime(r["exit_date"]) if pd.notna(r["exit_date"]) else pd.NaT
    realized = pd.notna(ex_d) and (ex_d <= end_ts)

    if realized:
        exit_px = float(r["exit_price"])
        ret_pct = (exit_px / entry - 1.0) * 100.0
        exit_d  = ex_d
        status  = "Realized"
        available_from = ex_d + pd.Timedelta(days=1)
    else:
        lc = _last_close_for(sym)
        if lc is None or pd.isna(lc):
            continue
        exit_px = float(lc)
        ret_pct = (exit_px / entry - 1.0) * 100.0
        exit_d  = pd.NaT
        status  = "Unrealized"
        available_from = end_ts + pd.Timedelta(days=1)

    cap_before   = capital
    invest_amt   = capital * (alloc_pct / 100.0)
    cash_amt     = capital - invest_amt
    invest_after = invest_amt * (1.0 + ret_pct/100.0)
    capital      = cash_amt + invest_after

    days_held = (exit_d.normalize() - entry_d.normalize()).days if pd.notna(exit_d) \
                else (end_ts.normalize() - entry_d.normalize()).days

    ledger.append({
        "symbol": sym,
        "entry_date": entry_d,
        "exit_date": exit_d,
        "status": status,
        "entry": entry,
        "exit_or_last": exit_px,
        "ret_pct": ret_pct,
        "days_held": days_held,
        "capital_before": cap_before,
        "capital_after": capital,
    })

if not ledger:
    st.info("No trades were taken (either none selected, or all overlapped with an active position).")
    st.stop()

res = pd.DataFrame(ledger).sort_values("entry_date").reset_index(drop=True)

# -------------------- KPIs + Ledger --------------------
n_trades = len(res)
n_real   = (res["status"] == "Realized").sum()
total_ret = (capital/starting_capital - 1.0)*100.0
win_rate = (res.loc[res["status"] == "Realized", "ret_pct"] > 0).mean() if n_real else float("nan")
avg_real = res.loc[res["status"] == "Realized", "ret_pct"].mean() if n_real else float("nan")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Trades taken", f"{n_trades}")
c2.metric("Final capital", f"${capital:,.2f}")
c3.metric("Total return", pct_str(total_ret))
c4.metric("Realized win rate", "—" if pd.isna(win_rate) else f"{win_rate:.0%}")
c5.metric("Avg realized return", "—" if pd.isna(avg_real) else pct_str(avg_real))

out = res.copy()
out = date_only_cols(out, ["entry_date","exit_date"])
out["entry"]         = out["entry"].map(money_str)
out["exit_or_last"]  = out["exit_or_last"].map(money_str)
out["ret_pct"]       = out["ret_pct"].map(lambda x: pct_str(x))
out["capital_before"]= out["capital_before"].map(money_str)
out["capital_after"] = out["capital_after"].map(money_str)

display_cols = ["symbol","status","entry_date","exit_date","entry","exit_or_last",
                "ret_pct","days_held","capital_before","capital_after"]
st.subheader("📜 Trade Ledger (simulated)")
st.dataframe(add_rownum(out.loc[:, display_cols]).rename(columns={
    "exit_or_last": "Exit / Last",
    "ret_pct": "Return",
}), use_container_width=True, hide_index=True)

st.caption(
    "Notes: seed-only detection on your chosen universe with a 400-day lookback for 200-SMA. "
    "Top-N ranks only within the selected universe."
)
