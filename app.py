import streamlit as st
import pandas as pd
from strategy import run_strategy
from datetime import timedelta

# --- Title ---
st.title("📈 Smart Backtester — Sector Reports + Open + Recent Trades + Ranking")

# --- Refresh Logic ---
def load_data():
    df = pd.read_csv("stocks.csv", parse_dates=["date"])
    mc = pd.read_csv("market_cap.csv")  # Make sure this file exists
    df = df.merge(mc, on="symbol", how="left")
    return df.sort_values(["symbol", "date"])

if st.button("🔄 Refresh from CSVs"):
    st.cache_data.clear()

@st.cache_data
def get_data():
    return load_data()

df = get_data()

# --- Run Strategy ---
with st.spinner("⏳ Detecting trades..."):
    trades = run_strategy(df)

if trades.empty:
    st.warning("⚠️ No trades found.")
    st.stop()

# --- Sector Mapping ---
if "sector" in df.columns:
    sector_map = df[["symbol", "sector"]].drop_duplicates().set_index("symbol")["sector"]
    trades["sector"] = trades["symbol"].map(sector_map)

# --- Add latest price per symbol ---
latest_prices = (
    df.sort_values("date")
      .groupby("symbol")
      .agg(latest_close=("close", "last"), latest_date=("date", "max"))
)
trades = trades.merge(latest_prices, on="symbol", how="left")

# --- Stop loss from entry ---
df["stop_loss"] = df.groupby("symbol")["low"].shift(1)
entry_lows = df[["symbol", "date", "stop_loss"]].rename(columns={"date": "entry_date"})
trades = trades.merge(entry_lows, on=["symbol", "entry_date"], how="left")

# --- Market cap mapping ---
if "market_cap" in df.columns:
    cap_map = df[["symbol", "market_cap"]].drop_duplicates().set_index("symbol")["market_cap"]
    trades["market_cap"] = trades["symbol"].map(cap_map)

# --- PnL calculations ---
trades["pct_return"] = (trades["exit_price"] / trades["entry"] - 1) * 100
trades["unrealized_pct_return"] = (trades["latest_close"] / trades["entry"] - 1) * 100
trades["final_pct"] = trades.apply(
    lambda row: row["pct_return"] if pd.notna(row["exit_price"]) else row["unrealized_pct_return"],
    axis=1
)

# --- Min/Max since entry for open trades ---
minmax = []
for _, row in trades[trades["outcome"] == 0].iterrows():
    d = df[(df["symbol"] == row["symbol"]) & (df["date"] >= row["entry_date"])]
    minmax.append((
        row["symbol"], row["entry_date"], d["low"].min(), d["high"].max()
    ))

minmax_df = pd.DataFrame(minmax, columns=["symbol", "entry_date", "min_low", "max_high"])
trades = trades.merge(minmax_df, on=["symbol", "entry_date"], how="left")

# --- SECTOR DOWNLOADS ---
st.subheader("📂 Download Trades by Sector")
for sector in trades["sector"].dropna().unique():
    sub = trades[trades["sector"] == sector]
    csv = sub.sort_values("entry_date", ascending=False).to_csv(index=False).encode("utf-8")
    st.download_button(f"📥 {sector} ({len(sub)})", csv, f"{sector}_trades.csv", "text/csv")

# --- OPEN TRADES ---
open_trades = trades[trades["outcome"] == 0].sort_values("entry_date", ascending=False)
st.subheader(f"🔓 Open Trades ({len(open_trades)})")
if open_trades.empty:
    st.info("✅ No open trades.")
else:
    st.dataframe(open_trades[[
        "symbol", "sector", "entry_date", "entry", "latest_close", "stop_loss",
        "unrealized_pct_return", "market_cap", "min_low", "max_high"
    ]], use_container_width=True)

    st.download_button(
        "📥 Download Open Trades",
        open_trades.to_csv(index=False).encode("utf-8"),
        "open_trades.csv",
        "text/csv"
    )

# --- RECENT TRADES ---
st.subheader("🕒 Trades Entered in the Last 7 Days")
cutoff = trades["entry_date"].max() - timedelta(days=7)
recent_trades = trades[trades["entry_date"] >= cutoff].sort_values("entry_date", ascending=False)

if recent_trades.empty:
    st.info("No trades entered in the last 7 days.")
else:
    st.dataframe(recent_trades[[
        "symbol", "sector", "entry_date", "entry",
        "exit_price", "exit_date", "stop_loss", "market_cap",
        "min_low", "max_high", "final_pct"
    ]], use_container_width=True)

    st.download_button(
        "📥 Download Recent Trades",
        recent_trades.to_csv(index=False).encode("utf-8"),
        "recent_trades.csv",
        "text/csv"
    )
