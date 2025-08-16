import streamlit as st
import pandas as pd
from strategy import run_strategy
from datetime import timedelta

# --- App title ---
st.title("📈 Smart Backtester — Sector Reports + Open + Recent Trades")

# --- Load stock data ---
@st.cache_data
def load_data():
    df = pd.read_csv("stocks.csv", parse_dates=["date"])
    return df.sort_values(["symbol", "date"])

df = load_data()

# --- Run strategy ---
with st.spinner("⏳ Detecting trades..."):
    trades = run_strategy(df)

if trades.empty:
    st.warning("⚠️ No trades found.")
    st.stop()
else:
    st.success(f"✅ {len(trades)} trades detected.")

# --- Map sector info ---
if "sector" in df.columns:
    sector_map = df[["symbol", "sector"]].drop_duplicates().set_index("symbol")["sector"]
    trades["sector"] = trades["symbol"].map(sector_map)
else:
    trades["sector"] = "Unknown"

# --- Add latest price per symbol (for open trade unrealized % PnL) ---
latest_prices = (
    df.sort_values("date")
      .groupby("symbol")
      .agg(latest_close=("close", "last"), latest_date=("date", "max"))
)

trades = trades.merge(latest_prices, on="symbol", how="left")

# --- Calculate P/L (%) ---
trades["pct_return"] = (trades["exit_price"] / trades["entry"] - 1) * 100
trades["unrealized_pct_return"] = (trades["latest_close"] / trades["entry"] - 1) * 100
trades["final_pct"] = trades.apply(
    lambda row: row["pct_return"] if pd.notna(row["exit_price"]) else row["unrealized_pct_return"],
    axis=1
)

# -------------------------------------
# 📂 Download by Sector
# -------------------------------------
st.subheader("📂 Download Trades by Sector")

for sector in trades["sector"].dropna().unique():
    sector_trades = trades[trades["sector"] == sector].copy()
    if not sector_trades.empty:
        sector_trades = sector_trades.sort_values("entry_date", ascending=False)
        csv = sector_trades.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download {sector} ({len(sector_trades)} trades)",
            data=csv,
            file_name=f"{sector.replace(' ', '_')}_trades.csv",
            mime="text/csv"
        )

# -------------------------------------
# 📌 All OPEN Trades
# -------------------------------------
open_trades = trades[trades["outcome"] == 0].sort_values("entry_date", ascending=False)

st.subheader(f"🔓 All Open Trades ({len(open_trades)})")

if open_trades.empty:
    st.info("✅ No open trades remaining.")
else:
    st.dataframe(
        open_trades[[
            "symbol", "sector", "entry_date", "entry", "latest_close",
            "unrealized_pct_return"
        ]],
        use_container_width=True
    )
    csv_open = open_trades.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Open Trades", csv_open, "open_trades.csv", "text/csv")

# -------------------------------------
# ⏰ Recent Entries (Last 7 Days)
# -------------------------------------
st.subheader("🕒 Trades Entered in the Last 7 Days")

latest_entry = trades["entry_date"].max()
recent_cutoff = latest_entry - timedelta(days=7)
recent_trades = trades[trades["entry_date"] >= recent_cutoff].copy()
recent_trades = recent_trades.sort_values("entry_date", ascending=False)

if recent_trades.empty:
    st.info("No recent trades in the last 7 days.")
else:
    st.dataframe(
        recent_trades[[
            "symbol", "sector", "entry_date", "entry",
            "exit_price", "exit_date", "final_pct"
        ]],
        use_container_width=True
    )
    csv_recent = recent_trades.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Recent Trades", csv_recent, "recent_trades.csv", "text/csv")
