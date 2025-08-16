import streamlit as st
import pandas as pd
from strategy import run_strategy
from datetime import timedelta

# --- App title ---
st.title("📈 Smart Backtester — Sector Exports + Recent Trades")

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

# --- Merge sector info ---
if "sector" in df.columns:
    sector_map = df[["symbol", "sector"]].drop_duplicates().set_index("symbol")["sector"]
    trades["sector"] = trades["symbol"].map(sector_map)
else:
    trades["sector"] = "Unknown"

# --- Sector CSV downloaders ---
st.subheader("📂 Download Trades by Sector")

for sector in trades["sector"].dropna().unique():
    sector_trades = trades[trades["sector"] == sector]
    if not sector_trades.empty:
        csv = sector_trades.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download {sector} ({len(sector_trades)} trades)",
            data=csv,
            file_name=f"{sector.replace(' ', '_')}_trades.csv",
            mime="text/csv"
        )

# --- Recent trades (last 7 days) ---
st.subheader("🕒 Trades Entered in the Last 7 Days")

latest_date = trades["entry_date"].max()
recent_cutoff = latest_date - timedelta(days=7)
recent_trades = trades[trades["entry_date"] >= recent_cutoff].sort_values("entry_date", ascending=False)

if recent_trades.empty:
    st.info("No trades entered in the last 7 days.")
else:
    st.dataframe(recent_trades, use_container_width=True)
    csv_recent = recent_trades.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Recent Trades (Last 7 Days)", csv_recent, "recent_trades.csv", "text/csv")
