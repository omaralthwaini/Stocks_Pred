import streamlit as st
import pandas as pd
from strategy import run_strategy
from datetime import timedelta

# --- App title ---
st.title("📈 Smart Backtester — Sector Reports + Open + Recent Trades")

# --- Refresh logic ---
def load_data():
    df = pd.read_csv("stocks.csv", parse_dates=["date"])
    return df.sort_values(["symbol", "date"])

if st.button("🔄 Refresh from stocks.csv"):
    st.cache_data.clear()

@st.cache_data
def get_cached_data():
    return load_data()

df = get_cached_data()

# --- Run strategy ---
with st.spinner("⏳ Detecting trades..."):
    trades = run_strategy(df)

if trades.empty:
    st.warning("⚠️ No trades found.")
    st.stop()
else:
    st.success(f"✅ {len(trades)} trades detected.")

# --- Sector mapping ---
sector_map = df[["symbol", "sector"]].drop_duplicates().set_index("symbol")["sector"]
trades["sector"] = trades["symbol"].map(sector_map).fillna("Unknown")

# --- Latest close info ---
latest_prices = (
    df.sort_values("date")
      .groupby("symbol")
      .agg(latest_close=("close", "last"), latest_date=("date", "max"))
)
trades = trades.merge(latest_prices, on="symbol", how="left")

# --- Stop Loss at entry ---
df["stop_loss"] = df.groupby("symbol")["low"].shift(1)
entry_lows = df[["symbol", "date", "stop_loss"]].rename(columns={"date": "entry_date"})
trades = trades.merge(entry_lows, on=["symbol", "entry_date"], how="left")

# --- Min/Max since entry (open trades only) ---
minmax_rows = []
for _, row in trades[trades["outcome"] == 0].iterrows():
    sym, entry_date = row["symbol"], row["entry_date"]
    df_slice = df[(df["symbol"] == sym) & (df["date"] >= entry_date)]
    minmax_rows.append({
        "symbol": sym,
        "entry_date": entry_date,
        "min_low": df_slice["low"].min(),
        "max_high": df_slice["high"].max()
    })

minmax_df = pd.DataFrame(minmax_rows)
trades = trades.merge(minmax_df, on=["symbol", "entry_date"], how="left")

# --- P/L calculations ---
trades["pct_return"] = (trades["exit_price"] / trades["entry"] - 1) * 100
trades["unrealized_pct_return"] = (trades["latest_close"] / trades["entry"] - 1) * 100
trades["final_pct"] = trades.apply(
    lambda row: row["pct_return"] if pd.notna(row["exit_price"]) else row["unrealized_pct_return"],
    axis=1
)

# --- 🏷 Market cap labels ---
try:
    cap_df = pd.read_csv("market_cap.csv")  # must have columns: symbol, market_cap
    cap_df = cap_df.drop_duplicates(subset="symbol")
    trades = trades.merge(cap_df, on="symbol", how="left")

    def label_cap(cap):
        if pd.isna(cap): return "❓ Unknown"
        elif cap >= 500_000_000_000: return "🏆 Mega Cap"
        elif cap >= 100_000_000_000: return "🔷 Large Cap"
        elif cap >= 10_000_000_000: return "🟢 Mid Cap"
        elif cap >= 2_000_000_000: return "🟡 Small Cap"
        else: return "🔴 Micro Cap"

    trades["cap_rank"] = trades["market_cap"].apply(label_cap)

except Exception as e:
    st.warning(f"⚠️ Market cap file not found or invalid. {e}")
    trades["cap_rank"] = "❓ Unknown"

# -------------------------------------
# 📌 All OPEN Trades
# -------------------------------------
open_trades = trades[trades["outcome"] == 0].sort_values("entry_date", ascending=False)

st.subheader(f"🔓 All Open Trades ({len(open_trades)})")

if open_trades.empty:
    st.info("✅ No open trades remaining.")
else:
    st.dataframe(
        open_trades[[ "symbol", "sector", "entry_date", "entry", "latest_close",
                      "stop_loss", "unrealized_pct_return", "min_low", "max_high", "cap_rank" ]],
        use_container_width=True
    )
    csv_open = open_trades.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Open Trades", csv_open, "open_trades.csv", "text/csv")

# -------------------------------------
# ⏰ Recent Entries (Last 7 Days)
# -------------------------------------
st.subheader("🕒 Trades Entered in the Last 7 Days")

latest_entry = trades["entry_date"].max()
cutoff = latest_entry - timedelta(days=7)
recent = trades[trades["entry_date"] >= cutoff].sort_values("entry_date", ascending=False)

if recent.empty:
    st.info("No recent trades in the last 7 days.")
else:
    st.dataframe(
        recent[[ "symbol", "sector", "entry_date", "entry", 
                 "exit_price", "exit_date", "stop_loss", "min_low", "max_high", "final_pct", "cap_rank" ]],
        use_container_width=True
    )
    csv_recent = recent.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Recent Trades", csv_recent, "recent_trades.csv", "text/csv")

# -------------------------------------
# 🧭 Detailed Sector Reports for Open Trades
# -------------------------------------
st.subheader("🧭 Open Trade Summaries by Sector")

open_trades = open_trades.sort_values(["sector", "entry_date"], ascending=[True, False])
for sector in open_trades["sector"].dropna().unique():
    sector_trades = open_trades[open_trades["sector"] == sector]
    with st.expander(f"📂 {sector} — {len(sector_trades)} Open Trades", expanded=False):
        for _, row in sector_trades.iterrows():
            sym = row["symbol"]
            df_sym = df[(df["symbol"] == sym) & (df["date"] >= row["entry_date"])].copy()
            for w in [10, 20, 50, 200]:
                df_sym[f"sma_{w}"] = df_sym["close"].rolling(w).mean()

            st.markdown(f"### {sym} — Entry: {row['entry_date'].date()} @ ${row['entry']:.2f} {row['cap_rank']}")

            st.line_chart(df_sym.set_index("date")[["close", "sma_10", "sma_20", "sma_50", "sma_200"]],
                          use_container_width=True)

            st.markdown(f"""
            - 🗓 **Days Since Entry**: {(df_sym['date'].max() - row['entry_date']).days}
            - ⛔ **Stop Loss**: ${row['stop_loss']:.2f}
            - 💵 **Latest Close**: ${row['latest_close']:.2f}
            - 📉 **Min Low Since Entry**: ${row['min_low']:.2f}
            - 📈 **Max High Since Entry**: ${row['max_high']:.2f}
            - 💹 **Unrealized % Return**: {row['unrealized_pct_return']:.2f}%
            """)
