import streamlit as st
import pandas as pd
from strategy import run_strategy
from datetime import timedelta

# --- App title ---
st.title("📈 Smart Backtester — Sector Reports + Open + Recent Trades")

# --- Load and cache data ---
def load_data():
    df = pd.read_csv("stocks.csv", parse_dates=["date"])
    caps = pd.read_csv("market_cap.csv")  # Must include symbol, cap_score, cap_emoji
    return df.sort_values(["symbol", "date"]), caps

if st.button("🔄 Refresh from stocks.csv"):
    st.cache_data.clear()

@st.cache_data
def get_cached_data():
    return load_data()

df, caps = get_cached_data()

# --- Run strategy ---
with st.spinner("⏳ Detecting trades..."):
    trades = run_strategy(df)

if trades.empty:
    st.warning("⚠️ No trades found.")
    st.stop()
else:
    st.success(f"✅ {len(trades)} trades detected.")

# --- Sector + Market Cap Mapping ---
sector_map = df[["symbol", "sector"]].drop_duplicates().set_index("symbol")["sector"]
trades["sector"] = trades["symbol"].map(sector_map)

cap_score_map = caps.set_index("symbol")["cap_score"]
cap_emoji_map = caps.set_index("symbol")["cap_emoji"]

trades["cap_score"] = trades["symbol"].map(cap_score_map)
trades["cap_emoji"] = trades["symbol"].map(cap_emoji_map)

# --- Latest close per symbol ---
latest_prices = (
    df.sort_values("date")
      .groupby("symbol")
      .agg(latest_close=("close", "last"), latest_date=("date", "max"))
)
trades = trades.merge(latest_prices, on="symbol", how="left")

# --- Stop loss (yesterday’s low) ---
df["stop_loss"] = df.groupby("symbol")["low"].shift(1)
entry_lows = df[["symbol", "date", "stop_loss"]].rename(columns={"date": "entry_date"})
trades = trades.merge(entry_lows, on=["symbol", "entry_date"], how="left")

# --- Returns ---
trades["pct_return"] = (trades["exit_price"] / trades["entry"] - 1) * 100
trades["unrealized_pct_return"] = (trades["latest_close"] / trades["entry"] - 1) * 100
trades["final_pct"] = trades.apply(
    lambda row: row["pct_return"] if pd.notna(row["exit_price"]) else row["unrealized_pct_return"],
    axis=1
)

# --- Attach emoji to symbol display ---
def attach_symbol_display(row):
    if pd.isna(row["cap_emoji"]):
        return row["symbol"]
    return f"{row['cap_emoji']} {row['symbol']}"

trades["symbol_display"] = trades.apply(attach_symbol_display, axis=1)

# --- Min/Max since entry (for open trades) ---
minmax = []
for _, row in trades[trades["outcome"] == 0].iterrows():
    sym = row["symbol"]
    entry_date = row["entry_date"]
    df_slice = df[(df["symbol"] == sym) & (df["date"] >= entry_date)]
    minmax.append((row["symbol"], entry_date, df_slice["low"].min(), df_slice["high"].max()))

minmax_df = pd.DataFrame(minmax, columns=["symbol", "entry_date", "min_low", "max_high"])
trades = trades.merge(minmax_df, on=["symbol", "entry_date"], how="left")

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
# 🔓 Open Trades
# -------------------------------------
open_trades = trades[trades["outcome"] == 0].sort_values("entry_date", ascending=False)

st.subheader(f"🔓 All Open Trades ({len(open_trades)})")

if open_trades.empty:
    st.info("✅ No open trades remaining.")
else:
    st.dataframe(
        open_trades[[ 
            "symbol_display", "sector", "entry_date", "entry", "latest_close",
            "stop_loss", "unrealized_pct_return", "min_low", "max_high"
        ]],
        use_container_width=True
    )
    csv_open = open_trades.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Open Trades", csv_open, "open_trades.csv", "text/csv")

# -------------------------------------
# ⏰ Trades in Last 7 Days
# -------------------------------------
st.subheader("🕒 Trades Entered in the Last 7 Days")

latest_entry = trades["entry_date"].max()
recent_cutoff = latest_entry - timedelta(days=7)
recent_trades = trades[trades["entry_date"] >= recent_cutoff].copy()

recent_trades = recent_trades.sort_values(
    ["entry_date", "cap_score"],
    ascending=[False, True]
)

if recent_trades.empty:
    st.info("No recent trades in the last 7 days.")
else:
    st.dataframe(
        recent_trades[[
            "symbol_display", "sector", "entry_date", "entry",
            "exit_price", "exit_date", "stop_loss", "min_low", "max_high", "final_pct"
        ]],
        use_container_width=True
    )
    csv_recent = recent_trades.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Recent Trades", csv_recent, "recent_trades.csv", "text/csv")

# -------------------------------------
# 🧭 Open Trade Summaries by Sector
# -------------------------------------
st.subheader("🧭 Open Trade Summaries by Sector")

for sector in open_trades["sector"].dropna().unique():
    group = open_trades[open_trades["sector"] == sector]
    with st.expander(f"📂 {sector} — {len(group)} Open Trades", expanded=False):
        for _, row in group.iterrows():
            symbol_disp = row["symbol_display"]
            sym = row["symbol"]  # actual raw symbol
            df_sym = df[(df["symbol"] == sym) & (df["date"] >= row["entry_date"])].copy()
            for w in [10, 20, 50, 200]:
                df_sym[f"sma_{w}"] = df_sym["close"].rolling(w).mean()

            st.markdown(f"### {symbol_disp} — Entry: {row['entry_date'].date()} @ ${row['entry']:.2f}")
            st.line_chart(df_sym.set_index("date")[["close", "sma_10", "sma_20", "sma_50", "sma_200"]],
                          use_container_width=True)

            st.markdown(f"""
            - 🗓 **Days Since Entry**: {(df_sym["date"].max() - row["entry_date"]).days}
            - ⛔ **Stop Loss**: ${row["stop_loss"]:.2f}
            - 💵 **Latest Close**: ${row["latest_close"]:.2f}
            - 📉 **Min Low Since Entry**: ${row["min_low"]:.2f}
            - 📈 **Max High Since Entry**: ${row["max_high"]:.2f}
            - 💹 **Unrealized Return**: {row["unrealized_pct_return"]:.2f}%
            """)

            if row["unrealized_pct_return"] >= 10:
                st.success("🟢 Strong Position")
            elif row["unrealized_pct_return"] >= 0:
                st.info("🟡 Moderate Gain")
            else:
                st.warning("🔴 Negative Return")
