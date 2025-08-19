import streamlit as st
import pandas as pd
from strategy import run_strategy
from datetime import timedelta
import plotly.graph_objects as go  # ✅ Added for candlestick charts

# --- App title ---
st.title("📈 Smart Backtester — Sector Reports + Open + Recent Trades")

# --- Load and cache data ---
def load_data():
    df = pd.read_csv("stocks.csv", parse_dates=["date"])
    caps = pd.read_csv("market_cap.csv")
    return df.sort_values(["symbol", "date"]), caps

# REMOVE this button and its effect entirely:
# if st.button("🔄 Refresh from stocks.csv"):
#     st.cache_data.clear()

# 🚫 DELETE this @st.cache_data decorator completely
# @st.cache_data
def get_latest_data():
    return load_data()

df, caps = get_latest_data()


# --- Run strategy ---
with st.spinner("⏳ Detecting trades..."):
    trades = run_strategy(df, caps)

if trades.empty:
    st.warning("⚠️ No trades found.")
    st.stop()
else:
    st.success(f"✅ {len(trades)} trades detected.")


# --- Sector & Cap Mapping ---
sector_map = df[["symbol", "sector"]].drop_duplicates().set_index("symbol")["sector"]
cap_score_map = caps.set_index("symbol")["cap_score"]
cap_emoji_map = caps.set_index("symbol")["cap_emoji"]

trades["sector"] = trades["symbol"].map(sector_map)
trades["cap_score"] = trades["symbol"].map(cap_score_map)
trades["cap_emoji"] = trades["symbol"].map(cap_emoji_map)

# --- Latest close ---
latest_prices = (
    df.sort_values("date")
      .groupby("symbol")
      .agg(latest_close=("close", "last"), latest_date=("date", "max"))
)
trades = trades.merge(latest_prices, on="symbol", how="left")

# --- Stop loss ---
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

# --- Emoji symbol display ---
trades["symbol_display"] = trades.apply(
    lambda row: f"{row['cap_emoji']} {row['symbol']}" if pd.notna(row["cap_emoji"]) else row["symbol"],
    axis=1
)

# --- Min/Max since entry ---
minmax = []
for _, row in trades[trades["outcome"] == 0].iterrows():
    sym = row["symbol"]
    entry_date = row["entry_date"]
    df_slice = df[(df["symbol"] == sym) & (df["date"] >= entry_date)]
    minmax.append((sym, entry_date, df_slice["low"].min(), df_slice["high"].max()))

minmax_df = pd.DataFrame(minmax, columns=["symbol", "entry_date", "min_low", "max_high"])
trades = trades.merge(minmax_df, on=["symbol", "entry_date"], how="left")

# 📥 Full Trades Export (Matching Sector Format)
st.subheader("📦 Download All Trades")

all_trades_to_export = trades.sort_values("entry_date", ascending=False)[[
    "symbol_display","cap_score", "sector", "entry_date", "entry", "outcome",
    "exit_price", "exit_date", "stop_loss",
    "min_low", "max_high", "final_pct"
]]

csv_all = all_trades_to_export.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Full Trade History", csv_all, "all_trades.csv", "text/csv")


# 📂 Download by Sector
st.subheader("📂 Download Trades by Sector")
for sector in trades["sector"].dropna().unique():
    subset = trades[trades["sector"] == sector]
    if not subset.empty:
        csv = subset.sort_values("entry_date", ascending=False).to_csv(index=False).encode("utf-8")
        st.download_button(f"📥 Download {sector} ({len(subset)})", csv, f"{sector}_trades.csv", "text/csv")

# 🔓 Open Trades Table
open_trades = trades[trades["outcome"] == 0].sort_values("entry_date", ascending=False)
st.subheader(f"🔓 All Open Trades ({len(open_trades)})")
if not open_trades.empty:
    st.dataframe(open_trades[[ "symbol_display", "sector", "entry_date", "entry", "latest_close", "stop_loss", "unrealized_pct_return", "min_low", "max_high" ]], use_container_width=True)
    st.download_button("📥 Download Open Trades", open_trades.to_csv(index=False).encode("utf-8"), "open_trades.csv", "text/csv")

# after you compute latest_close and have 'open_trades'
open_trades = trades[trades["outcome"] == 0].copy()
open_trades["target_price"] = open_trades["entry"] * 1.05
open_trades["to_target_pct"] = (open_trades["latest_close"] / open_trades["target_price"] - 1) * 100

near = open_trades.sort_values("to_target_pct", ascending=False)
near = near[near["to_target_pct"] <= 5]  # within +5% above target still shows as <=5
near = near.head(15)

st.subheader("🎯 Near Target (+5%) Watchlist")
if near.empty:
    st.info("No open positions are close to the +5% target yet.")
else:
    st.dataframe(
        near[["symbol_display","sector","entry_date","entry","latest_close","target_price","to_target_pct"]],
        use_container_width=True
    )

# 🕒 Trades in Last 7 Days
st.subheader("🕒 Trades Entered in the Last 7 Days")
cutoff = trades["entry_date"].max() - timedelta(days=7)
recent = trades[trades["entry_date"] >= cutoff].copy()
recent = recent.sort_values(["entry_date", "cap_score"], ascending=[False, True])
if not recent.empty:
    st.dataframe(recent[[ "symbol_display", "sector", "entry_date", "entry", "exit_price", "exit_date", "stop_loss", "min_low", "max_high", "final_pct" ]], use_container_width=True)
    st.download_button("📥 Download Recent Trades", recent.to_csv(index=False).encode("utf-8"), "recent_trades.csv", "text/csv")

# 📤 Trades Exited in the Last 7 Days
st.subheader("📤 Trades Exited in the Last 7 Days")

exit_cutoff = trades["exit_date"].max() - timedelta(days=7)
recent_exits = trades[
    (trades["exit_date"].notna()) & 
    (trades["exit_date"] >= exit_cutoff)
].copy()


# Outcome emojis
def format_exit(row):
    pct = row["pct_return"]
    if pct > 0:
        emoji = "✅"
    elif pct < 0:
        emoji = "❌"
    else:
        emoji = "⚪"
    return f"{emoji} {pct:.2f}%"

if recent_exits.empty:
    st.info("📭 No trades exited in the last 7 days.")
else:
    recent_exits["result"] = recent_exits.apply(format_exit, axis=1)

    st.dataframe(recent_exits[[ 
        "symbol_display", "sector", "entry_date", "exit_date", 
        "entry", "exit_price", "exit_reason", "result"
    ]].sort_values("exit_date", ascending=False), use_container_width=True)

    # Optional download
    csv_exits = recent_exits.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Recent Exits", csv_exits, "recent_exits.csv", "text/csv")

# after you build `recent_exits`
st.subheader("📊 Exit Summary (Last 7 Days)")
if recent_exits.empty:
    st.info("No exits in the last 7 days.")
else:
    win = (recent_exits["pct_return"] > 0).mean()
    avg = recent_exits["pct_return"].mean()
    best = recent_exits["pct_return"].max()
    worst = recent_exits["pct_return"].min()
    st.markdown(
        f"- **Count:** {len(recent_exits)}  "
        f"- **Win rate:** {win:.0%}  "
        f"- **Avg return:** {avg:.2f}%  "
        f"- **Best/Worst:** {best:.2f}% / {worst:.2f}%"
    )


# 🧭 Open Trade Summaries by Capital
st.subheader("💲Open Trade Summaries by Capital")

# Sort cap_emoji values by corresponding cap_score
sorted_emojis = sorted(
    open_trades["cap_emoji"].dropna().unique(),
    key=lambda emoji: open_trades.loc[open_trades["cap_emoji"] == emoji, "cap_score"].min()
)

for emoji in sorted_emojis:
    group = open_trades[open_trades["cap_emoji"] == emoji]
    with st.expander(f"📂 {emoji} — {len(group)} Open Trades", expanded=False):
        for _, row in group.iterrows():
            symbol_disp = row["symbol_display"]
            sym = row["symbol"]
            df_sym = df[(df["symbol"] == sym) & (df["date"] >= row["entry_date"])].copy()

            for w in [10, 20, 50, 200]:
                df_sym[f"sma_{w}"] = df_sym["close"].rolling(w).mean()

            st.markdown(f"### {symbol_disp} — Entry: {row['entry_date'].date()} @ ${row['entry']:.2f}")

            # ✅ Candlestick chart + 5% Target Line
            fig = go.Figure()

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df_sym["date"], open=df_sym["open"], high=df_sym["high"],
                low=df_sym["low"], close=df_sym["close"], name="Price"
            ))

            # SMAs
            for w in [10, 20, 50, 200]:
                fig.add_trace(go.Scatter(x=df_sym["date"], y=df_sym[f"sma_{w}"], mode="lines", name=f"SMA-{w}"))

            # 🎯 Add 5% target line
            target_price = row["entry"] * 1.05
            fig.add_trace(go.Scatter(
                x=df_sym["date"],
                y=[target_price] * len(df_sym),
                mode="lines",
                name="🎯 Target +5%",
                line=dict(dash="dash", color="green")
            ))

            fig.update_layout(
                height=500, margin=dict(l=10, r=10, t=30, b=10),
                showlegend=True, xaxis_title="Date", yaxis_title="Price",
                xaxis_rangeslider_visible=False
            )

            st.plotly_chart(fig, use_container_width=True)

            # 🎯 Distance to Target
            distance_to_target = (row["latest_close"] / target_price - 1) * 100

            # Trade summary with target info
            st.markdown(f"""
            - 🏢 **Sector**: {row['sector']}
            - 🗓 **Days Since Entry**: {(df_sym['date'].max() - row['entry_date']).days}
            - ⛔ **Stop Loss**: ${row["stop_loss"]:.2f}
            - 💵 **Latest Close**: ${row["latest_close"]:.2f}
            - 🎯 **Target (5%)**: ${target_price:.2f}
            - 📐 **Distance to Target**: {distance_to_target:.2f}%
            - 📉 **Min Low Since Entry**: ${row["min_low"]:.2f}
            - 📈 **Max High Since Entry**: ${row["max_high"]:.2f}
            - 💹 **Unrealized Return**: {row["unrealized_pct_return"]:.2f}%
            """)

            # Return strength label
            if row["unrealized_pct_return"] >= 10:
                st.success("🟢 Strong Position")
            elif row["unrealized_pct_return"] >= 0:
                st.info("🟡 Moderate Gain")
            else:
                st.warning("🔴 Negative Return")
