import streamlit as st
import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go
from strategy import run_strategy

st.title("📈 Smart Backtester — Sector Reports + Open + Recent Trades")

# --- Always read latest ---
def load_data():
    df = pd.read_csv("stocks.csv", parse_dates=["date"])
    caps = pd.read_csv("market_cap.csv")
    return df.sort_values(["symbol", "date"]), caps

df, caps = load_data()

# --- Run strategy ---
with st.spinner("⏳ Detecting trades..."):
    trades = run_strategy(df, caps)

if trades.empty:
    st.warning("⚠️ No trades found.")
    st.stop()
else:
    st.success(f"✅ {len(trades)} trades detected.")

# --- Sector & Cap mapping ---
sector_map    = df[["symbol", "sector"]].drop_duplicates().set_index("symbol")["sector"]
cap_score_map = caps.set_index("symbol")["cap_score"]
cap_emoji_map = caps.set_index("symbol")["cap_emoji"]

trades["sector"]    = trades["symbol"].map(sector_map)
trades["cap_score"] = trades["symbol"].map(cap_score_map)
trades["cap_emoji"] = trades["symbol"].map(cap_emoji_map)
# Drop cap_score 3 and 4 from all views
_cap = pd.to_numeric(trades["cap_score"], errors="coerce")
trades = trades[~_cap.isin([3, 4])].copy()


# --- Latest close per symbol ---
latest_prices = (
    df.sort_values("date")
      .groupby("symbol", as_index=False)
      .agg(latest_close=("close", "last"), latest_date=("date", "max"))
)
trades = trades.merge(latest_prices, on="symbol", how="left")

# --- Stop loss reference (yesterday's low at entry date) ---
df["stop_loss"] = df.groupby("symbol")["low"].shift(1)
entry_lows = df[["symbol", "date", "stop_loss"]].rename(columns={"date": "entry_date"})
trades = trades.merge(entry_lows, on=["symbol", "entry_date"], how="left")

# --- Returns ---
trades["pct_return"] = (trades["exit_price"] / trades["entry"] - 1) * 100
trades["unrealized_pct_return"] = (trades["latest_close"] / trades["entry"] - 1) * 100
trades["final_pct"] = trades.apply(
    lambda r: r["pct_return"] if pd.notna(r["exit_price"]) else r["unrealized_pct_return"],
    axis=1
)

# --- Emoji symbol display ---
trades["symbol_display"] = trades.apply(
    lambda r: f"{r['cap_emoji']} {r['symbol']}" if pd.notna(r["cap_emoji"]) else r["symbol"],
    axis=1
)

# --- Min/Max since entry for OPEN trades ---
open_mask = trades["outcome"] == 0
minmax = []
if open_mask.any():
    for _, r in trades[open_mask].iterrows():
        sym, entry_date = r["symbol"], r["entry_date"]
        sl = df[(df["symbol"] == sym) & (df["date"] >= entry_date)]
        if not sl.empty:
            minmax.append((sym, entry_date, sl["low"].min(), sl["high"].max()))
minmax_df = pd.DataFrame(minmax, columns=["symbol", "entry_date", "min_low", "max_high"]) if minmax else pd.DataFrame(columns=["symbol","entry_date","min_low","max_high"])
trades = trades.merge(minmax_df, on=["symbol", "entry_date"], how="left")

# ===========================
# 📦 Full Trades Export
# ===========================
st.subheader("📦 Download All Trades")
all_trades_to_export = trades.sort_values("entry_date", ascending=False)[[
    "symbol_display","cap_score","sector","entry_date","entry","outcome",
    "exit_price","exit_date","stop_loss","min_low","max_high","final_pct"
]]
st.download_button(
    "📥 Download Full Trade History",
    all_trades_to_export.to_csv(index=False).encode("utf-8"),
    "all_trades.csv","text/csv"
)

# ===========================
# 📂 Download by Sector
# ===========================
st.subheader("📂 Download Trades by Sector")
for sector in trades["sector"].dropna().unique():
    subset = trades[trades["sector"] == sector]
    if not subset.empty:
        st.download_button(
            f"📥 Download {sector} ({len(subset)})",
            subset.sort_values("entry_date", ascending=False).to_csv(index=False).encode("utf-8"),
            f"{sector}_trades.csv","text/csv"
        )

# ===========================
# 🔓 Open Trades Table
# ===========================
open_trades = trades[trades["outcome"] == 0].sort_values("entry_date", ascending=False)
st.subheader(f"🔓 All Open Trades ({len(open_trades)})")
if not open_trades.empty:
    st.dataframe(
        open_trades[[
            "symbol_display","sector","entry_date","entry",
            "latest_close","stop_loss","unrealized_pct_return",
            "min_low","max_high"
        ]],
        use_container_width=True
    )
    st.download_button(
        "📥 Download Open Trades",
        open_trades.to_csv(index=False).encode("utf-8"),
        "open_trades.csv","text/csv"
    )

# ===========================
# 🎯 Near Target (+5%) Watchlist
# ===========================
open_trades_nt = trades.loc[trades["outcome"] == 0].copy()
if not open_trades_nt.empty:
    open_trades_nt["target_price"]   = open_trades_nt["entry"] * 1.05
    open_trades_nt["to_target_pct"]  = (open_trades_nt["latest_close"] / open_trades_nt["target_price"] - 1) * 100
    near = (open_trades_nt.sort_values("to_target_pct", ascending=False)
                        .loc[open_trades_nt["to_target_pct"] <= 5]
                        .head(15))
else:
    near = pd.DataFrame(columns=["symbol_display","sector","entry_date","entry","latest_close","target_price","to_target_pct"])

st.subheader("🎯 Near Target (+5%) Watchlist")
if near.empty:
    st.info("No open positions are close to the +5% target yet.")
else:
    st.dataframe(
        near[[
            "symbol_display","sector","entry_date","entry",
            "latest_close","target_price","to_target_pct"
        ]],
        use_container_width=True
    )

# ===========================
# 🕒 Trades Entered in the Last 7 Days
# ===========================
st.subheader("🕒 Trades Entered in the Last 7 Days")
if trades["entry_date"].notna().any():
    entry_max = trades["entry_date"].dropna().max()
    if pd.notna(entry_max):
        cutoff = entry_max - timedelta(days=7)
        recent = trades[trades["entry_date"] >= cutoff].copy()
        recent = recent.sort_values(["entry_date","cap_score"], ascending=[False, True])
    else:
        recent = pd.DataFrame()
else:
    recent = pd.DataFrame()

if recent.empty:
    st.info("No recent entries available yet.")
else:
    st.dataframe(
        recent[[
            "symbol_display","sector","entry_date","entry",
            "exit_price","exit_date","stop_loss",
            "min_low","max_high","final_pct"
        ]],
        use_container_width=True
    )
    st.download_button(
        "📥 Download Recent Trades",
        recent.to_csv(index=False).encode("utf-8"),
        "recent_trades.csv","text/csv"
    )

# ===========================
# 📤 Trades Exited in the Last 7 Days + Summary
# ===========================
st.subheader("📤 Trades Exited in the Last 7 Days")
if trades["exit_date"].notna().any():
    exit_max = trades["exit_date"].dropna().max()
    if pd.notna(exit_max):
        exit_cutoff = exit_max - timedelta(days=7)
        recent_exits = trades[(trades["exit_date"].notna()) & (trades["exit_date"] >= exit_cutoff)].copy()
    else:
        recent_exits = pd.DataFrame()
else:
    recent_exits = pd.DataFrame()

def _format_exit(row):
    pct = row.get("pct_return")
    if pd.isna(pct):
        return "—"
    return ("✅" if pct > 0 else "❌" if pct < 0 else "⚪") + f" {pct:.2f}%"

if recent_exits.empty:
    st.info("📭 No trades exited in the last 7 days.")
else:
    recent_exits["result"] = recent_exits.apply(_format_exit, axis=1)
    st.dataframe(
        recent_exits[[
            "symbol_display","sector","entry_date","exit_date",
            "entry","exit_price","exit_reason","result"
        ]].sort_values("exit_date", ascending=False),
        use_container_width=True
    )

    st.subheader("📊 Exit Summary (Last 7 Days)")
    win  = (recent_exits["pct_return"] > 0).mean() if not recent_exits.empty else 0.0
    avg  = recent_exits["pct_return"].mean() if not recent_exits.empty else 0.0
    best = recent_exits["pct_return"].max() if not recent_exits.empty else 0.0
    worst= recent_exits["pct_return"].min() if not recent_exits.empty else 0.0
    st.markdown(
        f"- **Count:** {len(recent_exits)}  "
        f"- **Win rate:** {win:.0%}  "
        f"- **Avg return:** {avg:.2f}%  "
        f"- **Best/Worst:** {best:.2f}% / {worst:.2f}%"
    )

# ===========================
# 💲 Open Trade Summaries by Capital
# ===========================
st.subheader("💲Open Trade Summaries by Capital")
open_trades_for_panels = trades[trades["outcome"] == 0].copy()
sorted_emojis = sorted(
    open_trades_for_panels["cap_emoji"].dropna().unique(),
    key=lambda e: open_trades_for_panels.loc[open_trades_for_panels["cap_emoji"] == e, "cap_score"].min()
)

for emoji in sorted_emojis:
    group = open_trades_for_panels[open_trades_for_panels["cap_emoji"] == emoji]
    with st.expander(f"📂 {emoji} — {len(group)} Open Trades", expanded=False):
        for _, row in group.iterrows():
            sym = row["symbol"]
            symbol_disp = row["symbol_display"]
            # candles from entry date onward
            df_sym = df[(df["symbol"] == sym) & (df["date"] >= row["entry_date"])].copy()
            for w in [10, 20, 50, 200]:
                df_sym[f"sma_{w}"] = df_sym["close"].rolling(w).mean()

            st.markdown(f"### {symbol_disp} — Entry: {row['entry_date'].date()} @ ${row['entry']:.2f}")

            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df_sym["date"], open=df_sym["open"], high=df_sym["high"],
                low=df_sym["low"], close=df_sym["close"], name="Price"
            ))
            for w in [10, 20, 50, 200]:
                fig.add_trace(go.Scatter(x=df_sym["date"], y=df_sym[f"sma_{w}"], mode="lines", name=f"SMA-{w}"))

            # 5% target line
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

            distance_to_target = (row["latest_close"] / target_price - 1) * 100

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

            if row["unrealized_pct_return"] >= 10:
                st.success("🟢 Strong Position")
            elif row["unrealized_pct_return"] >= 0:
                st.info("🟡 Moderate Gain")
            else:
                st.warning("🔴 Negative Return")
