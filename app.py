import streamlit as st
import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go

from strategy import run_strategy, run_strategy_stacked

# ---------- Data loaders ----------
def load_data():
    df = pd.read_csv("stocks.csv", parse_dates=["date"])
    caps = pd.read_csv("market_cap.csv")
    return df.sort_values(["symbol", "date"]), caps

def get_latest_data():
    return load_data()

# ---------- Rendering helpers ----------
def enrich_trades(trades, df, caps):
    if trades.empty:
        return trades

    # sector & cap maps
    sector_map    = df[["symbol", "sector"]].drop_duplicates().set_index("symbol")["sector"]
    cap_score_map = caps.set_index("symbol")["cap_score"]
    cap_emoji_map = caps.set_index("symbol")["cap_emoji"]

    trades["sector"]     = trades["symbol"].map(sector_map)
    trades["cap_score"]  = trades["symbol"].map(cap_score_map)
    trades["cap_emoji"]  = trades["symbol"].map(cap_emoji_map)

    # latest close
    latest_prices = (
        df.sort_values("date")
          .groupby("symbol")
          .agg(latest_close=("close", "last"), latest_date=("date", "max"))
    )
    trades = trades.merge(latest_prices, on="symbol", how="left")

    # stop loss (yesterday's low relative to entry date)
    df_tmp = df.copy()
    df_tmp["stop_loss"] = df_tmp.groupby("symbol")["low"].shift(1)
    entry_lows = df_tmp[["symbol", "date", "stop_loss"]].rename(columns={"date": "entry_date"})
    trades = trades.merge(entry_lows, on=["symbol", "entry_date"], how="left")

    # returns
    trades["pct_return"] = (trades["exit_price"] / trades["entry"] - 1) * 100
    trades["unrealized_pct_return"] = (trades["latest_close"] / trades["entry"] - 1) * 100
    trades["final_pct"] = trades.apply(
        lambda r: r["pct_return"] if pd.notna(r["exit_price"]) else r["unrealized_pct_return"],
        axis=1
    )

    # symbol display with emoji
    trades["symbol_display"] = trades.apply(
        lambda r: f"{r['cap_emoji']} {r['symbol']}" if pd.notna(r.get("cap_emoji")) else r["symbol"],
        axis=1
    )

    # min/max since entry for open positions
    open_mask = trades["outcome"] == 0
    minmax = []
    for _, row in trades[open_mask].iterrows():
        sym, entry_date = row["symbol"], row["entry_date"]
        df_slice = df[(df["symbol"] == sym) & (df["date"] >= entry_date)]
        if not df_slice.empty:
            minmax.append((sym, entry_date, df_slice["low"].min(), df_slice["high"].max()))
    if minmax:
        minmax_df = pd.DataFrame(minmax, columns=["symbol", "entry_date", "min_low", "max_high"])
        trades = trades.merge(minmax_df, on=["symbol", "entry_date"], how="left")
    else:
        trades["min_low"] = pd.NA
        trades["max_high"] = pd.NA

    return trades

def plot_symbol(df_sym, row):
    fig = go.Figure()
    # candles
    fig.add_trace(go.Candlestick(
        x=df_sym["date"], open=df_sym["open"], high=df_sym["high"],
        low=df_sym["low"], close=df_sym["close"], name="Price"
    ))
    # SMAs
    for w in [10, 20, 50, 200]:
        if f"sma_{w}" in df_sym:
            fig.add_trace(go.Scatter(x=df_sym["date"], y=df_sym[f"sma_{w}"], mode="lines", name=f"SMA-{w}"))

    # 5% target
    target_price = row["entry"] * 1.05
    fig.add_trace(go.Scatter(
        x=df_sym["date"], y=[target_price] * len(df_sym),
        mode="lines", name="🎯 Target +5%", line=dict(dash="dash")
    ))

    fig.update_layout(
        height=500, margin=dict(l=10, r=10, t=30, b=10),
        showlegend=True, xaxis_title="Date", yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )
    return fig

def render_dashboard(trades, df, caps, title_suffix=""):
    st.subheader(f"Results {title_suffix}")
    trades = enrich_trades(trades, df, caps)

    if trades.empty:
        st.info("No trades found.")
        return

    st.success(f"✅ {len(trades)} trades detected.")

    # ---------- Full export ----------
    st.subheader("📦 Download All Trades")
    all_cols = [
        "symbol_display","cap_score","sector","entry_date","entry","outcome",
        "exit_price","exit_date","stop_loss","min_low","max_high","final_pct"
    ]
    csv_all = trades.sort_values("entry_date", ascending=False)[all_cols].to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Full Trade History", csv_all, "all_trades.csv", "text/csv")

    # ---------- By sector ----------
    st.subheader("📂 Download Trades by Sector")
    for sector in trades["sector"].dropna().unique():
        subset = trades[trades["sector"] == sector]
        if not subset.empty:
            csv = subset.sort_values("entry_date", ascending=False).to_csv(index=False).encode("utf-8")
            st.download_button(f"📥 Download {sector} ({len(subset)})", csv, f"{sector}_trades.csv", "text/csv")

    # ---------- Open trades ----------
    open_trades = trades[trades["outcome"] == 0].sort_values("entry_date", ascending=False)
    st.subheader(f"🔓 All Open Trades ({len(open_trades)})")
    if not open_trades.empty:
        st.dataframe(
            open_trades[["symbol_display","sector","entry_date","entry","latest_close","stop_loss",
                         "unrealized_pct_return","min_low","max_high"]],
            use_container_width=True
        )
        st.download_button("📥 Download Open Trades",
                           open_trades.to_csv(index=False).encode("utf-8"),
                           "open_trades.csv", "text/csv")

    # ---------- Near target watchlist ----------
    if not open_trades.empty:
        st.subheader("🎯 Near Target (+5%) Watchlist")
        tmp = open_trades.copy()
        tmp["target_price"] = tmp["entry"] * 1.05
        tmp["to_target_pct"] = (tmp["latest_close"] / tmp["target_price"] - 1) * 100
        # Keep those not far above target (<= 5% above) and sort closest first
        tmp = tmp[tmp["to_target_pct"] <= 5].sort_values("to_target_pct", ascending=False).head(15)
        if tmp.empty:
            st.info("No open positions are close to the +5% target yet.")
        else:
            # Show overall return (not distance-to-target)
            tmp["return_pct"] = tmp["unrealized_pct_return"]
            st.dataframe(
                tmp[["symbol_display","sector","entry_date","entry","latest_close","target_price","return_pct"]],
                use_container_width=True
            )

    # ---------- Recent entries ----------
    st.subheader("🕒 Trades Entered in the Last 7 Days")
    cutoff = trades["entry_date"].max() - timedelta(days=7)
    recent = trades[trades["entry_date"] >= cutoff].copy()
    recent = recent.sort_values(["entry_date", "cap_score"], ascending=[False, True])
    if not recent.empty:
        st.dataframe(
            recent[["symbol_display","sector","entry_date","entry","exit_price","exit_date",
                    "stop_loss","min_low","max_high","final_pct"]],
            use_container_width=True
        )
        st.download_button("📥 Download Recent Trades",
                           recent.to_csv(index=False).encode("utf-8"),
                           "recent_trades.csv", "text/csv")

    # ---------- Recent exits ----------
    st.subheader("📤 Trades Exited in the Last 7 Days")
    if trades["exit_date"].notna().any():
        exit_cutoff = trades["exit_date"].max() - timedelta(days=7)
        recent_exits = trades[(trades["exit_date"].notna()) & (trades["exit_date"] >= exit_cutoff)].copy()
    else:
        recent_exits = pd.DataFrame(columns=trades.columns)

    def format_exit(row):
        pct = row["pct_return"]
        if pd.isna(pct): return "—"
        return ("✅ " if pct > 0 else "❌ " if pct < 0 else "⚪ ") + f"{pct:.2f}%"

    if recent_exits.empty:
        st.info("📭 No trades exited in the last 7 days.")
    else:
        recent_exits["result"] = recent_exits.apply(format_exit, axis=1)
        st.dataframe(
            recent_exits[["symbol_display","sector","entry_date","exit_date","entry","exit_price","exit_reason","result"]]
            .sort_values("exit_date", ascending=False),
            use_container_width=True
        )
        st.download_button("📥 Download Recent Exits",
                           recent_exits.to_csv(index=False).encode("utf-8"),
                           "recent_exits.csv", "text/csv")

        # quick stats
        st.subheader("📊 Exit Summary (Last 7 Days)")
        win  = (recent_exits["pct_return"] > 0).mean()
        avg  = recent_exits["pct_return"].mean()
        best = recent_exits["pct_return"].max()
        worst= recent_exits["pct_return"].min()
        st.markdown(
            f"- **Count:** {len(recent_exits)}  "
            f"- **Win rate:** {win:.0%}  "
            f"- **Avg return:** {avg:.2f}%  "
            f"- **Best/Worst:** {best:.2f}% / {worst:.2f}%"
        )

    # ---------- Open Trade Summaries by Capital ----------
    st.subheader("💲Open Trade Summaries by Capital")
    open_trades = trades[trades["outcome"] == 0].copy()
    if not open_trades.empty:
        sorted_emojis = sorted(
            open_trades["cap_emoji"].dropna().unique(),
            key=lambda e: open_trades.loc[open_trades["cap_emoji"] == e, "cap_score"].min()
        )
        for emoji in sorted_emojis:
            group = open_trades[open_trades["cap_emoji"] == emoji]
            with st.expander(f"📂 {emoji} — {len(group)} Open Trades", expanded=False):
                for _, row in group.iterrows():
                    sym = row["symbol"]
                    df_sym = df[(df["symbol"] == sym) & (df["date"] >= row["entry_date"])].copy()
                    # add SMAs for the chart
                    for w in [10, 20, 50, 200]:
                        df_sym[f"sma_{w}"] = df_sym["close"].rolling(w).mean()

                    st.markdown(f"### {row['symbol_display']} — Entry: {row['entry_date'].date()} @ ${row['entry']:.2f}")
                    fig = plot_symbol(df_sym, row)
                    st.plotly_chart(fig, use_container_width=True)

                    target_price = row["entry"] * 1.05
                    distance_to_target = (row["latest_close"] / target_price - 1) * 100

                    st.markdown(f"""
- 🏢 **Sector**: {row['sector']}
- 🗓 **Days Since Entry**: {(df_sym['date'].max() - row['entry_date']).days}
- ⛔ **Stop Loss**: ${row["stop_loss"]:.2f}
- 💵 **Latest Close**: ${row["latest_close"]:.2f}
- 🎯 **Target (5%)**: ${target_price:.2f}
- 📐 **Distance to Target**: {distance_to_target:.2f}%
- 📉 **Min Low Since Entry**: ${row.get("min_low", float('nan')):.2f}
- 📈 **Max High Since Entry**: ${row.get("max_high", float('nan')):.2f}
- 💹 **Unrealized Return**: {row["unrealized_pct_return"]:.2f}%
                    """)

                    if row["unrealized_pct_return"] >= 10:
                        st.success("🟢 Strong Position")
                    elif row["unrealized_pct_return"] >= 0:
                        st.info("🟡 Moderate Gain")
                    else:
                        st.warning("🔴 Negative Return")

# ---------- App ----------
st.title("📈 Smart Backtester — Two Strategies")

df, caps = get_latest_data()

tab1, tab2 = st.tabs(["Baseline Strategy", "Stacked SMA Strategy"])

with tab1:
    with st.spinner("⏳ Running Baseline Strategy..."):
        trades_base = run_strategy(df, caps, k_days_rising=3, body_min=0.003)
    render_dashboard(trades_base, df, caps, title_suffix="(Baseline)")

with tab2:
    with st.spinner("⏳ Running Stacked SMA Strategy..."):
        trades_stack = run_strategy_stacked(df, caps, k_days_rising=3, body_min=0.003)
    render_dashboard(trades_stack, df, caps, title_suffix="(Stacked SMA)")
