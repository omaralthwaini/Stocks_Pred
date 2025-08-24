# app.py
import streamlit as st
import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go
from strategy import run_strategy

st.set_page_config(page_title="Smart Backtester", layout="wide")
st.title("📈 Smart Backtester")

# =============== Helpers ===============
def load_data():
    df = pd.read_csv("stocks.csv", parse_dates=["date"])
    caps = pd.read_csv("market_cap.csv")
    return df.sort_values(["symbol", "date"]), caps

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

def date_only_cols(df_in, cols=("entry_date","exit_date","latest_date","date")):
    """
    Return a copy with selected datetime-like columns coerced to string 'YYYY-MM-DD'
    so Streamlit shows them without the '00:00:00' time.
    """
    df = df_in.copy()
    for c in cols:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            df[c] = s.dt.strftime("%Y-%m-%d").where(s.notna(), df[c])
    return df

# =============== Sidebar ===============
st.sidebar.header("View")
page = st.sidebar.radio("Pick a page", ["Home", "Insights"], index=0)

st.sidebar.header("Watchlist Settings")
near_band_pp = st.sidebar.number_input(
    "Near-band (± percentage points)", min_value=0.1, max_value=10.0, step=0.1, value=1.0
)
watchlist_limit = st.sidebar.number_input(
    "Max rows per watchlist", min_value=5, max_value=50, step=1, value=15
)

# =============== Load & Run Strategy ===============
df, caps = load_data()

with st.spinner("⏳ Detecting trades..."):
    trades = run_strategy(df, caps)

if trades.empty:
    st.warning("⚠️ No trades found.")
    st.stop()
else:
    st.success(f"✅ {len(trades)} trades detected.")

# ---- Sector / Cap maps
sector_map     = df[["symbol", "sector"]].drop_duplicates().set_index("symbol")["sector"]
cap_score_map  = caps.set_index("symbol")["cap_score"]
cap_emoji_map  = caps.set_index("symbol")["cap_emoji"]

trades["sector"]     = trades["symbol"].map(sector_map)
trades["cap_score"]  = trades["symbol"].map(cap_score_map)
trades["cap_emoji"]  = trades["symbol"].map(cap_emoji_map)

# Exclude cap 3 & 4
_cap = pd.to_numeric(trades["cap_score"], errors="coerce")
trades = trades[~_cap.isin([3, 4])].copy()

# ---- Latest close per symbol
latest_prices = (
    df.sort_values("date")
      .groupby("symbol", as_index=False)
      .agg(latest_close=("close", "last"), latest_date=("date", "max"))
)
trades = trades.merge(latest_prices, on="symbol", how="left")

# ---- Stop-loss reference (prior day low at entry)
df["stop_loss"] = df.groupby("symbol")["low"].shift(1)
entry_lows = df[["symbol", "date", "stop_loss"]].rename(columns={"date": "entry_date"})
trades = trades.merge(entry_lows, on=["symbol", "entry_date"], how="left")

# ---- Returns
trades["pct_return"]            = (trades["exit_price"] / trades["entry"] - 1) * 100
trades["unrealized_pct_return"] = (trades["latest_close"] / trades["entry"] - 1) * 100
trades["final_pct"] = trades.apply(
    lambda r: r["pct_return"] if pd.notna(r["exit_price"]) else r["unrealized_pct_return"],
    axis=1
)

# ---- Emoji symbol display
trades["symbol_display"] = trades.apply(
    lambda r: f"{r['cap_emoji']} {r['symbol']}" if pd.notna(r["cap_emoji"]) else r["symbol"],
    axis=1
)

# ---- Min/Max since entry for OPEN trades
open_mask = trades["outcome"] == 0
minmax = []
if open_mask.any():
    for _, r in trades[open_mask].iterrows():
        sym, entry_date = r["symbol"], r["entry_date"]
        sl = df[(df["symbol"] == sym) & (df["date"] >= entry_date)]
        if not sl.empty:
            minmax.append((sym, entry_date, sl["low"].min(), sl["high"].max()))
minmax_df = (
    pd.DataFrame(minmax, columns=["symbol", "entry_date", "min_low", "max_high"])
    if minmax else pd.DataFrame(columns=["symbol","entry_date","min_low","max_high"])
)
trades = trades.merge(minmax_df, on=["symbol", "entry_date"], how="left")

# ---- CLOSED-only historical stats per ticker
closed = trades[trades["exit_date"].notna()].copy()
if not closed.empty:
    closed["win"] = closed["pct_return"] > 0
    closed["days_held"] = (closed["exit_date"] - closed["entry_date"]).dt.days

    base = (
        closed.groupby("symbol")
              .agg(
                  win_rate=("win", "mean"),       # 0..1
                  avg_return=("pct_return", "mean"),
                  n_closed=("pct_return", "size"),
                  avg_days=("days_held", "mean")
              )
              .reset_index()
    )
    win_mean  = (closed.loc[closed["pct_return"] > 0]
                        .groupby("symbol")["pct_return"]
                        .mean().rename("avg_win_return"))
    loss_mean = (closed.loc[closed["pct_return"] < 0]
                        .groupby("symbol")["pct_return"]
                        .mean().rename("avg_loss_return"))
    perf = (base.merge(win_mean, on="symbol", how="left")
                 .merge(loss_mean, on="symbol", how="left"))
else:
    perf = pd.DataFrame(columns=[
        "symbol","win_rate","avg_return","n_closed","avg_days",
        "avg_win_return","avg_loss_return"
    ])

# Quick maps for lookups
win_rate_map      = perf.set_index("symbol")["win_rate"].to_dict()          # 0..1
avg_return_map    = perf.set_index("symbol")["avg_return"].to_dict()        # %
avg_win_ret_map   = perf.set_index("symbol")["avg_win_return"].to_dict()    # %
avg_loss_ret_map  = perf.set_index("symbol")["avg_loss_return"].to_dict()   # %
n_closed_map      = perf.set_index("symbol")["n_closed"].to_dict()
avg_days_map      = perf.set_index("symbol")["avg_days"].to_dict()

# =========================================
#                 HOME
# =========================================
if page == "Home":
    # ---------- KPI Summary (last 7 days exits) ----------
    open_trades = trades[trades["outcome"] == 0].copy()
    entries_cutoff = trades["entry_date"].max() - timedelta(days=7) if trades["entry_date"].notna().any() else None
    recent_entries = trades[(trades["entry_date"].notna()) & (trades["entry_date"] >= entries_cutoff)] if entries_cutoff else pd.DataFrame()
    exits_cutoff = trades["exit_date"].dropna().max() - timedelta(days=7) if trades["exit_date"].notna().any() else None
    recent_exits = trades[(trades["exit_date"].notna()) & (trades["exit_date"] >= exits_cutoff)] if exits_cutoff else pd.DataFrame()

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Open trades", len(open_trades))
    kpi_cols[1].metric("Entries (7d)", len(recent_entries))
    kpi_cols[2].metric("Exits (7d)", len(recent_exits))
    if not recent_exits.empty:
        kpi_cols[3].metric("Win rate (7d)", f"{(recent_exits['pct_return'] > 0).mean():.0%}")
        kpi_cols[4].metric("Avg exit return (7d)", pct_str(recent_exits["pct_return"].mean()))
    else:
        kpi_cols[3].metric("Win rate (7d)", "—")
        kpi_cols[4].metric("Avg exit return (7d)", "—")

    # ---------- Latest Entries ----------
    st.subheader("🆕 Latest Entries")

    if recent_entries.empty:
        st.info("No recent entries in the last 7 days.")
    else:
        latest = recent_entries.copy()

        # attach hist stats
        latest["win_rate"]        = latest["symbol"].map(win_rate_map)         # 0..1
        latest["avg_return"]      = latest["symbol"].map(avg_return_map)       # %
        latest["avg_win_return"]  = latest["symbol"].map(avg_win_ret_map)      # %
        latest["avg_loss_return"] = latest["symbol"].map(avg_loss_ret_map)     # %
        latest["n_closed"]        = latest["symbol"].map(n_closed_map).fillna(0).astype(int)

        # price levels implied by historical %s
        latest["guard_loss_price"] = latest.apply(
            lambda r: r["entry"] * (1 + r["avg_loss_return"]/100.0)
            if pd.notna(r["avg_loss_return"]) else pd.NA,
            axis=1
        )
        latest["first_target_price"] = latest.apply(
            lambda r: r["entry"] * (1 + r["avg_return"]/100.0)
            if pd.notna(r["avg_return"]) else pd.NA,
            axis=1
        )
        latest["win_target_price"] = latest.apply(
            lambda r: r["entry"] * (1 + r["avg_win_return"]/100.0)
            if pd.notna(r["avg_win_return"]) else pd.NA,
            axis=1
        )

        # zone label based on proximity to hist %s (uses sidebar near_band_pp)
        def _zone_label(r):
            u  = r["unrealized_pct_return"]
            aw = r["avg_win_return"]
            ar = r["avg_return"]
            al = r["avg_loss_return"]
            if pd.notna(u) and pd.notna(al) and abs(u - al) <= near_band_pp:
                return "🟥 near avg loss"
            if pd.notna(u) and pd.notna(ar) and abs(u - ar) <= near_band_pp:
                return "🟧 near avg return"
            if pd.notna(u) and pd.notna(aw) and abs(u - aw) <= near_band_pp:
                return "🟩 near avg win"
            return "—"

        latest["zone"] = latest.apply(_zone_label, axis=1)

        # sort: newest first, then higher avg win return
        latest["sort_win"] = latest["avg_win_return"].fillna(-1e9)
        latest = latest.sort_values(by=["entry_date", "sort_win"], ascending=[False, False]).drop(columns="sort_win")

        # pretty columns for display
        show = latest.loc[:, [
            "symbol_display","sector","entry_date","entry","latest_close","unrealized_pct_return",
            "guard_loss_price","avg_loss_return",
            "first_target_price","avg_return",
            "win_target_price","avg_win_return",
            "zone","n_closed","win_rate"
        ]].copy()

        # format
        show = date_only_cols(show, ["entry_date"])  # date-only
        show["entry"]                 = show["entry"].map(money_str)
        show["latest_close"]          = show["latest_close"].map(money_str)
        show["unrealized_pct_return"] = show["unrealized_pct_return"].map(lambda x: pct_str(x))
        show["guard_loss_price"]      = show["guard_loss_price"].map(money_str)
        show["first_target_price"]    = show["first_target_price"].map(money_str)
        show["win_target_price"]      = show["win_target_price"].map(money_str)
        show["avg_loss_return"]       = show["avg_loss_return"].map(lambda x: pct_str(x))
        show["avg_return"]            = show["avg_return"].map(lambda x: pct_str(x))
        show["avg_win_return"]        = show["avg_win_return"].map(lambda x: pct_str(x))
        show["win_rate"]              = show["win_rate"].map(lambda x: "—" if pd.isna(x) else f"{x:.0%}")

        # friendly column names
        show = show.rename(columns={
            "entry_date": "Entry date",
            "unrealized_pct_return": "Unrealized",
            "guard_loss_price": "Guard (avg loss)",
            "avg_loss_return": "Avg loss %",
            "first_target_price": "1st target (avg return)",
            "avg_return": "Avg return %",
            "win_target_price": "Win target (avg win)",
            "avg_win_return": "Avg win %",
            "n_closed": "# closed",
            "win_rate": "Win rate"
        })

        # row numbers
        show = add_rownum(show)
        st.dataframe(show, use_container_width=True, hide_index=True)

    # ---------- Positive / Negative Watchlists ----------
    # Attach hist stats to open trades
    open_perf = open_trades.copy()
    open_perf["avg_win_return"]  = open_perf["symbol"].map(avg_win_ret_map)
    open_perf["avg_loss_return"] = open_perf["symbol"].map(avg_loss_ret_map)
    open_perf["n_closed"]        = open_perf["symbol"].map(n_closed_map).fillna(0).astype(int)
    open_perf["win_rate"]        = open_perf["symbol"].map(win_rate_map)

    # Positive: unrealized >= 0 and near avg_win_return
    pos_mask = (
        open_perf["unrealized_pct_return"].notna()
        & open_perf["avg_win_return"].notna()
        & (open_perf["unrealized_pct_return"] >= 0)
        & (open_perf["unrealized_pct_return"] - open_perf["avg_win_return"]).abs() <= near_band_pp
    )
    positive = open_perf[pos_mask].copy().sort_values("unrealized_pct_return", ascending=False).head(int(watchlist_limit))

    # Negative: unrealized <= 0 and near avg_loss_return (usually negative)
    neg_mask = (
        open_perf["unrealized_pct_return"].notna()
        & open_perf["avg_loss_return"].notna()
        & (open_perf["unrealized_pct_return"] <= 0)
        & (open_perf["unrealized_pct_return"] - open_perf["avg_loss_return"]).abs() <= near_band_pp
    )
    negative = open_perf[neg_mask].copy().sort_values("unrealized_pct_return").head(int(watchlist_limit))

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.subheader(f"📈 Positive Watchlist (±{near_band_pp:.1f}pp)")
        if positive.empty:
            st.info("No open trades are near their historical avg **win** return.")
        else:
            cols = [
                "symbol_display","sector","entry_date","entry","latest_close",
                "unrealized_pct_return","avg_win_return","n_closed","win_rate"
            ]
            table = positive.loc[:, cols].copy()
            table = date_only_cols(table, ["entry_date"])   # date-only
            table["unrealized_pct_return"] = table["unrealized_pct_return"].map(lambda x: pct_str(x))
            table["avg_win_return"]        = table["avg_win_return"].map(lambda x: pct_str(x))
            table["win_rate"]              = table["win_rate"].map(lambda x: "—" if pd.isna(x) else f"{x:.0%}")
            table = add_rownum(table)
            st.dataframe(table, use_container_width=True, hide_index=True)

    with c2:
        st.subheader(f"📉 Negative Watchlist (±{near_band_pp:.1f}pp)")
        if negative.empty:
            st.info("No open trades are near their historical avg **loss** return.")
        else:
            cols = [
                "symbol_display","sector","entry_date","entry","latest_close",
                "unrealized_pct_return","avg_loss_return","n_closed","win_rate"
            ]
            table = negative.loc[:, cols].copy()
            table = date_only_cols(table, ["entry_date"])   # date-only
            table["unrealized_pct_return"] = table["unrealized_pct_return"].map(lambda x: pct_str(x))
            table["avg_loss_return"]       = table["avg_loss_return"].map(lambda x: pct_str(x))
            table["win_rate"]              = table["win_rate"].map(lambda x: "—" if pd.isna(x) else f"{x:.0%}")
            table = add_rownum(table)
            st.dataframe(table, use_container_width=True, hide_index=True)

    # ---------- Open Trade Charts by Capital ----------
    st.subheader("💲 Open Trade Summaries by Capital")
    open_for_panels = open_trades.copy()
    sorted_emojis = sorted(
        open_for_panels["cap_emoji"].dropna().unique(),
        key=lambda e: open_for_panels.loc[open_for_panels["cap_emoji"] == e, "cap_score"].min()
    )

    for emoji in sorted_emojis:
        group = open_for_panels[open_for_panels["cap_emoji"] == emoji].copy()
        with st.expander(f"📂 {emoji} — {len(group)} Open Trades", expanded=False):
            for _, row in group.iterrows():
                sym = row["symbol"]
                symbol_disp = row["symbol_display"]

                # series since entry
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

                # Avg win / loss target lines (if we have them)
                aw = avg_win_ret_map.get(sym)   # %
                al = avg_loss_ret_map.get(sym)  # %
                if pd.notna(aw):
                    win_price = row["entry"] * (1 + aw/100.0)
                    fig.add_trace(go.Scatter(
                        x=df_sym["date"], y=[win_price]*len(df_sym),
                        mode="lines", name="Avg Win Return",
                        line=dict(dash="dash")
                    ))
                if pd.notna(al):
                    loss_price = row["entry"] * (1 + al/100.0)
                    fig.add_trace(go.Scatter(
                        x=df_sym["date"], y=[loss_price]*len(df_sym),
                        mode="lines", name="Avg Loss Return",
                        line=dict(dash="dot")
                    ))

                fig.update_layout(
                    height=500, margin=dict(l=10, r=10, t=30, b=10),
                    showlegend=True, xaxis_title="Date", yaxis_title="Price",
                    xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig, use_container_width=True)

                # Panel text
                st.markdown(f"""
                - 🏢 **Sector**: {row['sector']}
                - 🗓 **Days Since Entry**: {(df_sym['date'].max() - row['entry_date']).days}
                - ⛔ **Stop Loss** (prior day low): ${row['stop_loss']:.2f}
                - 💵 **Latest Close**: ${row['latest_close']:.2f}
                - 💹 **Unrealized Return**: {pct_str(row['unrealized_pct_return'])}
                - 📉 **Min Low Since Entry**: ${row['min_low']:.2f}
                - 📈 **Max High Since Entry**: ${row['max_high']:.2f}
                """)

# =========================================
#               INSIGHTS
# =========================================
else:
    # ---- Full trades export
    st.subheader("📦 Download All Trades")
    all_trades_to_export = trades.sort_values("entry_date", ascending=False)[[
        "symbol_display","cap_score","sector","entry_date","entry","outcome",
        "exit_price","exit_date","stop_loss","min_low","max_high","final_pct"
    ]]
    # date-only in the CSV:
    csv_df = date_only_cols(all_trades_to_export, ["entry_date", "exit_date"])
    st.download_button(
        "📥 Download Full Trade History",
        csv_df.to_csv(index=False).encode("utf-8"),
        "all_trades.csv","text/csv"
    )

    # ---- Download by Sector
    st.subheader("📂 Download Trades by Sector")
    for sector in trades["sector"].dropna().unique():
        subset = trades[trades["sector"] == sector]
        if not subset.empty:
            st.download_button(
                f"📥 Download {sector} ({len(subset)})",
                date_only_cols(subset.sort_values("entry_date", ascending=False),
                               ["entry_date","exit_date"]).to_csv(index=False).encode("utf-8"),
                f"{sector}_trades.csv","text/csv"
            )

    # ---- Recent Exits (7 days) + summary
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
        cols = ["symbol_display","sector","entry_date","exit_date","entry","exit_price","exit_reason","result"]
        show_df = recent_exits.loc[:, cols].sort_values("exit_date", ascending=False)
        show_df = date_only_cols(show_df, ["entry_date","exit_date"])  # date-only
        show_df = add_rownum(show_df)
        st.dataframe(show_df, use_container_width=True, hide_index=True)

        st.subheader("📊 Exit Summary (Last 7 Days)")
        win  = (recent_exits["pct_return"] > 0).mean() if not recent_exits.empty else 0.0
        avg  = recent_exits["pct_return"].mean()        if not recent_exits.empty else 0.0
        best = recent_exits["pct_return"].max()         if not recent_exits.empty else 0.0
        worst= recent_exits["pct_return"].min()         if not recent_exits.empty else 0.0
        st.markdown(
            f"- **Count:** {len(recent_exits)}  "
            f"- **Win rate:** {win:.0%}  "
            f"- **Avg return:** {avg:.2f}%  "
            f"- **Best/Worst:** {best:.2f}% / {worst:.2f}%"
        )

    # ---- Top 15 Tickers by Avg Return (Closed Trades)
    st.subheader("🏆 Top 15 Tickers by Avg Return (Closed Trades)")
    if not closed.empty:
        best = (
            closed.groupby("symbol")
                  .agg(
                      n_trades=("pct_return", "size"),
                      avg_return=("pct_return", "mean"),
                      avg_days=("days_held", "mean")
                  )
                  .reset_index()
        )
        win_mean  = (closed.loc[closed["pct_return"] > 0].groupby("symbol")["pct_return"].mean().rename("avg_win_return"))
        loss_mean = (closed.loc[closed["pct_return"] < 0].groupby("symbol")["pct_return"].mean().rename("avg_loss_return"))
        best = best.merge(win_mean, on="symbol", how="left").merge(loss_mean, on="symbol", how="left")

        # enrich
        best["sector"]        = best["symbol"].map(sector_map)
        best["cap_emoji"]     = best["symbol"].map(cap_emoji_map)
        best["symbol_display"]= best.apply(lambda r: f"{r['cap_emoji']} {r['symbol']}" if pd.notna(r["cap_emoji"]) else r["symbol"], axis=1)

        best = best.sort_values("avg_return", ascending=False).head(15)

        disp = best.copy()
        disp["avg_return_str"]     = disp["avg_return"].map(lambda x: pct_str(x))
        disp["avg_win_return_str"] = disp["avg_win_return"].map(lambda x: "—" if pd.isna(x) else pct_str(x))
        disp["avg_loss_return_str"]= disp["avg_loss_return"].map(lambda x: "—" if pd.isna(x) else pct_str(x))
        disp["avg_days_str"]       = disp["avg_days"].map(lambda x: f"{x:.1f}")

        cols = ["symbol_display","sector","n_trades","avg_return_str","avg_win_return_str","avg_loss_return_str","avg_days_str"]
        show_df = add_rownum(disp.loc[:, cols])
        st.dataframe(
            show_df.rename(columns={
                "n_trades": "Closed trades",
                "avg_return_str": "Avg return",
                "avg_win_return_str": "Avg win return",
                "avg_loss_return_str": "Avg loss return",
                "avg_days_str": "Avg days held"
            }),
            use_container_width=True, hide_index=True
        )

        st.download_button(
            "📥 Download Top 15 (Avg Return + Win/Loss)",
            best.loc[:, ["symbol","sector","n_trades","avg_return","avg_win_return","avg_loss_return","avg_days"]]
                .rename(columns={
                    "avg_return": "avg_return_pct",
                    "avg_win_return": "avg_win_return_pct",
                    "avg_loss_return": "avg_loss_return_pct",
                    "avg_days": "avg_days_held"
                })
                .to_csv(index=False).encode("utf-8"),
            "top15_avg_return_with_win_loss.csv","text/csv"
        )
    else:
        st.info("Not enough closed trades yet to compute historical leaders.")
