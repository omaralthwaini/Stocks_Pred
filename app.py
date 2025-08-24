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
    """Coerce selected datetime-like columns to YYYY-MM-DD strings for display."""
    df = df_in.copy()
    for c in cols:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            df[c] = s.dt.strftime("%Y-%m-%d").where(s.notna(), df[c])
    return df

# =============== Sidebar ===============
st.sidebar.header("View")
page = st.sidebar.radio("Pick a page", ["Home", "Insights"], index=0)

st.sidebar.header("Zone Sensitivity")
near_band_pp = st.sidebar.number_input(
    "Near-band (± percentage points)", min_value=0.1, max_value=10.0, step=0.1, value=1.0
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
    perf = base.merge(win_mean, on="symbol", how="left").merge(loss_mean, on="symbol", how="left")
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
#                 HOME (Open trades)
# =========================================
if page == "Home":
    st.subheader("🔓 Open Trades")

    open_trades = trades[trades["outcome"] == 0].copy()
    if open_trades.empty:
        st.info("No open trades.")
    else:
        # attach historical %s for price targets
        open_trades["avg_return"]      = open_trades["symbol"].map(avg_return_map)
        open_trades["avg_win_return"]  = open_trades["symbol"].map(avg_win_ret_map)
        open_trades["avg_loss_return"] = open_trades["symbol"].map(avg_loss_ret_map)

        # implied price levels
        open_trades["guard_loss_price"]   = open_trades.apply(
            lambda r: r["entry"] * (1 + r["avg_loss_return"]/100.0)
            if pd.notna(r["avg_loss_return"]) else pd.NA, axis=1
        )
        open_trades["first_target_price"] = open_trades.apply(
            lambda r: r["entry"] * (1 + r["avg_return"]/100.0)
            if pd.notna(r["avg_return"]) else pd.NA, axis=1
        )
        open_trades["win_target_price"]   = open_trades.apply(
            lambda r: r["entry"] * (1 + r["avg_win_return"]/100.0)
            if pd.notna(r["avg_win_return"]) else pd.NA, axis=1
        )

        # --- REVISED ZONE LOGIC (uses price vs targets) ---
        # 🟩 if latest_close >= 2nd target (within band below or above)
        # 🟥 if latest_close <= guard (within band above or below)
        # 🟧 if near 1st target (within band)
        # ◻️ otherwise
        def zone_emoji_by_price(r):
            price = r["latest_close"]
            t2 = r["win_target_price"]
            g  = r["guard_loss_price"]
            t1 = r["first_target_price"]
            band = near_band_pp / 100.0  # convert pp to fraction

            if pd.notna(price) and pd.notna(t2):
                if price >= t2 * (1 - band):   # near or above 2nd target
                    return "🟩"
            if pd.notna(price) and pd.notna(g):
                if price <= g * (1 + band):   # near or below guard
                    return "🟥"
            if pd.notna(price) and pd.notna(t1):
                if abs(price - t1) / t1 <= band:  # around 1st target
                    return "🟧"
            return "◻️"

        open_trades["zone"] = open_trades.apply(zone_emoji_by_price, axis=1)

        # sort newest entries first, then by cap_score (lower first)
        open_trades = open_trades.sort_values(["entry_date", "cap_score"], ascending=[False, True])

        # assemble display
        show = open_trades.loc[:, [
            "zone",
            "symbol_display","sector","entry_date",
            "entry","guard_loss_price","first_target_price","win_target_price",
            "latest_close"
        ]].copy()

        # formatting
        show = date_only_cols(show, ["entry_date"])
        for c in ["entry","guard_loss_price","first_target_price","win_target_price","latest_close"]:
            show[c] = show[c].map(money_str)

        # rename headers to requested labels
        show = show.rename(columns={
            "symbol_display": "Symbol",
            "sector": "Sector",
            "entry_date": "Entry date",
            "entry": "Entry",
            "guard_loss_price": "Guard",
            "first_target_price": "1st target",
            "win_target_price": "2nd target",
            "latest_close": "Latest close",
            "zone": " "  # minimal header for the emoji
        })

        st.dataframe(show, use_container_width=True, hide_index=True)

        # --- Legends (compact, below the table) ---
        st.markdown("**Zone legend**")
        st.caption(
            f"🟩 near/above **2nd target** (Avg Win %)  •  "
            f"🟥 near/below **Guard** (Avg Loss %)  •  "
            f"🟧 near **1st target** (Avg Return %)  •  "
            f"◻️ not near any  "
            f"(band ±{near_band_pp:.1f} pp on price vs target)"
        )

        cap_legend_df = (
            open_trades[["cap_emoji","cap_score"]]
            .dropna()
            .drop_duplicates()
            .sort_values("cap_score")
        )
        if not cap_legend_df.empty:
            parts = [f"{r.cap_emoji} — score {int(r.cap_score)}" for r in cap_legend_df.itertuples(index=False)]
            st.markdown("**Cap legend**")
            st.caption(" | ".join(parts))

# =========================================
#               INSIGHTS (Closed/analytics)
# =========================================
else:
    # ---- Full trades export
    st.subheader("📦 Download All Trades")
    all_trades_to_export = trades.sort_values("entry_date", ascending=False)[[
        "symbol_display","cap_score","sector","entry_date","entry","outcome",
        "exit_price","exit_date","stop_loss","min_low","max_high","final_pct"
    ]]
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
        show_df = date_only_cols(show_df, ["entry_date","exit_date"])
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
