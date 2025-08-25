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
page = st.sidebar.radio("Pick a page", ["Home", "Insights", "Tester", "Optimizer"], index=0)


st.sidebar.header("Zone Sensitivity")
near_band_pp = st.sidebar.number_input(
    "Near-band (± percentage points)", min_value=0.1, max_value=10.0, step=0.1, value=1.0
)

# --- Strategy (2025+ guards) tuning ---
st.sidebar.header("Strategy (2025+ guards)")

use_enhanced = st.sidebar.checkbox(
    "Enable enhanced exits for 2025+ entries", value=True, key="guard_enable"
)

cutoff_date_ui = st.sidebar.date_input(
    "Enhanced cutoff date",
    value=pd.Timestamp(2025, 1, 1).date(),
    key="guard_cutoff",
)

guard_buffer_pp = st.sidebar.number_input(
    "Guard buffer (percentage points)", min_value=0.0, max_value=5.0, step=0.25,
    value=0.75, key="guard_buf"
)

guard_confirm_bars = st.sidebar.number_input(
    "Confirmation bars (beyond threshold)", min_value=1, max_value=5, step=1,
    value=2, key="guard_conf"
)

min_hold_bars = st.sidebar.number_input(
    "Minimum hold bars before any guard can fire", min_value=0, max_value=10, step=1,
    value=2, key="guard_minhold"
)

profit_mode = st.sidebar.radio(
    "Profit guard mode", ["Avg-win re-cross", "Peak giveback"],
    index=0, key="guard_mode"
)

# Only show the giveback % when "Peak giveback" is selected
if profit_mode == "Peak giveback":
    profit_trail_peak_dd = st.sidebar.number_input(
        "Giveback from peak (%)", min_value=0.5, max_value=10.0, step=0.5,
        value=3.0, key="guard_giveback"
    )
else:
    profit_trail_peak_dd = None

# =============== Load & Run Strategy ===============
df, caps = load_data()

# ---- 1) Seed run (no guards) -> build avg win/loss maps from *all* closed trades
with st.spinner("⏳ Detecting trades (seed)…"):
    trades_seed = run_strategy(df, caps)

# Build maps from seed closed trades
closed_seed = trades_seed[trades_seed["exit_date"].notna()].copy()
if not closed_seed.empty:
    closed_seed["pct_return"] = (closed_seed["exit_price"] / closed_seed["entry"] - 1) * 100
    avg_win_map  = (closed_seed.loc[closed_seed["pct_return"] > 0]
                               .groupby("symbol")["pct_return"].mean().to_dict())
    avg_loss_map = (closed_seed.loc[closed_seed["pct_return"] < 0]
                               .groupby("symbol")["pct_return"].mean().to_dict())
else:
    avg_win_map, avg_loss_map = {}, {}

# ---- 2) Main run (optionally enhanced for 2025+ using the maps & sidebar knobs)
if use_enhanced:
    with st.spinner("⏳ Detecting trades (enhanced 2025+ guards)…"):
        trades = run_strategy(
            df, caps,
            avg_win_map=avg_win_map,
            avg_loss_map=avg_loss_map,
            enhanced_cutoff=str(pd.Timestamp(cutoff_date_ui).date()),  # "YYYY-MM-DD"
            guard_buffer_pp=float(guard_buffer_pp),
            guard_confirm_bars=int(guard_confirm_bars),
            min_hold_bars=int(min_hold_bars),
            # Pass giveback % only when using Peak giveback mode; otherwise None = disabled
            profit_trail_peak_dd=float(profit_trail_peak_dd) if profit_trail_peak_dd is not None else None,
        )
else:
    with st.spinner("⏳ Detecting trades…"):
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
if page == "Insights":
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
    st.subheader("🏆 Top Tickers by Avg Return (Closed Trades)")
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

        best = best.sort_values("avg_return", ascending=False)

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
            "📥 Download (Avg Return + Win/Loss)",
            best.loc[:, ["symbol","sector","n_trades","avg_return","avg_win_return","avg_loss_return","avg_days"]]
                .rename(columns={
                    "avg_return": "avg_return_pct",
                    "avg_win_return": "avg_win_return_pct",
                    "avg_loss_return": "avg_loss_return_pct",
                    "avg_days": "avg_days_held"
                })
                .to_csv(index=False).encode("utf-8"),
            "top_avg_return_with_win_loss.csv","text/csv"
        )
    else:
        st.info("Not enough closed trades yet to compute historical leaders.")
# =========================================
#               TESTER
# =========================================
if page == "Tester":
    st.subheader("🧪 Strategy Tester (one position at a time)")

    # ---------- Sidebar inputs ----------
    st.sidebar.header("Tester Settings")
    min_entry = pd.to_datetime(trades["entry_date"]).min()
    max_date  = pd.to_datetime(df["date"]).max()

    start_date = st.sidebar.date_input(
        "Start date",
        value=min_entry.date() if pd.notna(min_entry) else pd.Timestamp.today().date(),
        min_value=(min_entry.date() if pd.notna(min_entry) else pd.Timestamp.today().date()),
        key="tester_start_date"
    )
    end_date = st.sidebar.date_input(
        "End date",
        value=max_date.date() if pd.notna(max_date) else pd.Timestamp.today().date(),
        min_value=start_date,
        key="tester_end_date"
    )

    starting_capital = st.sidebar.number_input(
        "Starting capital ($)", min_value=1000.0, step=500.0, value=10000.0, key="tester_capital"
    )
    alloc_pct = st.sidebar.number_input(
        "Allocation per trade (% of capital)", min_value=1.0, max_value=100.0, step=1.0, value=100.0, key="tester_alloc"
    )

    # ---------- Selection mode ----------
    st.sidebar.subheader("Selection mode")
    selection_mode = st.sidebar.radio(
        "Trade selection mode",
        ["By entries", "By tickers", "Top-N (avg return)", "Top-N (avg win return)"],
        index=0,
        key="tester_mode"
    )

    # ---------- Candidates in window ----------
    start_ts = pd.Timestamp(start_date)
    end_ts   = pd.Timestamp(end_date)

    candidates = trades[(trades["entry_date"] >= start_ts) & (trades["entry_date"] <= end_ts)].copy()
    candidates = candidates.sort_values("entry_date")
    if candidates.empty:
        st.info("No strategy entries within the selected window.")
        st.stop()

    candidates["label"] = candidates.apply(
        lambda r: f"{r['symbol']} — {pd.to_datetime(r['entry_date']).date()}",
        axis=1
    )

    # ---------- Build chosen set according to mode ----------
    chosen = pd.DataFrame(columns=candidates.columns)

    if selection_mode == "By entries":
        default_selection = candidates["label"].tolist()
        chosen_labels = st.sidebar.multiselect(
            "Choose entries (chronological; overlapping ones are skipped automatically)",
            default_selection,
            default=default_selection,
            key="tester_entries"
        )
        chosen = candidates[candidates["label"].isin(chosen_labels)].copy()

    elif selection_mode == "By tickers":
        syms_avail = sorted(candidates["symbol"].unique().tolist())
        chosen_syms = st.sidebar.multiselect(
            "Choose tickers (all their entries inside window will be considered)",
            syms_avail, default=syms_avail, key="tester_tickers"
        )
        chosen = candidates[candidates["symbol"].isin(chosen_syms)].copy()

    elif selection_mode == "Top-N (avg return)":
        if perf.empty:
            st.warning("No historical performance available (no closed trades).")
            st.stop()

        N = st.sidebar.number_input("Top-N by avg return", min_value=1, max_value=50, step=1, value=5, key="tester_topn")
        min_closed = st.sidebar.number_input("Min # closed trades", min_value=1, max_value=50, step=1, value=1, key="tester_minclosed")

        perf_rank = (perf.loc[perf["n_closed"] >= min_closed]
                        .dropna(subset=["avg_return"])
                        .sort_values("avg_return", ascending=False))

        if perf_rank.empty:
            st.warning("No tickers meet the minimum closed-trade filter.")
            st.stop()

        syms_in_window = set(candidates["symbol"].unique().tolist())
        ranked_syms = [s for s in perf_rank["symbol"].tolist() if s in syms_in_window][:int(N)]

        if not ranked_syms:
            st.warning("Top-N ranked tickers have no entries in the selected window.")
            st.stop()

        st.caption("Selected top performers (by historical **avg return**):")
        preview = perf_rank.loc[perf_rank["symbol"].isin(ranked_syms),
                                ["symbol","avg_return","win_rate","n_closed"]].copy()
        preview["avg_return"] = preview["avg_return"].map(lambda x: pct_str(x))
        preview["win_rate"]   = preview["win_rate"].map(lambda x: "—" if pd.isna(x) else f"{x:.0%}")
        st.dataframe(add_rownum(preview).rename(columns={
            "symbol":"Ticker","avg_return":"Avg return","win_rate":"Win rate","#":"#","n_closed":"Closed"
        }), use_container_width=True, hide_index=True)

        chosen = candidates[candidates["symbol"].isin(ranked_syms)].copy()

    else:  # Top-N (avg win return)
        if perf.empty:
            st.warning("No historical performance available (no closed trades).")
            st.stop()

        N = st.sidebar.number_input("Top-N by avg **win** return", min_value=1, max_value=50, step=1, value=5, key="tester_topn_win")
        min_closed = st.sidebar.number_input("Min # closed trades", min_value=1, max_value=50, step=1, value=1, key="tester_minclosed_win")

        perf_win_rank = (perf.loc[perf["n_closed"] >= min_closed]
                            .dropna(subset=["avg_win_return"])
                            .sort_values("avg_win_return", ascending=False))

        if perf_win_rank.empty:
            st.warning("No tickers meet the minimum closed-trade filter for avg win return.")
            st.stop()

        syms_in_window = set(candidates["symbol"].unique().tolist())
        ranked_syms = [s for s in perf_win_rank["symbol"].tolist() if s in syms_in_window][:int(N)]

        if not ranked_syms:
            st.warning("Top-N (avg win) ranked tickers have no entries in the selected window.")
            st.stop()

        st.caption("Selected top performers (by historical **avg win return**):")
        preview = perf_win_rank.loc[perf_win_rank["symbol"].isin(ranked_syms),
                                    ["symbol","avg_win_return","win_rate","n_closed"]].copy()
        preview["avg_win_return"] = preview["avg_win_return"].map(lambda x: pct_str(x))
        preview["win_rate"]       = preview["win_rate"].map(lambda x: "—" if pd.isna(x) else f"{x:.0%}")
        st.dataframe(add_rownum(preview).rename(columns={
            "symbol":"Ticker","avg_win_return":"Avg win return","win_rate":"Win rate","#":"#","n_closed":"Closed"
        }), use_container_width=True, hide_index=True)

        chosen = candidates[candidates["symbol"].isin(ranked_syms)].copy()

    # ---------- Last close per symbol up to end date ----------
    last_close_upto = (
        df[df["date"] <= end_ts].sort_values("date")
          .groupby("symbol", as_index=False)
          .agg(last_close_upto=("close", "last"))
          .set_index("symbol")["last_close_upto"]
          .to_dict()
    )
    latest_fallback = (
        df.sort_values("date").groupby("symbol", as_index=False)
          .agg(latest_close=("close","last"))
          .set_index("symbol")["latest_close"]
          .to_dict()
    )
    def _last_close_for(symbol):
        v = last_close_upto.get(symbol)
        if pd.isna(v) or v is None:
            return latest_fallback.get(symbol)
        return v

    # ---------- Run simulation ----------
    capital = float(starting_capital)
    available_from = start_ts
    ledger = []

    for _, r in chosen.sort_values("entry_date").iterrows():
        entry_d = pd.to_datetime(r["entry_date"])
        if entry_d < available_from:
            continue

        sym   = r["symbol"]
        entry = float(r["entry"])
        ex_d  = pd.to_datetime(r["exit_date"]) if pd.notna(r["exit_date"]) else pd.NaT
        realized = pd.notna(ex_d) and (ex_d <= end_ts)

        if realized:
            exit_px = float(r["exit_price"])
            ret_pct = (exit_px / entry - 1.0) * 100.0
            exit_d  = ex_d
            status  = "Realized"
            available_from = ex_d + pd.Timedelta(days=1)
        else:
            lc = _last_close_for(sym)
            if lc is None or pd.isna(lc):
                continue
            exit_px = float(lc)
            ret_pct = (exit_px / entry - 1.0) * 100.0
            exit_d  = pd.NaT
            status  = "Unrealized"
            available_from = end_ts + pd.Timedelta(days=1)

        cap_before   = capital
        invest_amt   = capital * (alloc_pct / 100.0)
        cash_amt     = capital - invest_amt
        invest_after = invest_amt * (1.0 + ret_pct/100.0)
        capital      = cash_amt + invest_after

        days_held = (exit_d.normalize() - entry_d.normalize()).days if pd.notna(exit_d) \
                    else (end_ts.normalize() - entry_d.normalize()).days

        ledger.append({
            "symbol": sym,
            "entry_date": entry_d,
            "exit_date": exit_d,
            "status": status,
            "entry": entry,
            "exit_or_last": exit_px,
            "ret_pct": ret_pct,
            "days_held": days_held,
            "capital_before": cap_before,
            "capital_after": capital,
        })

    if not ledger:
        st.info("No trades were taken (either none selected, or all overlapped with an active position).")
        st.stop()

    res = pd.DataFrame(ledger).sort_values("entry_date").reset_index(drop=True)

    # KPIs
    n_trades = len(res)
    n_real   = (res["status"] == "Realized").sum()
    total_ret = (capital/starting_capital - 1.0)*100.0
    win_rate = (res.loc[res["status"] == "Realized", "ret_pct"] > 0).mean() if n_real else float("nan")
    avg_real = res.loc[res["status"] == "Realized", "ret_pct"].mean() if n_real else float("nan")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Trades taken", f"{n_trades}")
    c2.metric("Final capital", f"${capital:,.2f}")
    c3.metric("Total return", pct_str(total_ret))
    c4.metric("Realized win rate", "—" if pd.isna(win_rate) else f"{win_rate:.0%}")
    c5.metric("Avg realized return", "—" if pd.isna(avg_real) else pct_str(avg_real))

    out = res.copy()
    out = date_only_cols(out, ["entry_date","exit_date"])
    out["entry"]         = out["entry"].map(money_str)
    out["exit_or_last"]  = out["exit_or_last"].map(money_str)
    out["ret_pct"]       = out["ret_pct"].map(lambda x: pct_str(x))
    out["capital_before"]= out["capital_before"].map(money_str)
    out["capital_after"] = out["capital_after"].map(money_str)

    display_cols = ["symbol","status","entry_date","exit_date","entry","exit_or_last",
                    "ret_pct","days_held","capital_before","capital_after"]
    st.subheader("📜 Trade Ledger (simulated)")
    st.dataframe(add_rownum(out.loc[:, display_cols]).rename(columns={
        "exit_or_last": "Exit / Last",
        "ret_pct": "Return",
    }), use_container_width=True, hide_index=True)

    st.caption("Rules: one position at a time; overlapping selected entries are skipped. "
               "Unrealized P/L uses the last available close at or before the selected end date. "
               "Top-N can rank by historical avg return OR avg win return (from closed trades), "
               "and only tickers with entries in the chosen window are considered.")
    
    # =========================================# =========================================
#                 OPTIMIZER
# =========================================
if page == "Optimizer":
    st.subheader("🧪 Parameter Optimizer — Peak-Giveback (one position at a time)")

    st.sidebar.header("Optimizer Settings")

    # ---- Date window (only entries INSIDE this window are simulated)
    min_entry = pd.to_datetime(trades["entry_date"]).min()
    max_date  = pd.to_datetime(df["date"]).max()
    start_date = st.sidebar.date_input(
        "Start date",
        value=max(pd.to_datetime("2025-01-01").date(), (min_entry.date() if pd.notna(min_entry) else pd.Timestamp.today().date()))
    )
    end_date = st.sidebar.date_input(
        "End date",
        value=max_date.date() if pd.notna(max_date) else pd.Timestamp.today().date(),
        min_value=start_date
    )

    # ---- Capital & allocation for the simulation
    starting_capital = st.sidebar.number_input("Starting capital ($)", min_value=1000.0, step=500.0, value=10000.0)
    alloc_pct = st.sidebar.number_input("Allocation per trade (% of capital)", min_value=1.0, max_value=100.0, step=1.0, value=100.0)

    # ---- Enhanced exits cutoff (keep at 2025-01-01 unless you want to compare)
    cutoff_default = pd.to_datetime("2025-01-01").date()
    enhanced_cutoff = st.sidebar.date_input("Enhanced exits from date", value=cutoff_default)

    # ---- Universe selection: All vs Top-N by avg return
    st.sidebar.subheader("Universe")
    universe_mode = st.sidebar.radio(
        "Which symbols should the optimizer test?",
        ["All tickers", "Top-N by avg return"],
        index=1,  # default to Top-N per your plan
        key="opt_universe_mode"
    )

    topN = None
    min_closed_for_rank = None
    if universe_mode == "Top-N by avg return":
        topN = int(st.sidebar.number_input("Top-N (avg return)", min_value=1, max_value=50, step=1, value=5, key="opt_topN"))
        min_closed_for_rank = int(st.sidebar.number_input("Min # closed trades", min_value=1, max_value=50, step=1, value=1, key="opt_min_closed"))

    # ---- Grid choices (Peak, Confirm, Min-hold). Buffer fixed at 1.0
    st.sidebar.markdown("**Grid — what to try**")
    peak_choices = st.sidebar.multiselect("Peak giveback (%)", [6, 7, 8, 9, 10, 12], default=[8, 9, 10], key="opt_peaks")
    confirm_choices = st.sidebar.multiselect("Confirm bars (1..3)", [1, 2, 3], default=[2, 3], key="opt_confirms")
    min_hold_choices = st.sidebar.multiselect("Min hold bars (2 or 3)", [2, 3], default=[3], key="opt_minholds")

    buffer_mult = 1.0                # fixed (don’t over-iterate)
    use_median_maps = True           # more robust average for guards
    # We are NOT using avg-win guard in the optimizer loop; only peak-giveback.

    run_btn = st.sidebar.button("🚀 Run optimization")

    # ---------- Helpers ----------
    def _build_last_close_dicts(end_ts: pd.Timestamp):
        last_close_upto = (
            df[df["date"] <= end_ts].sort_values("date")
              .groupby("symbol", as_index=False)
              .agg(last_close_upto=("close", "last"))
              .set_index("symbol")["last_close_upto"]
              .to_dict()
        )
        latest_fallback = (
            df.sort_values("date").groupby("symbol", as_index=False)
              .agg(latest_close=("close","last"))
              .set_index("symbol")["latest_close"]
              .to_dict()
        )
        return last_close_upto, latest_fallback

    def _last_close_for(symbol, last_close_upto, latest_fallback):
        v = last_close_upto.get(symbol)
        if pd.isna(v) or v is None:
            return latest_fallback.get(symbol)
        return v

    # One-position-at-a-time simulation (same logic as Tester)
    def simulate_one_position_at_a_time(trades_df, start_ts, end_ts, alloc_pct, last_close_upto, latest_fallback):
        capital = float(starting_capital)
        available_from = start_ts
        ledger = []

        # only entries inside the window
        chosen = trades_df[(trades_df["entry_date"] >= start_ts) & (trades_df["entry_date"] <= end_ts)].copy()
        if chosen.empty:
            return capital, pd.DataFrame(ledger)

        for _, r in chosen.sort_values("entry_date").iterrows():
            entry_d = pd.to_datetime(r["entry_date"])
            if entry_d < available_from:
                continue

            sym   = r["symbol"]
            entry = float(r["entry"])
            ex_d  = pd.to_datetime(r["exit_date"]) if pd.notna(r["exit_date"]) else pd.NaT
            realized = pd.notna(ex_d) and (ex_d <= end_ts)

            if realized:
                exit_px = float(r["exit_price"])
                ret_pct = (exit_px / entry - 1.0) * 100.0
                exit_d  = ex_d
                status  = "Realized"
                available_from = ex_d + pd.Timedelta(days=1)
            else:
                lc = _last_close_for(sym, last_close_upto, latest_fallback)
                if lc is None or pd.isna(lc):
                    continue
                exit_px = float(lc)
                ret_pct = (exit_px / entry - 1.0) * 100.0
                exit_d  = pd.NaT
                status  = "Unrealized"
                available_from = end_ts + pd.Timedelta(days=1)

            cap_before   = capital
            invest_amt   = capital * (alloc_pct / 100.0)
            cash_amt     = capital - invest_amt
            invest_after = invest_amt * (1.0 + ret_pct/100.0)
            capital      = cash_amt + invest_after

            days_held = (exit_d.normalize() - entry_d.normalize()).days if pd.notna(exit_d) \
                        else (end_ts.normalize() - entry_d.normalize()).days

            ledger.append({
                "symbol": sym,
                "entry_date": entry_d,
                "exit_date": exit_d,
                "status": status,
                "entry": entry,
                "exit_or_last": exit_px,
                "ret_pct": ret_pct,
                "days_held": days_held,
                "capital_after": capital,
            })

        return capital, pd.DataFrame(ledger)

    if run_btn:
        if not peak_choices or not confirm_choices or not min_hold_choices:
            st.warning("Pick at least one value for each of: Peak, Confirm bars, Min hold.")
            st.stop()

        start_ts = pd.Timestamp(start_date)
        end_ts   = pd.Timestamp(end_date)
        cutoff_ts = pd.Timestamp(enhanced_cutoff)

        # ---- 1) Build the Universe we will test ----
        # We’ll use `trades_seed` for “who has entries in the window?”, then intersect with Top-N if selected.
        seed_window = trades_seed[
            (pd.to_datetime(trades_seed["entry_date"]) >= start_ts) &
            (pd.to_datetime(trades_seed["entry_date"]) <= end_ts)
        ].copy()

        if seed_window.empty:
            st.info("No strategy entries (seed) inside the selected window — widen dates.")
            st.stop()

        if universe_mode == "All tickers":
            universe_syms = sorted(seed_window["symbol"].unique().tolist())
            st.caption(f"Universe: All symbols with entries in window ({len(universe_syms)} tickers).")
        else:
            if perf.empty:
                st.warning("No historical performance available (no closed trades) to rank Top-N.")
                st.stop()

            perf_rank = (
                perf.loc[perf["n_closed"] >= min_closed_for_rank]
                    .dropna(subset=["avg_return"])
                    .sort_values("avg_return", ascending=False)
            )
            if perf_rank.empty:
                st.warning("No tickers meet the minimum closed-trade filter.")
                st.stop()

            syms_in_window = set(seed_window["symbol"].unique().tolist())
            ranked_syms = [s for s in perf_rank["symbol"].tolist() if s in syms_in_window][:topN]
            if not ranked_syms:
                st.warning("Top-N ranked tickers have no entries in the selected window.")
                st.stop()

            # Show who we picked
            st.caption("Selected Top-N (by historical avg return):")
            preview = perf_rank.loc[perf_rank["symbol"].isin(ranked_syms),
                                    ["symbol","avg_return","win_rate","n_closed"]].copy()
            preview["avg_return"] = preview["avg_return"].map(lambda x: pct_str(x))
            preview["win_rate"]   = preview["win_rate"].map(lambda x: "—" if pd.isna(x) else f"{x:.0%}")
            st.dataframe(add_rownum(preview).rename(columns={
                "symbol":"Ticker","avg_return":"Avg return","win_rate":"Win rate",
                "#":"#","n_closed":"Closed"
            }), use_container_width=True, hide_index=True)

            universe_syms = ranked_syms

        # ---- 2) Price lookups once (for unrealized P/L)
        last_close_upto, latest_fallback = _build_last_close_dicts(end_ts)

        # ---- 3) Grid search over Peak/Confirm/Min-hold (peak-giveback only)
        results = []
        grid_total = len(peak_choices) * len(confirm_choices) * len(min_hold_choices)
        prog = st.progress(0)
        step = 0

        for peak in sorted(set(peak_choices)):
            for confirm in sorted(set(confirm_choices)):
                for min_hold in sorted(set(min_hold_choices)):
                    step += 1
                    prog.progress(min(step / grid_total, 1.0))

                    trades_combo = run_strategy(
                        df, caps,
                        # maps from SEED run (already computed above)
                        avg_win_map=avg_win_map,
                        avg_loss_map=avg_loss_map,
                        # Enhanced exits only for entries on/after cutoff
                        enhanced_cutoff=cutoff_ts.strftime("%Y-%m-%d"),
                        enhanced_guards_on=True,
                        # We are using PEAK-GIVEBACK guard only:
                        require_confirm_bars=int(confirm),
                        min_hold_bars=int(min_hold),
                        buffer_mult=float(buffer_mult),
                        peak_giveback_pct=float(peak),
                        maps_use_median=use_median_maps,
                        # avg-win guard is OFF by passing no threshold param (or leave as implemented in strategy)
                    )

                    if trades_combo.empty:
                        continue

                    # Filter to UNIVERSE and DATE WINDOW before sim
                    trades_combo = trades_combo[trades_combo["symbol"].isin(universe_syms)].copy()
                    if trades_combo.empty:
                        continue

                    final_cap, ledger = simulate_one_position_at_a_time(
                        trades_combo, start_ts, end_ts, alloc_pct, last_close_upto, latest_fallback
                    )

                    # Metrics
                    n_trades = len(ledger)
                    realized_mask = (ledger["status"] == "Realized") if n_trades else pd.Series([], dtype=bool)
                    n_real = int(realized_mask.sum()) if n_trades else 0
                    win_rate = (ledger.loc[realized_mask, "ret_pct"] > 0).mean() if n_real else float("nan")
                    avg_real = ledger.loc[realized_mask, "ret_pct"].mean() if n_real else float("nan")
                    best_real = ledger.loc[realized_mask, "ret_pct"].max() if n_real else float("nan")
                    worst_real = ledger.loc[realized_mask, "ret_pct"].min() if n_real else float("nan")
                    total_ret_pct = (final_cap / starting_capital - 1.0) * 100.0

                    results.append({
                        "Peak %": peak,
                        "Confirm": confirm,
                        "Min hold": min_hold,
                        "Final capital": final_cap,
                        "Total return %": total_ret_pct,
                        "Trades": n_trades,
                        "Realized": n_real,
                        "Win rate": win_rate,
                        "Avg realized %": avg_real,
                        "Best %": best_real,
                        "Worst %": worst_real,
                        "ledger": ledger
                    })

        if not results:
            st.info("No valid trades found for the selected grid / window / universe.")
            st.stop()

        # ---- 4) Show results
        res_df = pd.DataFrame(results).sort_values("Final capital", ascending=False).reset_index(drop=True)

        res_show = res_df.drop(columns=["ledger"]).copy()
        res_show["Final capital"]   = res_show["Final capital"].map(money_str)
        res_show["Total return %"]  = res_show["Total return %"].map(lambda x: pct_str(x))
        res_show["Win rate"]        = res_show["Win rate"].map(lambda x: "—" if pd.isna(x) else f"{x:.0%}")
        for c in ["Avg realized %", "Best %", "Worst %"]:
            res_show[c] = res_show[c].map(lambda x: pct_str(x))
        st.subheader("🏆 Grid Results (sorted by Final capital)")
        st.dataframe(add_rownum(res_show), use_container_width=True, hide_index=True)

        best = res_df.iloc[0]
        st.markdown(
            f"**Best combo:** Peak={best['Peak %']}%, Confirm={best['Confirm']}, Min hold={best['Min hold']}  "
            f"• Final capital: {money_str(best['Final capital'])}  "
            f"• Total return: {pct_str(best['Total return %'])}  "
            f"• Win rate: {'—' if pd.isna(best['Win rate']) else f'{best['Win rate']:.0%}'}"
        )

        st.subheader("📜 Best Combo — Trade Ledger")
        ledger = best["ledger"].copy()
        if not ledger.empty:
            ledger = date_only_cols(ledger, ["entry_date","exit_date"])
            ledger["entry"]        = ledger["entry"].map(money_str)
            ledger["exit_or_last"] = ledger["exit_or_last"].map(money_str)
            ledger["ret_pct"]      = ledger["ret_pct"].map(lambda x: pct_str(x))
            ledger["capital_after"]= ledger["capital_after"].map(money_str)
            st.dataframe(add_rownum(ledger.rename(columns={
                "exit_or_last": "Exit / Last",
                "ret_pct": "Return",
            })), use_container_width=True, hide_index=True)

        st.caption(
            "Notes: Optimizer uses the peak-giveback enhanced exits only for entries on/after the chosen cutoff. "
            "Top-N is ranked by historical avg return from closed trades (the 'Insights' perf table), "
            "then intersected with symbols that actually have entries in your date window (based on the seed run). "
            "Simulation assumes one position at a time with compounding."
        )
    else:
        st.info("Pick your grid and click **Run optimization** to benchmark parameter combinations.")
