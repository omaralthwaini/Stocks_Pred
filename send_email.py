# send_email.py
import os
import smtplib
from email.message import EmailMessage
from datetime import datetime
import pytz
import numpy as np
import pandas as pd

from strategy import run_strategy
from arch import arch_model  # GARCH

# ==============================
# Formatting helpers
# ==============================
def fmt_money(x):
    return f"${x:,.2f}" if pd.notna(x) else "—"

def fmt_pct_signed(x, digits=2):
    return f"{x:+.{digits}f}%" if pd.notna(x) else "—"

def fmt_pct_plain(x, digits=0):
    return f"{x:.{digits}f}%" if pd.notna(x) else "—"

def date_only(s):
    try:
        s = pd.to_datetime(s)
        return s.strftime("%Y-%m-%d") if pd.notna(s) else "—"
    except Exception:
        return "—"

# Shared page styles (HTML)
PAGE_CSS = """
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
  .wrap { max-width: 980px; margin: 0 auto; }
  .kpis { display:flex; gap:12px; flex-wrap:wrap; margin: 8px 0 14px; }
  .chip { border:1px solid #eee; border-radius:8px; padding:8px 10px; background:#fafafa; }
  h2 { margin: 18px 0 10px; font-size: 18px; }
  h3 { margin: 16px 0 8px; font-size: 16px; }
  .muted { color:#666; }
  .foot { color:#888; font-size:12px; margin-top:14px; }
  table { border-collapse: collapse; width: 100%; margin: 8px 0 16px; }
  th, td { text-align: left; padding: 8px; font-size: 13px; }
  thead th { border-bottom: 2px solid #ddd; background:#f6f8fa; }
  tbody tr { border-bottom: 1px solid #eee; }
  .pos { color: #0a7a0a; font-weight: 600; }
  .neg { color: #c23232; font-weight: 600; }
</style>
"""

# ==============================
# Email helpers
# ==============================
def send_email(subject, body_text, body_html):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ["EMAIL_USER"]
    msg["To"] = os.environ["EMAIL_TO"]
    msg.set_content(body_text)
    msg.add_alternative(body_html, subtype="html")

    with smtplib.SMTP(os.environ["EMAIL_SMTP_HOST"], int(os.environ["EMAIL_SMTP_PORT"])) as server:
        server.starttls()
        server.login(os.environ["EMAIL_USER"], os.environ["EMAIL_PASS"])
        server.send_message(msg)
        print("✅ Email sent successfully.")

# ==============================
# GARCH + OAAT (same logic as app)
# ==============================
def garch_volatility_forecast(series: pd.Series) -> float | None:
    px = pd.Series(series).astype(float)
    px = px.replace([np.inf, -np.inf], np.nan).dropna()
    px = px[px > 0]
    rets = np.log(px / px.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(rets) < 50:
        return None
    try:
        am = arch_model(rets, vol="GARCH", p=1, q=1, mean="Zero", dist="t")
        res = am.fit(disp="off")
        fc = res.forecast(horizon=1, reindex=False)
        var1 = float(fc.variance.iloc[-1, 0])
        ann_vol = float(np.sqrt(var1) * np.sqrt(252))
        return ann_vol if np.isfinite(ann_vol) and ann_vol > 0 else None
    except Exception:
        return None

def attach_garch_risk_index(df_prices: pd.DataFrame,
                            trades_df: pd.DataFrame,
                            base_vol_dec: float = 0.20,
                            lookback: int | None = 252) -> pd.DataFrame:
    df_sorted = df_prices.sort_values(["symbol", "date"])
    groups = {sym: g[["date","close"]].reset_index(drop=True)
              for sym, g in df_sorted.groupby("symbol")}
    out = []
    for _, r in trades_df.iterrows():
        sym = r["symbol"]
        entry_date = pd.to_datetime(r["entry_date"])
        if sym not in groups:
            out.append(np.nan); continue
        g = groups[sym]
        hist = g[g["date"] < entry_date]["close"]
        if lookback and lookback > 0:
            hist = hist.tail(lookback)
        if len(hist) < 51:
            out.append(np.nan); continue
        ann_vol = garch_volatility_forecast(hist)
        if ann_vol is None or not np.isfinite(ann_vol) or ann_vol <= 1e-9:
            out.append(np.nan); continue
        out.append(float(min(1.0, base_vol_dec / ann_vol)))
    trades2 = trades_df.copy()
    trades2["garch_risk_index"] = out
    return trades2

def simulate_oaat(
    trades_in: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    starting_capital: float = 10000.0,
    alloc_pct: float = 100.0,
    *,
    threshold: float = 0.50,
    require_garch: bool = True
):
    t = trades_in.copy().sort_values("entry_date")
    if require_garch:
        t = t[t["garch_risk_index"].notna()]
    else:
        t["garch_risk_index"] = t["garch_risk_index"].fillna(1.0)
    t = t[t["garch_risk_index"].ge(threshold)]

    capital = float(starting_capital)
    available_from = start_ts
    taken = []

    for _, r in t.iterrows():
        entry_d = pd.to_datetime(r["entry_date"])
        if entry_d < available_from:
            continue
        entry = float(r["entry"])
        ex_d  = pd.to_datetime(r["exit_date"]) if pd.notna(r["exit_date"]) else pd.NaT
        realized = pd.notna(ex_d) and (ex_d <= end_ts)
        if realized:
            exit_px = float(r["exit_price"])
            ret_pct = (exit_px / entry - 1.0) * 100.0
            available_from = ex_d + pd.Timedelta(days=1)
        else:
            exit_px = float(r["latest_close"])
            ret_pct = (exit_px / entry - 1.0) * 100.0
            available_from = end_ts + pd.Timedelta(days=1)
        invest_amt = capital * (alloc_pct / 100.0)
        capital = capital - invest_amt + invest_amt * (1.0 + ret_pct / 100.0)
        taken.append({"entry_date": entry_d, "ret_pct": ret_pct})
    n_real = len(taken)
    win_rate = (pd.Series([r["ret_pct"] for r in taken]) > 0).mean() if n_real else float("nan")
    total_ret = (capital / starting_capital - 1.0) * 100.0
    return {"total_return": total_ret, "win_rate": win_rate, "trades": n_real}, pd.DataFrame(taken)

def optimize_thresholds_per_symbol_closed(
    trades_all: pd.DataFrame, *, step: float = 0.05, min_trades: int = 3
) -> tuple[dict, pd.DataFrame]:
    closed = trades_all[trades_all["exit_date"].notna()].copy()
    if closed.empty:
        sym_set = sorted(trades_all["symbol"].unique())
        return ({s: 0.00 for s in sym_set}, pd.DataFrame(columns=["symbol","threshold","trades","total_return_%","win_rate_%"]))
    start_ts = closed["entry_date"].min()
    end_ts   = closed["exit_date"].max()
    thr_values = np.round(np.arange(0.0, 1.0 + 1e-9, step), 2)
    rows, best_map = [], {}
    for sym, cand in closed.groupby("symbol"):
        best = None
        for thr in thr_values:
            m, _ = simulate_oaat(
                cand, start_ts, end_ts,
                starting_capital=10000.0, alloc_pct=100.0,
                threshold=thr, require_garch=True
            )
            tr = m["trades"]; score = m["total_return"]
            rows.append({"symbol": sym, "threshold": thr, "trades": tr,
                         "total_return_%": score,
                         "win_rate_%": (m["win_rate"]*100 if pd.notna(m["win_rate"]) else np.nan)})
            if tr < min_trades:
                continue
            candidate = (score, tr, -thr, thr)
            if (best is None) or (candidate > best):
                best = candidate
        best_map[sym] = (best[3] if best is not None else 0.00)
    return best_map, pd.DataFrame(rows)

# ==============================
# HTML rendering helpers
# ==============================
def render_table(df: pd.DataFrame, cols: list[str], headers: list[str]) -> str:
    if df.empty:
        return '<p class="muted">None today.</p>'
    thead = "<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>"
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            val = r.get(c)
            if c in ("entry", "exit_price", "stop_loss", "exit_or_last"):
                cells.append(fmt_money(val))
            elif c in ("pct_return", "unrealized_pct_return", "avg_return", "avg_win_return"):
                cls = "pos" if (pd.notna(val) and val > 0) else "neg" if (pd.notna(val) and val < 0) else "muted"
                cells.append(f'<span class="{cls}">{fmt_pct_signed(val)}</span>')
            elif c == "win_rate":
                cells.append("—" if pd.isna(val) else fmt_pct_plain(val*100, 0))
            elif c in ("entry_date", "exit_date"):
                cells.append(date_only(val))
            else:
                cells.append("—" if pd.isna(val) else str(val))
        body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return "<table>" + thead + "<tbody>" + "".join(body) + "</tbody></table>"

def open_trades_table_html(open_trades: pd.DataFrame) -> str:
    cols = ["symbol_display","sector","entry_date","entry","stop_loss",
            "avg_return","avg_win_return","win_rate","unrealized_pct_return"]
    headers = ["Symbol","Sector","Entry Date","Entry","Stop Loss",
               "Avg Return","Avg Win","Win Rate","Unrealized"]
    return render_table(open_trades[cols], cols, headers) if not open_trades.empty else '<p class="muted">No open trades.</p>'

def entries_today_table_html(entries_today: pd.DataFrame) -> str:
    cols = ["symbol_display","sector","entry_date","entry","stop_loss"]
    headers = ["Symbol","Sector","Entry Date","Entry","Stop Loss"]
    return render_table(entries_today[cols], cols, headers)

def exits_today_table_html(exits_today: pd.DataFrame) -> str:
    cols = ["symbol_display","sector","entry_date","exit_date","entry","exit_price","pct_return"]
    headers = ["Symbol","Sector","Entry Date","Exit Date","Entry","Exit","P&L %"]
    return render_table(exits_today[cols], cols, headers)

# ==============================
# Core: build snapshot then email
# ==============================
if __name__ == "__main__":
    # Load prices & caps
    df0 = pd.read_csv("stocks.csv", parse_dates=["date"])
    if "asset_type" not in df0.columns:
        df0["asset_type"] = "stock"
    if "sector" not in df0.columns:
        df0["sector"] = None
    df0["asset_type"] = df0["asset_type"].astype(str).str.lower()
    df0.loc[(df0["asset_type"]=="crypto") & (df0["sector"].isna()), "sector"] = "Crypto"

    caps = pd.read_csv("market_cap.csv")

    # Universe selection (same as app)
    stocks_only = df0[df0["asset_type"] == "stock"].copy()
    crypto_only = df0[df0["asset_type"] == "crypto"].copy()
    stocks_caps = caps[~caps["cap_score"].isin([3, 4])].sort_values("cap_score")
    top_stock_symbols = stocks_caps.head(100)["symbol"].unique().tolist()
    crypto_symbols = crypto_only["symbol"].unique().tolist()
    final_symbols = set(top_stock_symbols) | set(crypto_symbols)
    df = (df0[df0["symbol"].isin(final_symbols)]
             .copy()
             .sort_values(["symbol","date"]))

    cap_score_map = caps.set_index("symbol")["cap_score"]
    cap_emoji_map = caps.set_index("symbol")["cap_emoji"]

    # Strategy → Trades
    trades_raw = run_strategy(df)
    et = datetime.now(pytz.timezone("US/Eastern"))
    trading_day = et.date()

    if trades_raw.empty:
        subject = f"📊 Signals — {trading_day} • 0 entries • 0 exits • 0 open"
        body_text = f"No trades or signals detected on {trading_day}."
        body_html = PAGE_CSS + f'<div class="wrap"><h2>No trades or signals — {trading_day}</h2></div>'
        send_email(subject, body_text, body_html)
        raise SystemExit(0)

    # Attach GARCH & auto-threshold filter (same as app)
    trades_all = attach_garch_risk_index(df, trades_raw, base_vol_dec=0.20, lookback=252)
    AUTO_MIN_TRADES = 3
    AUTO_GRID_STEP  = 0.05
    thr_map, _ = optimize_thresholds_per_symbol_closed(
        trades_all, step=AUTO_GRID_STEP, min_trades=AUTO_MIN_TRADES
    )
    def _accept_row(r):
        thr = thr_map.get(r["symbol"], 0.00)
        return pd.notna(r["garch_risk_index"]) and (r["garch_risk_index"] >= thr)
    trades = trades_all[trades_all.apply(_accept_row, axis=1)].copy()

    # Enrich
    sector_map = df[["symbol","sector"]].drop_duplicates().set_index("symbol")["sector"]
    trades["sector"] = trades["symbol"].map(sector_map)
    trades["cap_score"] = trades["symbol"].map(cap_score_map)
    trades["cap_emoji"] = trades["symbol"].map(cap_emoji_map)
    trades["symbol_display"] = trades.apply(
        lambda r: f"{r['cap_emoji']} {r['symbol']}" if pd.notna(r["cap_emoji"]) else r["symbol"], axis=1
    )

    # Stop loss ref = yesterday's low at entry-date
    df["stop_loss"] = df.groupby("symbol")["low"].shift(1)
    entry_lows = df[["symbol","date","stop_loss"]].rename(columns={"date":"entry_date"})
    trades = trades.merge(entry_lows, on=["symbol","entry_date"], how="left")

    # Latest close & returns
    latest_prices = df.groupby("symbol", as_index=False).agg(latest_close=("close","last"))
    trades = trades.merge(latest_prices, on="symbol", how="left")
    trades["pct_return"] = (trades["exit_price"] / trades["entry"] - 1) * 100
    trades["unrealized_pct_return"] = (trades["latest_close"] / trades["entry"] - 1) * 100
    trades["final_pct"] = trades.apply(
        lambda r: r["pct_return"] if pd.notna(r["exit_price"]) else r["unrealized_pct_return"], axis=1
    )

    # Closed perf aggregates (avg_return / avg_win / win_rate)
    closed_hist = trades[trades["exit_date"].notna()].copy()
    if not closed_hist.empty:
        closed_hist["pct_return"] = (closed_hist["exit_price"] / closed_hist["entry"] - 1) * 100
        closed_hist["win"] = closed_hist["pct_return"] > 0
        avg_return = closed_hist.groupby("symbol")["pct_return"].mean().rename("avg_return")
        avg_win_return = (closed_hist[closed_hist["pct_return"] > 0]
                          .groupby("symbol")["pct_return"].mean()
                          .rename("avg_win_return"))
        win_rate = closed_hist.groupby("symbol")["win"].mean().rename("win_rate")
        trades = (trades
                  .merge(avg_return, on="symbol", how="left")
                  .merge(avg_win_return, on="symbol", how="left")
                  .merge(win_rate, on="symbol", how="left"))
    else:
        trades["avg_return"] = None
        trades["avg_win_return"] = None
        trades["win_rate"] = None

    # --------------------------
    # Split into sections (today)
    # --------------------------
    # Normalize dates to date() for comparison
    trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="coerce")
    trades["exit_date"]  = pd.to_datetime(trades["exit_date"], errors="coerce")

    entries_today = trades[trades["entry_date"].dt.date == trading_day].copy()
    exits_today   = trades[trades["exit_date"].notna() & (trades["exit_date"].dt.date == trading_day)].copy()

    # Days held for exits section
    if not exits_today.empty:
        exits_today["days_held"] = (exits_today["exit_date"] - exits_today["entry_date"]).dt.days

    # Open trades snapshot
    open_trades = trades[trades["exit_date"].isna()].copy()
    open_trades = open_trades.sort_values(["entry_date", "avg_return"], ascending=[False, False])

    # KPIs
    kpi_entries = len(entries_today)
    kpi_exits   = len(exits_today)
    kpi_open    = len(open_trades)
    unrealized_avg = open_trades["unrealized_pct_return"].mean() if kpi_open else np.nan
    best_txt = worst_txt = "—"
    if kpi_open:
        best_row = open_trades.loc[open_trades["unrealized_pct_return"].idxmax()]
        worst_row = open_trades.loc[open_trades["unrealized_pct_return"].idxmin()]
        best_txt = f"{best_row['symbol']} ({fmt_pct_signed(best_row['unrealized_pct_return'])}) ✅"
        worst_txt = f"{worst_row['symbol']} ({fmt_pct_signed(worst_row['unrealized_pct_return'])}) ❌"

    # ---------- TEXT BODY ----------
    lines = []
    lines.append(f"Signals — {trading_day} (US/Eastern)")
    lines.append(f"Entries today: {kpi_entries} | Exits today: {kpi_exits} | Open trades: {kpi_open}")
    lines.append("")

    # Exits today (ACTION)
    lines.append("== ACTION NEEDED: Close Today ==")
    if kpi_exits == 0:
        lines.append("• None today.")
    else:
        for _, r in exits_today.sort_values("exit_date").iterrows():
            lines.append(
                f"• {r['symbol']}  | Exit {date_only(r['exit_date'])} @ {fmt_money(r.get('exit_price'))}  "
                f"(Entry {date_only(r['entry_date'])} @ {fmt_money(r['entry'])}, P&L {fmt_pct_signed(r['pct_return'])})"
            )
    lines.append("")

    # Entries today
    lines.append("== New Entries Today ==")
    if kpi_entries == 0:
        lines.append("• None today.")
    else:
        for _, r in entries_today.sort_values("entry_date").iterrows():
            lines.append(
                f"• {r['symbol']}  | Enter {date_only(r['entry_date'])} @ {fmt_money(r['entry'])}  "
                f"(Stop {fmt_money(r.get('stop_loss'))}, WR hist {fmt_pct_plain(r.get('win_rate')*100,0) if pd.notna(r.get('win_rate')) else '—'})"
            )
    lines.append("")

    # Open snapshot lines
    lines.append("== Open Trades Snapshot ==")
    lines.append(f"Avg unrealized: {fmt_pct_signed(unrealized_avg) if pd.notna(unrealized_avg) else '—'} | Best: {best_txt} | Worst: {worst_txt}")
    if kpi_open:
        for _, r in open_trades.iterrows():
            lines.append(
                f"- {r['symbol']}: {r.get('sector','—')} | "
                f"Entry {date_only(r['entry_date'])} @ {fmt_money(r['entry'])} | "
                f"Unrl {fmt_pct_signed(r['unrealized_pct_return'])} | "
                f"WR {fmt_pct_plain(r.get('win_rate')*100,0) if pd.notna(r.get('win_rate')) else '—'} | "
                f"Avg {fmt_pct_signed(r.get('avg_return')) if pd.notna(r.get('avg_return')) else '—'}"
            )
    body_text = "\n".join(lines)

    # ---------- HTML BODY ----------
    html = []
    html.append(PAGE_CSS)
    html.append('<div class="wrap">')
    html.append(f'<h2>Signals — {trading_day} <span class="muted">(US/Eastern)</span></h2>')
    html.append('<div class="kpis">')
    html.append(f'<div class="chip"><strong>Entries today:</strong> {kpi_entries}</div>')
    html.append(f'<div class="chip"><strong>Exits today:</strong> {kpi_exits}</div>')
    html.append(f'<div class="chip"><strong>Open trades:</strong> {kpi_open}</div>')
    html.append(f'<div class="chip"><strong>Avg unrealized:</strong> {fmt_pct_signed(unrealized_avg) if pd.notna(unrealized_avg) else "—"}</div>')
    html.append('</div>')

    # Exits today
    html.append('<h3>🚨 Action Needed: Close Today</h3>')
    html.append(exits_today_table_html(exits_today))

    # Entries today
    html.append('<h3>🟢 New Entries Today</h3>')
    html.append(entries_today_table_html(entries_today))

    # Open trades snapshot
    html.append('<h3>📦 Open Trades Snapshot</h3>')
    html.append(open_trades_table_html(open_trades))

    html.append(f'<div class="foot">Generated at {et.strftime("%Y-%m-%d %H:%M %Z")} • '
                'GARCH-filtered & auto-thresholded (same logic as the app).</div>')
    html.append('</div>')
    body_html = "".join(html)

    # Subject & send
    subject = (f"📨 Signals — {trading_day} • "
               f"{kpi_entries} entries • {kpi_exits} exits • {kpi_open} open")
    send_email(subject, body_text, body_html)
