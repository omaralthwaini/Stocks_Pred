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
# HTML rendering (Open Trades table)
# ==============================
def open_trades_table_html(open_trades: pd.DataFrame) -> str:
    styles = """
      <style>
        body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
        .wrap { max-width: 980px; margin: 0 auto; }
        .kpis { display:flex; gap:12px; flex-wrap:wrap; margin: 8px 0 14px; }
        .chip { border:1px solid #eee; border-radius:8px; padding:8px 10px; background:#fafafa; }
        table { border-collapse: collapse; width: 100%; }
        th, td { text-align: left; padding: 8px; font-size: 13px; }
        thead th { border-bottom: 2px solid #ddd; background:#f6f8fa; }
        tbody tr { border-bottom: 1px solid #eee; }
        .pos { color: #0a7a0a; font-weight: 600; }
        .neg { color: #c23232; font-weight: 600; }
        .muted { color:#666; }
        h2 { margin: 18px 0 8px; font-size: 18px; }
        h3 { margin: 12px 0 6px; font-size: 16px; }
        .foot { color:#888; font-size:12px; margin-top:14px; }
      </style>
    """
    cols = ["symbol_display","sector","entry_date","entry","stop_loss",
            "avg_return","avg_win_return","win_rate","unrealized_pct_return"]
    headers = ["Symbol","Sector","Entry Date","Entry","Stop Loss",
               "Avg Return","Avg Win","Win Rate","Unrealized"]

    thead = "<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>"
    body_rows = []
    for _, r in open_trades.iterrows():
        symbol = r.get("symbol_display", r.get("symbol","—"))
        sector = r.get("sector", "—")
        edate  = date_only(r.get("entry_date"))
        entry  = fmt_money(r.get("entry"))
        sl     = fmt_money(r.get("stop_loss"))
        a_ret  = r.get("avg_return")
        a_win  = r.get("avg_win_return")
        wr     = r.get("win_rate")  # 0..1
        unrl   = r.get("unrealized_pct_return")

        def pct_cell(val, signed=True):
            if pd.isna(val): return '<span class="muted">—</span>'
            cls = "pos" if val > 0 else "neg" if val < 0 else "muted"
            txt = fmt_pct_signed(val) if signed else fmt_pct_plain(val*100, 0)
            return f'<span class="{cls}">{txt}</span>'

        cells = [
            symbol,
            sector,
            edate,
            entry,
            sl,
            pct_cell(a_ret),
            pct_cell(a_win),
            "—" if pd.isna(wr) else fmt_pct_plain(wr*100, 0),
            pct_cell(unrl)
        ]
        body_rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    tbody = "<tbody>" + "".join(body_rows) + "</tbody>"
    return styles + f"<table>{thead}{tbody}</table>"

# ==============================
# Core: build snapshot then email
# ==============================
if __name__ == "__main__":
    # Load prices (stocks + crypto) and caps
    df0 = pd.read_csv("stocks.csv", parse_dates=["date"])
    if "asset_type" not in df0.columns:
        df0["asset_type"] = "stock"
    if "sector" not in df0.columns:
        df0["sector"] = None
    df0["asset_type"] = df0["asset_type"].astype(str).str.lower()
    # Ensure crypto rows have sector
    df0.loc[(df0["asset_type"]=="crypto") & (df0["sector"].isna()), "sector"] = "Crypto"

    caps = pd.read_csv("market_cap.csv")

    # Universe selection (same as app):
    #  - Stocks: exclude cap_score 3 & 4, then top 100 by cap_score
    #  - Crypto: keep ALL crypto symbols
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
    if trades_raw.empty:
        # still send a "no trades" email
        et = datetime.now(pytz.timezone("US/Eastern"))
        subject = f"📊 Trades — {et.date()} | No trades detected"
        send_email(subject, "No trades detected.", "<p>No trades detected.</p>")
        raise SystemExit(0)

    # Attach GARCH & learn per-symbol thresholds from CLOSED trades; keep >= thr
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
    if trades.empty:
        et = datetime.now(pytz.timezone("US/Eastern"))
        subject = f"📊 Trades — {et.date()} | All filtered by thresholds"
        send_email(subject, "All trades were filtered out by thresholds.", "<p>All trades were filtered out by thresholds.</p>")
        raise SystemExit(0)

    # Enrich (same as app)
    sector_map = df[["symbol","sector"]].drop_duplicates().set_index("symbol")["sector"]
    trades["sector"] = trades["symbol"].map(sector_map)
    trades["cap_score"] = trades["symbol"].map(cap_score_map)
    trades["cap_emoji"] = trades["symbol"].map(cap_emoji_map)
    trades["symbol_display"] = trades.apply(
        lambda r: f"{r['cap_emoji']} {r['symbol']}" if pd.notna(r["cap_emoji"]) else r["symbol"], axis=1
    )

    # Stop loss = yesterday's low at entry-date reference
    df["stop_loss"] = df.groupby("symbol")["low"].shift(1)
    entry_lows = df[["symbol","date","stop_loss"]].rename(columns={"date":"entry_date"})
    trades = trades.merge(entry_lows, on=["symbol","entry_date"], how="left")

    # Latest close, returns
    latest_prices = df.groupby("symbol", as_index=False).agg(latest_close=("close","last"))
    trades = trades.merge(latest_prices, on="symbol", how="left")
    trades["pct_return"] = (trades["exit_price"] / trades["entry"] - 1) * 100
    trades["unrealized_pct_return"] = (trades["latest_close"] / trades["entry"] - 1) * 100
    trades["final_pct"] = trades.apply(
        lambda r: r["pct_return"] if pd.notna(r["exit_price"]) else r["unrealized_pct_return"], axis=1
    )

    # Closed perf aggregates (avg_return / avg_win / win_rate)
    closed = trades[trades["exit_date"].notna()].copy()
    if not closed.empty:
        closed["pct_return"] = (closed["exit_price"] / closed["entry"] - 1) * 100
        closed["win"] = closed["pct_return"] > 0
        avg_return = closed.groupby("symbol")["pct_return"].mean().rename("avg_return")
        avg_win_return = (closed[closed["pct_return"] > 0]
                          .groupby("symbol")["pct_return"].mean()
                          .rename("avg_win_return"))
        win_rate = closed.groupby("symbol")["win"].mean().rename("win_rate")
        trades = (trades
                  .merge(avg_return, on="symbol", how="left")
                  .merge(avg_win_return, on="symbol", how="left")
                  .merge(win_rate, on="symbol", how="left"))
    else:
        trades["avg_return"] = None
        trades["avg_win_return"] = None
        trades["win_rate"] = None

    # Open trades snapshot (like the Home page)
    open_trades = trades[trades["exit_date"].isna()].copy()
    open_trades = open_trades.sort_values(["entry_date", "avg_return"], ascending=[False, False])

    # KPIs (match the spirit of the app)
    et = datetime.now(pytz.timezone("US/Eastern"))
    trading_day = et.date()
    kpi_open = len(open_trades)
    unrealized_avg = open_trades["unrealized_pct_return"].mean() if kpi_open else np.nan
    best_txt = worst_txt = "—"
    if kpi_open:
        best_row = open_trades.loc[open_trades["unrealized_pct_return"].idxmax()]
        worst_row = open_trades.loc[open_trades["unrealized_pct_return"].idxmin()]
        best_txt = f"{best_row['symbol']} ({fmt_pct_signed(best_row['unrealized_pct_return'])}) ✅"
        worst_txt = f"{worst_row['symbol']} ({fmt_pct_signed(worst_row['unrealized_pct_return'])}) ❌"

    # ---------- TEXT BODY ----------
    lines = []
    lines.append(f"Open Trades Snapshot — {trading_day} (US/Eastern)\n")
    lines.append(f"Open trades: {kpi_open} | Avg unrealized: {fmt_pct_signed(unrealized_avg) if pd.notna(unrealized_avg) else '—'}")
    lines.append(f"Best: {best_txt} | Worst: {worst_txt}\n")
    if kpi_open:
        lines.append("Symbols:")
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
    header = f"""
    <div class="wrap">
      <h2>Open Trades Snapshot — {trading_day} <span class="muted">(US/Eastern)</span></h2>
      <div class="kpis">
        <div class="chip"><strong>Open trades:</strong> {kpi_open}</div>
        <div class="chip"><strong>Avg unrealized:</strong> {fmt_pct_signed(unrealized_avg) if pd.notna(unrealized_avg) else '—'}</div>
        <div class="chip"><strong>Best:</strong> {best_txt}</div>
        <div class="chip"><strong>Worst:</strong> {worst_txt}</div>
      </div>
    """

    table_html = open_trades_table_html(open_trades) if kpi_open else "<p>No open trades.</p>"
    foot = f'<div class="foot">Generated at {et.strftime("%Y-%m-%d %H:%M %Z")} • Uses GARCH-filtered, auto-thresholded strategy and the same logic as the app Home page.</div></div>'
    body_html = header + table_html + foot

    # Subject & send
    subject = f"📈 Open Trades — {trading_day} • {kpi_open} open • Avg {fmt_pct_signed(unrealized_avg) if pd.notna(unrealized_avg) else '—'}"
    send_email(subject, body_text, body_html)
