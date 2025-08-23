import os
import smtplib
from email.message import EmailMessage
from datetime import datetime
import pytz
import pandas as pd

from strategy import run_strategy

# ---------- Helpers ----------
def fmt_money(x):
    return f"${x:,.2f}" if pd.notna(x) else "—"

def fmt_pct_signed(x, digits=2):
    return f"{x:+.{digits}f}%" if pd.notna(x) else "—"

def fmt_pct_plain(x, digits=0):
    return f"{x:.{digits}f}%" if pd.notna(x) else "—"

def df_to_html_table(df, columns, headers=None):
    headers = headers or columns
    # Minimal, inbox-friendly styling
    styles = """
      <style>
        body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
        .wrap { max-width: 820px; margin: 0 auto; }
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
    thead = "<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>"
    body_rows = []
    for _, r in df.iterrows():
        tds = []
        for c in columns:
            val = r[c]
            # Money columns
            if c in {"entry", "exit_price", "target_price"}:
                cell = fmt_money(val)
            # Return columns: realized exit return (ret_pct) OR historical avg_return
            elif c in {"ret_pct", "avg_return"}:
                if pd.isna(val):
                    cell = '<span class="muted">—</span>'
                else:
                    cls = "pos" if val > 0 else "neg" if val < 0 else "muted"
                    cell = f'<span class="{cls}">{fmt_pct_signed(val)}</span>'
            # Win rate (0..1 → %)
            elif c == "win_rate":
                cell = fmt_pct_plain(val * 100 if pd.notna(val) else val, 0)
            # Dates and other text
            else:
                cell = str(val) if pd.notna(val) else "—"
            tds.append(f"<td>{cell}</td>")
        body_rows.append("<tr>" + "".join(tds) + "</tr>")
    tbody = "<tbody>" + "".join(body_rows) + "</tbody>"
    return styles + f"<table>{thead}{tbody}</table>"

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

# ---------- Load & compute ----------
df = pd.read_csv("stocks.csv", parse_dates=["date"])
caps = pd.read_csv("market_cap.csv")

# Use US/Eastern trading day for “today”
et = datetime.now(pytz.timezone("US/Eastern"))
trading_day = et.date()

# Strategy
trades = run_strategy(df, caps)

# Normalize to date-only comparisons
trades["entry_date"] = pd.to_datetime(trades["entry_date"]).dt.date
trades["exit_date"]  = pd.to_datetime(trades["exit_date"]).dt.date

# Enrich with sector / cap info
sector_map   = df[["symbol","sector"]].drop_duplicates().set_index("symbol")["sector"]
cap_score    = caps.set_index("symbol")["cap_score"]
cap_emoji    = caps.set_index("symbol")["cap_emoji"]

trades["sector"]      = trades["symbol"].map(sector_map)
trades["cap_score"]   = trades["symbol"].map(cap_score)
trades["cap_emoji"]   = trades["symbol"].map(cap_emoji)
trades["symbol_disp"] = trades.apply(
    lambda r: f"{r['cap_emoji']} {r['symbol']}" if pd.notna(r.get("cap_emoji")) else r["symbol"], axis=1
)

# Latest close snapshot (not used for exit P/L)
latest = (df.sort_values("date")
            .groupby("symbol")
            .agg(latest_close=("close","last")))
trades = trades.merge(latest, on="symbol", how="left")

# Compute pct_return for ALL closed trades (needed for per-ticker history)
trades["pct_return"] = (trades["exit_price"] / trades["entry"] - 1) * 100

# Per-ticker performance from CLOSED trades
closed = trades[trades["exit_date"].notna()].copy()
if not closed.empty:
    perf = (closed.assign(win=lambda d: d["pct_return"] > 0)
                  .groupby("symbol")
                  .agg(win_rate=("win","mean"),           # 0..1
                       avg_return=("pct_return","mean"))   # %
                  .reset_index())
else:
    perf = pd.DataFrame(columns=["symbol","win_rate","avg_return"])

win_rate_map   = perf.set_index("symbol")["win_rate"].to_dict()
avg_return_map = perf.set_index("symbol")["avg_return"].to_dict()

# Today slices
today_entries = trades.loc[trades["entry_date"] == trading_day].copy()
today_exits   = trades.loc[trades["exit_date"]  == trading_day].copy()

# Compute exit P/L for today exits only (already have pct_return overall, but keep ret_pct name for table)
if not today_exits.empty:
    today_exits["ret_pct"] = ((today_exits["exit_price"] / today_exits["entry"]) - 1) * 100

# ---------- Build TEXT ----------
lines = []
lines.append(f"Daily Trade Summary — {trading_day} (US/Eastern)\n")

# KPI line
kpi_entries = len(today_entries)
kpi_exits   = len(today_exits)
win_rate_day = f"{(today_exits['ret_pct'] > 0).mean():.0%}" if kpi_exits else "—"
avg_ret_day  = f"{today_exits['ret_pct'].mean():+.2f}%" if kpi_exits else "—"
lines.append(f"New entries: {kpi_entries} | Exits: {kpi_exits} | Win rate: {win_rate_day} | Avg exit return: {avg_ret_day}\n")

# New entries (now with historical WR/Avg per ticker)
if kpi_entries:
    lines.append("NEW ENTRIES:")
    for _, r in today_entries.sort_values(["cap_score","symbol"]).iterrows():
        target = r["entry"] * 1.05 if pd.notna(r["entry"]) else None
        wr  = win_rate_map.get(r["symbol"])
        ar  = avg_return_map.get(r["symbol"])
        wrt = fmt_pct_plain(wr*100, 0) if wr is not None else "—"
        art = fmt_pct_signed(ar, 2)    if ar is not None else "—"
        lines.append(
            f"- {r['symbol']} ({r.get('sector','—')}) "
            f"Entry {r['entry_date']} @ {fmt_money(r['entry'])}"
            + (f" | 5% target: {fmt_money(target)}" if target else "")
            + f" | Hist: WR {wrt}, Avg {art}"
        )
else:
    lines.append("No new entries today.")

# Exits
lines.append("")
if kpi_exits:
    lines.append("EXITS:")
    for _, r in today_exits.sort_values("ret_pct", ascending=False).iterrows():
        reason = {"sma_below_2": "SMA Cross"}.get(str(r.get("exit_reason")), str(r.get("exit_reason","—")).title())
        lines.append(
            f"- {r['symbol']} ({r.get('sector','—')}) "
            f"Entry {r['entry_date']} @ {fmt_money(r['entry'])} → "
            f"Exit {r['exit_date']} @ {fmt_money(r['exit_price'])} "
            f"({reason}) | Return {fmt_pct_signed(r['ret_pct'])}"
        )
else:
    lines.append("No exits today.")

body_text = "\n".join(lines)

# ---------- Build HTML ----------
header = f"""
<div class="wrap">
  <h2>Daily Trade Summary — {trading_day} <span class="muted">(US/Eastern)</span></h2>
  <div class="kpis">
    <div class="chip"><strong>New entries:</strong> {kpi_entries}</div>
    <div class="chip"><strong>Exits:</strong> {kpi_exits}</div>
    <div class="chip"><strong>Win rate:</strong> {win_rate_day}</div>
    <div class="chip"><strong>Avg exit return:</strong> {avg_ret_day}</div>
  </div>
"""

entries_html = ""
if kpi_entries:
    df_entries = today_entries.sort_values(["cap_score","symbol"]).copy()
    df_entries["target_price"] = (df_entries["entry"] * 1.05).round(2)
    df_entries["cap"] = df_entries.apply(
        lambda r: f"{r['cap_emoji']} {int(r['cap_score'])}" if pd.notna(r.get("cap_score")) else "—", axis=1
    )
    # attach per-ticker history
    df_entries["win_rate"]   = df_entries["symbol"].map(win_rate_map)      # 0..1
    df_entries["avg_return"] = df_entries["symbol"].map(avg_return_map)    # %
    cols = ["symbol_disp","sector","cap","entry","target_price","win_rate","avg_return"]
    headers = ["Symbol","Sector","Cap","Entry","Target (+5%)","Win rate","Avg return"]
    entries_html = f"<h3>New Entries ({kpi_entries})</h3>" + df_to_html_table(df_entries, cols, headers)

exits_html = ""
if kpi_exits:
    df_exits = today_exits.sort_values("ret_pct", ascending=False).copy()
    df_exits["reason_print"] = df_exits["exit_reason"].map({"sma_below_2":"SMA Cross"}).fillna(
        df_exits["exit_reason"].astype(str).str.title()
    )
    cols = ["symbol_disp","sector","entry_date","exit_date","entry","exit_price","ret_pct","reason_print"]
    headers = ["Symbol","Sector","Entry Date","Exit Date","Entry","Exit","Return","Reason"]
    exits_html = f"<h3>Exits ({kpi_exits})</h3>" + df_to_html_table(df_exits, cols, headers)

foot = f"""
  <div class="foot">Generated at {et.strftime('%Y-%m-%d %H:%M %Z')} • This email includes only trades opened/closed today.</div>
</div>
"""

body_html = (entries_html or exits_html) and (entries_html + exits_html) or "<p>No activity today.</p>"
body_html = header + body_html + foot

# ---------- Subject & Send ----------
subject = f"📊 Trades — {trading_day} | {kpi_entries} new • {kpi_exits} exits"
if __name__ == "__main__":
    send_email(subject, body_text, body_html)
