import os
import smtplib
from email.message import EmailMessage
from datetime import datetime
import pandas as pd
from strategy import run_strategy

# --- Load data ---
df = pd.read_csv("stocks.csv", parse_dates=["date"])
caps = pd.read_csv("market_cap.csv")

# --- Normalize today's date ---
today = pd.Timestamp.now().normalize()

# --- Run strategy and normalize dates ---
trades = run_strategy(df, caps)
trades["entry_date"] = pd.to_datetime(trades["entry_date"]).dt.normalize()
trades["exit_date"] = pd.to_datetime(trades["exit_date"]).dt.normalize()

# --- Today's new entries ---
today_entries = trades[trades["entry_date"] == today]

# --- Today's exits ---
today_exits = trades[trades["exit_date"] == today]

# --- Build message ---
lines = []

# --- Section: New Entries ---
if not today_entries.empty:
    lines.append(f"📈 {len(today_entries)} new trade(s) opened today:\n")
    for _, row in today_entries.iterrows():
        lines.append(f"- {row['symbol']} | Entry: {row['entry_date'].date()} @ ${row['entry']:.2f}")
else:
    lines.append("📭 No new trades opened today.\n")

# --- Section: Closed Trades ---
if not today_exits.empty:
    lines.append(f"\n📤 {len(today_exits)} trade(s) exited today:\n")
    for _, row in today_exits.iterrows():
        entry_price = row["entry"]
        exit_price = row["exit_price"]
        pct = ((exit_price / entry_price) - 1) * 100 if entry_price else 0
        pct_str = f"{pct:+.2f}%"

        # Outcome icon
        if pct > 0:
            outcome = "✅ Profit"
        elif pct < 0:
            outcome = "❌ Loss"
        else:
            outcome = "⚪ Break-even"

        reason = "SMA Cross" if row["exit_reason"] == "sma_below_2" else "Stop Loss"
        lines.append(f"- {row['symbol']} | Exit: {row['exit_date'].date()} @ ${exit_price:.2f} ({reason}) — {outcome} ({pct_str})")
else:
    lines.append("\n📭 No trades exited today.")

# --- Final email ---
body = "\n".join(lines)
subject = f"📊 Trade Summary — {today.date()}"

# --- Send email ---
def send_trade_summary(subject, body):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ["EMAIL_USER"]
    msg["To"] = os.environ["EMAIL_TO"]
    msg.set_content(body)

    with smtplib.SMTP(os.environ["EMAIL_SMTP_HOST"], int(os.environ["EMAIL_SMTP_PORT"])) as server:
        server.starttls()
        server.login(os.environ["EMAIL_USER"], os.environ["EMAIL_PASS"])
        server.send_message(msg)
        print("✅ Email sent successfully.")

# --- Trigger if main ---
if __name__ == "__main__":
    send_trade_summary(subject, body)
