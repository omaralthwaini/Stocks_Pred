import os
import smtplib
from email.message import EmailMessage
from datetime import datetime

import pandas as pd
from strategy import run_strategy

# --- Load stock data ---
df = pd.read_csv("stocks.csv", parse_dates=["date"])
today = pd.Timestamp.now().normalize()

# --- Run strategy and filter today's trades ---
trades = run_strategy(df)
trades["entry_date"] = pd.to_datetime(trades["entry_date"]).dt.normalize()
today_trades = trades[trades["entry_date"] == today]

# --- Compose message ---
if not today_trades.empty:
    lines = [f"✅ {len(today_trades)} new trades detected today:\n"]
    for _, row in today_trades.iterrows():
        lines.append(f"- {row['symbol']} @ ${row['entry']:.2f}")
    body = "\n".join(lines)
    subject = f"📈 {len(today_trades)} New Trades Detected — {today.date()}"
else:
    body = "No new trades were detected today."
    subject = f"📉 No New Trades — {today.date()}"

# --- Send Email ---
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

if __name__ == "__main__":
    send_trade_summary(subject, body)
