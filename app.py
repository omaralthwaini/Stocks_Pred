import streamlit as st
import pandas as pd
import joblib  # optional if you re-add ML later
from strategy import run_strategy

# --- Title ---
st.title("📈 Smart Backtester — Trade Signal Viewer")

# --- Load cached stock data ---
@st.cache_data
def load_data():
    df = pd.read_csv("stocks.csv", parse_dates=["date"])
    return df.sort_values(["symbol", "date"])

# --- Load data + run strategy ---
with st.spinner("⏳ Running strategy to detect trades..."):
    df = load_data()
    trades = run_strategy(df)

if trades.empty:
    st.warning("⚠️ No trades were detected.")
    st.stop()
else:
    st.success(f"✅ {len(trades)} trades detected.")

    # --- Filter Options ---
    st.subheader("🔍 Filter Trades")

    symbols = sorted(trades["symbol"].unique())
    sectors = sorted(df["sector"].dropna().unique()) if "sector" in df.columns else []

    selected_symbols = st.multiselect("Filter by Symbol", options=symbols, default=symbols)
    selected_sectors = st.multiselect("Filter by Sector", options=sectors, default=sectors) if sectors else []

    # Apply filters
    filtered = trades.copy()
    if selected_symbols:
        filtered = filtered[filtered["symbol"].isin(selected_symbols)]
    if sectors and selected_sectors:
        filtered = filtered[filtered["symbol"].isin(
            df[df["sector"].isin(selected_sectors)]["symbol"].unique()
        )]

    # --- Show table ---
    st.subheader(f"📋 Filtered Trades ({len(filtered)} shown)")
    st.dataframe(filtered.sort_values("entry_date", ascending=False), use_container_width=True)

    # --- Download ---
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Filtered Trades", csv, "filtered_trades.csv", "text/csv")
