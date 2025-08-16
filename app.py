import streamlit as st
import pandas as pd
from strategy import run_strategy

# --- Title ---
st.title("📈 Smart Backtester — SMA Strategy")

# --- Load and sort stock data ---
@st.cache_data
def load_data():
    df = pd.read_csv("stocks.csv", parse_dates=["date"])
    return df.sort_values(["symbol", "date"])

# --- Load and run strategy ---
@st.cache_data
def run_all_trades(df):
    return run_strategy(df)

# --- Main execution ---
with st.spinner("⏳ Running strategy..."):
    df = load_data()
    trades = run_all_trades(df)

    if trades.empty:
        st.warning("⚠️ No trades were detected.")
    else:
        st.success(f"✅ {len(trades)} trades detected")

        # --- Filter Controls ---
        st.subheader("🔍 Filter Trades")

        # Options
        symbols = sorted(trades["symbol"].unique())
        sectors = sorted(df["sector"].dropna().unique())
        trades["entry_date"] = pd.to_datetime(trades["entry_date"])
        min_date = trades["entry_date"].min()
        max_date = trades["entry_date"].max()

        selected_symbols = st.multiselect("Filter by Symbol", symbols, default=symbols)
        selected_sectors = st.multiselect("Filter by Sector", sectors, default=sectors)
        # Ensure dates are proper Python date objects, not pandas Timestamps
        min_date = pd.to_datetime(min_date).date()
        max_date = pd.to_datetime(max_date).date()

        selected_date_range = st.slider(
            "Filter by Entry Date",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date)
        )

        # Apply filters
        filtered = trades[
            (trades["symbol"].isin(selected_symbols)) &
            (trades["entry_date"] >= selected_date_range[0]) &
            (trades["entry_date"] <= selected_date_range[1])
        ]

        # Join sector (optional)
        sector_map = df[["symbol", "sector"]].drop_duplicates().set_index("symbol")["sector"]
        filtered["sector"] = filtered["symbol"].map(sector_map)

        filtered = filtered[filtered["sector"].isin(selected_sectors)]

        # --- Display ---
        st.subheader(f"📋 Filtered Trades ({len(filtered)} shown)")
        st.dataframe(filtered.sort_values("entry_date", ascending=False), use_container_width=True)

        # --- Download ---
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Filtered Trades", csv, "filtered_trades.csv", "text/csv")
