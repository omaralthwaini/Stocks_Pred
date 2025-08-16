import streamlit as st
import pandas as pd
import joblib
from strategy import run_strategy
from ml_features import build_ml_dataset
from model_loader import load_model_and_predict

# --- Title ---
st.title("📈 Smart Backtester + ML Exit Predictor")

# --- Load cached data ---
@st.cache_data
def load_data():
    df = pd.read_csv("stocks.csv", parse_dates=["date"])
    return df.sort_values(["symbol", "date"])

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

# --- Begin execution ---
with st.spinner("⏳ Running strategy and predicting exit prices..."):
    df = load_data()
    trades = run_strategy(df)

    if trades.empty:
        st.warning("⚠️ No trades found.")
    else:
        st.success(f"✅ {len(trades)} trades detected")

        # Show all trades
        st.subheader("📋 All Detected Trades")
        st.dataframe(trades.sort_values("entry_date", ascending=False), use_container_width=True)

        # Download trades
        csv_all = trades.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download All Trades", csv_all, "all_trades.csv", "text/csv")

        # Build ML dataset
        ml_df = build_ml_dataset(df, trades)

        # Filter only open trades (outcome = 0, entry row)
        open_trades = trades[trades["outcome"] == 0].copy()
        ml_open = ml_df[
            (ml_df["days_before_entry"] == 0) &
            (ml_df["symbol"].isin(open_trades["symbol"])) &
            (ml_df["date"].isin(open_trades["entry_date"]))
        ].copy()

        if ml_open.empty:
            st.info("✅ No open trades. All have been exited.")
        else:
            model = load_model()
            ml_pred_df = load_model_and_predict(ml_open, model)

            # Compute predicted % return
            ml_pred_df["predicted_pct_return"] = 100 * (ml_pred_df["predicted_exit"] / ml_pred_df["entry"] - 1)

            # ----------------------------
            # FILTERS
            # ----------------------------
            st.subheader("🔍 Filter Predictions")

            # Available filter options
            symbols = sorted(ml_pred_df["symbol"].unique())
            sectors = sorted(ml_pred_df["sector"].dropna().unique())
            dates = pd.to_datetime(ml_pred_df["date"])
            min_date, max_date = dates.min(), dates.max()

            # Filters
            selected_symbols = st.multiselect("Filter by Symbol", options=symbols, default=symbols)
            selected_sectors = st.multiselect("Filter by Sector", options=sectors, default=sectors)
            selected_date_range = st.slider("Filter by Entry Date", min_value=min_date, max_value=max_date, value=(min_date, max_date))

            # Apply filters
            filtered_df = ml_pred_df[
                (ml_pred_df["symbol"].isin(selected_symbols)) &
                (ml_pred_df["sector"].isin(selected_sectors)) &
                (ml_pred_df["date"] >= selected_date_range[0]) &
                (ml_pred_df["date"] <= selected_date_range[1])
            ].copy()

            filtered_df = filtered_df.sort_values("predicted_pct_return", ascending=False)

            # ----------------------------
            # DISPLAY
            # ----------------------------
            st.subheader(f"🤖 ML Predictions for Open Trades ({len(filtered_df)} shown)")
            st.dataframe(
                filtered_df[[
                    "symbol", "sector", "date", "entry",
                    "predicted_exit", "predicted_pct_return"
                ]],
                use_container_width=True
            )

            # Download filtered
            csv_filtered = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download ML Predictions", csv_filtered, "ml_predictions_filtered.csv", "text/csv")
