import streamlit as st
import pandas as pd
import joblib
from strategy import run_strategy
from ml_features import build_ml_dataset
from model_loader import load_model_and_predict

# --- Title ---
st.title("📈 Smart Backtester + ML Exit Predictor")

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv("stocks.csv", parse_dates=["date"])
    return df.sort_values(["symbol", "date"])

# --- Load pre-trained model ---
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

# --- Load strategy trades ---
@st.cache_data
def get_trades(df):
    return run_strategy(df)

# --- Begin app execution ---
with st.spinner("⏳ Running strategy and predicting exit prices..."):
    df = load_data()
    trades = get_trades(df)

    if trades.empty:
        st.warning("⚠️ No trades found.")
    else:
        st.success(f"✅ {len(trades)} trades detected")

        # Show detected trades
        st.subheader("📋 All Detected Trades")
        st.dataframe(trades.sort_values("entry_date", ascending=False), use_container_width=True)

        # Download detected trades
        csv_all = trades.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download All Trades", csv_all, "all_trades.csv", "text/csv")

        # --- Build ML dataset ---
        ml_df = build_ml_dataset(df, trades)

        # --- Select only open trades (outcome == 0) ---
        open_trades = trades[trades["outcome"] == 0].copy()

        # Select entry rows only
        entry_rows = ml_df[ml_df["days_before_entry"] == 0].copy()

        # Join on symbol + entry_date to get open ML rows
        ml_open = entry_rows.merge(
            open_trades[["symbol", "entry_date"]],
            left_on=["symbol", "date"],
            right_on=["symbol", "entry_date"],
            how="inner"
        ).copy()

        if ml_open.empty:
            st.info("✅ No open trades found (all have exited).")
        else:
            # Predict
            model = load_model()
            ml_pred_df = load_model_and_predict(ml_open, model)

            # Predicted return
            ml_pred_df["predicted_pct_return"] = 100 * (
                ml_pred_df["predicted_exit"] / ml_pred_df["entry"] - 1
            )

            # --- Filters ---
            st.subheader("🔍 Filter Predictions")

            available_symbols = sorted(ml_pred_df["symbol"].unique())
            available_sectors = sorted(ml_pred_df["sector"].dropna().unique())
            min_date = ml_pred_df["date"].min()
            max_date = ml_pred_df["date"].max()

            selected_symbols = st.multiselect(
                "Filter by Symbol", available_symbols, default=available_symbols
            )
            selected_sectors = st.multiselect(
                "Filter by Sector", available_sectors, default=available_sectors
            )
            selected_date_range = st.slider(
                "Filter by Entry Date",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date)
            )

            # Apply filters
            filtered_df = ml_pred_df[
                (ml_pred_df["symbol"].isin(selected_symbols)) &
                (ml_pred_df["sector"].isin(selected_sectors)) &
                (ml_pred_df["date"] >= selected_date_range[0]) &
                (ml_pred_df["date"] <= selected_date_range[1])
            ].sort_values("predicted_pct_return", ascending=False).copy()

            # --- Display results ---
            st.subheader(f"🤖 ML Predictions for Open Trades ({len(filtered_df)} shown)")
            st.dataframe(
                filtered_df[[
                    "symbol", "sector", "date", "entry",
                    "predicted_exit", "predicted_pct_return"
                ]],
                use_container_width=True
            )

            # --- Download predictions ---
            csv_pred = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download ML Predictions", csv_pred, "ml_predictions_filtered.csv", "text/csv")
