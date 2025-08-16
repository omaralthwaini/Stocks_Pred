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

        # Show all trades (latest first)
        st.subheader("📋 All Detected Trades")
        st.dataframe(trades.sort_values("entry_date", ascending=False), use_container_width=True)

        # Download all trades
        csv_all = trades.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download All Trades", csv_all, "all_trades.csv", "text/csv")

        # Build ML dataset
        ml_df = build_ml_dataset(df, trades)

        # --- Filter open trades only ---
# Step 1: Get list of open trade IDs from trades
        open_ids = trades[trades["outcome"] == 0].apply(
        lambda row: f"{row['symbol']}_{row['entry_date'].strftime('%Y%m%d')}", axis=1
)

# Step 2: Filter ML dataset only for the entry row of these trades
        ml_open = ml_df[
        (ml_df["days_before_entry"] == 0) &
        (ml_df["trade_id"].isin(open_ids))
        ].copy()

        if ml_open.empty:
            st.info("✅ No open trades. All have been exited.")
        else:
            model = load_model()
            ml_pred_df = load_model_and_predict(ml_open, model)

            st.subheader("🤖 ML Predictions for Open Trades")

            # Calculate predicted % return
            ml_pred_df["predicted_pct_return"] = 100 * (ml_pred_df["predicted_exit"] / ml_pred_df["entry"] - 1)

            # Sort and display
            st.dataframe(
                ml_pred_df[[
                    "symbol", "sector", "date", "entry",
                    "predicted_exit", "predicted_pct_return"
                ]].sort_values("predicted_pct_return", ascending=False),
                use_container_width=True
            )

            # Download predictions
            csv_pred = ml_pred_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download ML Predictions", csv_pred, "ml_predictions_open_trades.csv", "text/csv")
