import streamlit as st
import pandas as pd
import joblib
import numpy as np
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

# --- Load model ---
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

# --- Begin Execution ---
with st.spinner("⏳ Running strategy and predicting exit prices..."):
    df = load_data()
    trades = run_strategy(df)

    if trades.empty:
        st.warning("⚠️ No valid trades were detected.")
    else:
        st.success(f"✅ {len(trades)} trades detected")

        # Build full ML dataset (historical + open)
        ml_df = build_ml_dataset(df, trades)

        # ✅ Filter: Only open trades (no exit recorded)
        # ✅ Filter: Only open trades (outcome = 0 means not exited yet)
        ml_open = ml_df[
        (ml_df["days_before_entry"] == 0) & 
        (ml_df["outcome"] == 0)
        ].copy()


        if ml_open.empty:
            st.info("✅ No open trades found. All trades have already exited.")
        else:
            model = load_model()  # ✅ Use cached loader
            ml_pred_df = load_model_and_predict(ml_open, model)

            # --- Show predictions ---
            st.dataframe(ml_pred_df[[
                "symbol", "sector", "entry_date", "entry", "target", 
                "predicted_exit", "abs_error", "pct_error", "exit_date"
            ]].sort_values("entry_date", ascending=False))

            # --- Download button ---
            csv = ml_pred_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download ML Predictions (Open Trades)", 
                csv, 
                "predicted_open_trades.csv", 
                "text/csv"
            )
