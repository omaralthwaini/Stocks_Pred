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

        ml_df = build_ml_dataset(df, trades)
        model = load_model()
        ml_pred_df = load_model_and_predict(ml_df, model)

        # --- Show table ---
        st.dataframe(
            ml_pred_df[[
                "symbol", "sector", "entry_date", "entry", "target",
                "predicted_exit", "abs_error", "pct_error",
                "pct_return", "exit_date"
            ]].sort_values("entry_date", ascending=False),
            use_container_width=True
        )

        # --- Download ---
        csv = ml_pred_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download results CSV",
            csv,
            "predicted_trades.csv",
            "text/csv"
        )
