import streamlit as st
import pandas as pd
import joblib
from strategy import run_strategy
from ml_features import build_ml_dataset
from model_loader import load_model_and_predict

# --- Title ---
st.title("📈 Smart Backtester + ML Exit Predictor")

# --- Load data from file in repo ---
@st.cache_data
def load_data():
    df = pd.read_csv("stocks.csv", parse_dates=["date"])
    df = df.sort_values(["symbol", "date"])
    return df

df = load_data()

# --- Spinner wrapper for heavy logic ---
with st.spinner("⏳ Running strategy and predicting exit prices..."):
    
    # --- Run strategy to detect trades ---
    trades = run_strategy(df)

    if trades.empty:
        st.warning("No valid trades found.")
    else:
        st.success(f"✅ {len(trades)} trades detected")

        # --- Build ML dataset ---
        ml_df = build_ml_dataset(df, trades)

        # --- Load model and predict ---
        model = joblib.load("model.pkl")
        ml_pred_df = load_model_and_predict(ml_df, model)

        # --- Show results ---
        st.dataframe(ml_pred_df[[
            "symbol", "sector", "entry_date", "entry", "target", 
            "predicted_exit", "abs_error", "pct_error", "pct_return", "exit_date"
        ]].sort_values("entry_date", ascending=False))

        # --- Downloadable CSV ---
        csv = ml_pred_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download results CSV", csv, "predicted_trades.csv", "text/csv")
