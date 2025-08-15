import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np


def load_model_and_predict(ml_df, model):
    df = ml_df[ml_df["days_before_entry"] == 0].copy()
    df = df.dropna()

    le_symbol = LabelEncoder()
    df["symbol_encoded"] = le_symbol.fit_transform(df["symbol"])

    le_sector = LabelEncoder()
    df["sector_encoded"] = le_sector.fit_transform(df["sector"])

    features = [
        "sma_10", "sma_20", "sma_50", "sma_200",
        "stop_loss", "open", "high", "low", "close", "volume",
        "symbol_encoded", "sector_encoded"
    ]

    df["predicted_exit"] = model.predict(df[features])
    df["abs_error"] = abs(df["predicted_exit"] - df["target"])
    df["pct_error"] = 100 * df["abs_error"] / df["target"]
    return df
