import os
import pandas as pd
import numpy as np

# --- Paths ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
MASTER = os.path.join(PROCESSED_DIR, "master_top50_companies.csv")
FEATURE_FILE = os.path.join(PROCESSED_DIR, "features.csv")

# Ensure the processed directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def engineer():
    # Load data
    df = pd.read_csv(MASTER, parse_dates=["lastTradedTime"])
    df = df.rename(columns={"lastTradedTime": "date"})

    # Ensure numeric columns
    numeric_cols = ["price", "open", "high", "low", "sharevolume", "turnover"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(by=["symbol", "date"]).reset_index(drop=True)

    # --- Basic Features ---
    df["daily_return"] = df.groupby("symbol")["price"].pct_change()
    df["momentum_7"] = df.groupby("symbol")["price"].pct_change(periods=7) * 100
    df["momentum_30"] = df.groupby("symbol")["price"].pct_change(periods=30) * 100
    df["vol_7"] = df.groupby("symbol")["daily_return"].rolling(7, min_periods=1).std().reset_index(0, drop=True)
    df["vol_30"] = df.groupby("symbol")["daily_return"].rolling(30, min_periods=1).std().reset_index(0, drop=True)
    df["avg_vol_7"] = df.groupby("symbol")["sharevolume"].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
    df["avg_vol_30"] = df.groupby("symbol")["sharevolume"].rolling(30, min_periods=1).mean().reset_index(0, drop=True)

    # --- RSI ---
    df["rsi_14"] = df.groupby("symbol")["price"].apply(lambda x: compute_rsi(x, 14)).reset_index(0, drop=True)

    # --- Moving Averages ---
    df["sma_7"] = df.groupby("symbol")["price"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df["sma_30"] = df.groupby("symbol")["price"].transform(lambda x: x.rolling(30, min_periods=1).mean())
    df["ema_7"] = df.groupby("symbol")["price"].transform(lambda x: x.ewm(span=7, adjust=False).mean())
    df["ema_30"] = df.groupby("symbol")["price"].transform(lambda x: x.ewm(span=30, adjust=False).mean())

    # --- Bollinger Bands ---
    df["bb_upper_20"] = df.groupby("symbol")["price"].transform(
        lambda x: x.rolling(20, min_periods=1).mean() + 2 * x.rolling(20, min_periods=1).std()
    )
    df["bb_lower_20"] = df.groupby("symbol")["price"].transform(
        lambda x: x.rolling(20, min_periods=1).mean() - 2 * x.rolling(20, min_periods=1).std()
    )

    # --- MACD ---
    df["ema_12"] = df.groupby("symbol")["price"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    df["ema_26"] = df.groupby("symbol")["price"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df.groupby("symbol")["macd"].transform(lambda x: x.ewm(span=9, adjust=False).mean())

    # --- Stochastic Oscillator ---
    df["low_14"] = df.groupby("symbol")["low"].transform(lambda x: x.rolling(14, min_periods=1).min())
    df["high_14"] = df.groupby("symbol")["high"].transform(lambda x: x.rolling(14, min_periods=1).max())
    df["stochastic_14"] = (df["price"] - df["low_14"]) / (df["high_14"] - df["low_14"] + 1e-8) * 100

    # --- Price Rate of Change (ROC) ---
    df["roc_7"] = df.groupby("symbol")["price"].pct_change(7) * 100
    df["roc_30"] = df.groupby("symbol")["price"].pct_change(30) * 100

    # --- Lag Features ---
    for lag in [1, 2, 3]:
        df[f"return_lag_{lag}"] = df.groupby("symbol")["daily_return"].shift(lag)

    # --- Cumulative Returns ---
    df["cum_return"] = df.groupby("symbol")["daily_return"].cumsum()

    # --- Target: forward 5-day return ---
    df["pred_return_5d"] = df.groupby("symbol")["price"].shift(-5) / df["price"] - 1

    # Drop rows with missing values
    df = df.dropna()

    # Save features
    df.to_csv(FEATURE_FILE, index=False)
    return FEATURE_FILE


if __name__ == "__main__":
    out = engineer()
    print("Features saved to:", out)
