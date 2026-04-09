# backend/risk.py
from __future__ import annotations
import math
import json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TOP10_JSON = PROCESSED_DIR / "top10_uptrend.json"
RISK_CSV   = PROCESSED_DIR / "risk_metrics.csv"
RANK_CSV   = PROCESSED_DIR / "risk_ranking.csv"
RISK_JSON  = PROCESSED_DIR / "risk_metrics.json"


BENCHMARK = "^GSPC"     
LOOKBACK_DAYS = 32     
TRADING_DAYS = 252
CONF_LEVEL = 0.95
Z_95 = 1.65
RISK_FREE_ANNUAL = 0.02


def fetch_close(symbol: str, start: datetime, end: datetime) -> pd.Series:
    df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        return pd.Series(dtype=float, name=symbol)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].copy()
    s.name = symbol
    return s

def annualize_vol(daily_std: float) -> float:
    return float(daily_std) * math.sqrt(TRADING_DAYS)

def compute_beta(stock_ret: pd.Series, mkt_ret: pd.Series) -> float:
    aligned = pd.concat([stock_ret, mkt_ret], axis=1, join="inner").dropna()
    if len(aligned) < 10:
        return np.nan
    cov = np.cov(aligned.iloc[:,0], aligned.iloc[:,1], ddof=1)[0,1]
    var_m = np.var(aligned.iloc[:,1], ddof=1)
    return cov / var_m if var_m != 0 else np.nan

def parametric_var_pct(mu: float, sigma: float, z: float = Z_95) -> float:
    # VaR% as positive one-day loss magnitude under Normal assumption
    # Example: 0.025 -> 2.5%
    return max(0.0, -(mu - z * sigma))

def historical_var_pct(returns: pd.Series, conf: float = CONF_LEVEL) -> float:
    # Empirical 1-day VaR: 5th percentile loss (positive number)
    if len(returns) == 0:
        return np.nan
    q = np.percentile(returns, (1 - conf) * 100)  # 5th percentile for 95%
    return max(0.0, -float(q))

def sortino_ratio(returns: pd.Series, rf_annual: float = RISK_FREE_ANNUAL) -> float:
    if len(returns) == 0:
        return np.nan
    rf_daily = rf_annual / TRADING_DAYS
    downside = returns[returns < 0]
    if len(downside) == 0:
        return np.nan
    downside_std = math.sqrt(np.mean(np.square(downside)))
    if downside_std == 0:
        return np.nan
    return ((returns.mean() - rf_daily) / downside_std) * math.sqrt(TRADING_DAYS)

def load_top10_symbols() -> list[str]:
    if TOP10_JSON.exists():
        return json.loads(TOP10_JSON.read_text())
    # fallback: infer from features.csv if top10 file missing
    feats = PROCESSED_DIR / "features.csv"
    if feats.exists():
        df = pd.read_csv(feats, parse_dates=["date"])
        latest = df.sort_values("date").groupby("symbol").tail(1)
        return latest.sort_values("momentum_30", ascending=False).head(10)["symbol"].tolist()
    # very last fallback
    return ["INTC","ASML","AAPL","TSM","CRM","TSLA","SAP","ORCL","ABBV","SONY"]

def risk_level_from_score(score: float) -> str:
    # score is 0..1; thresholds can be tuned
    if pd.isna(score):
        return "Unknown"
    if score >= 0.66:
        return "High"
    if score >= 0.33:
        return "Medium"
    return "Low"

# -------------------- Main ---------------------
if __name__ == "__main__":
    end = datetime.utcnow()
    start = end - timedelta(days=LOOKBACK_DAYS)

    symbols = load_top10_symbols()
    print(f"Fetching ~1M data for: {', '.join(symbols)}")

    bench_close = fetch_close(BENCHMARK, start, end)
    bench_ret = bench_close.pct_change()

    rows = []
    for sym in symbols:
        close = fetch_close(sym, start, end)
        if close.empty:
            print(f"⚠ No data for {sym}")
            rows.append({"symbol": sym, "latest_price": np.nan, "window_days": 0,
                         "vol_annual_pct": np.nan, "beta": np.nan,
                         "var_1d_95_pct": np.nan, "var_hist_1d_95_pct": np.nan,
                         "sharpe": np.nan, "sortino": np.nan})
            continue

        ret = close.pct_change().dropna()
        mu = ret.mean() if len(ret) else np.nan
        sd = ret.std(ddof=1) if len(ret) else np.nan

        vol_ann = annualize_vol(sd) if not pd.isna(sd) else np.nan
        beta    = compute_beta(ret, bench_ret)
        var_p   = parametric_var_pct(mu, sd) if (not pd.isna(mu) and not pd.isna(sd)) else np.nan
        var_h   = historical_var_pct(ret)
        rf_daily = RISK_FREE_ANNUAL / TRADING_DAYS
        sharpe  = (((mu - rf_daily) / sd) * math.sqrt(TRADING_DAYS)) if (sd and sd > 0) else np.nan
        sortino = sortino_ratio(ret)

        rows.append({
            "symbol": sym,
            "latest_price": float(close.iloc[-1]),
            "window_days": int(len(close)),
            "vol_annual_pct": float(vol_ann) if not pd.isna(vol_ann) else np.nan,
            "beta": float(beta) if not pd.isna(beta) else np.nan,
            "var_1d_95_pct": float(var_p) if not pd.isna(var_p) else np.nan,
            "var_hist_1d_95_pct": float(var_h) if not pd.isna(var_h) else np.nan,
            "sharpe": float(sharpe) if not pd.isna(sharpe) else np.nan,
            "sortino": float(sortino) if not pd.isna(sortino) else np.nan
        })

        print(f"✅ {sym}: Vol={vol_ann:.2%}  Beta={beta:.2f}  VaR(N)={var_p:.2%}  VaR(H)={var_h:.2%}  Sharpe={sharpe:.2f}  Sortino={sortino:.2f}")

    if not rows:
        print("No valid risk rows. Exiting.")
        raise SystemExit

    # Save raw metrics
    df = pd.DataFrame(rows)
    df.to_csv(RISK_CSV, index=False)

    # Blended risk score (0..1) and risk level
    work = df.copy()
    for col in ["var_1d_95_pct", "vol_annual_pct", "beta"]:
        if work[col].notna().sum() > 0:
            c = work[col]
            rng = (c.max() - c.min())
            work[col + "_norm"] = (c - c.min()) / rng if (rng and rng > 0) else 0.5
        else:
            work[col + "_norm"] = 0.5

    work["risk_score"] = (
        0.50 * work["var_1d_95_pct_norm"] +
        0.30 * work["vol_annual_pct_norm"] +
        0.20 * work["beta_norm"]
    )
    work["risk_level"] = work["risk_score"].apply(risk_level_from_score)

    work.sort_values("risk_score", ascending=False).to_csv(RANK_CSV, index=False)

    RISK_JSON.write_text(json.dumps({
        "as_of_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "benchmark": BENCHMARK,
        "conf_level": CONF_LEVEL,
        "z_score": Z_95,
        "lookback_days": LOOKBACK_DAYS,
        "rows": rows
    }, indent=2))

    print(f"\nSaved risk metrics  → {RISK_CSV}")
    print(f"Saved risk ranking  → {RANK_CSV}")
    print(f"Saved risk meta     → {RISK_JSON}")
