
import numpy as np
import pandas as pd
from arch import arch_model

def garch_volatility_forecast(series: pd.Series) -> float | None:
    px = pd.Series(series).astype(float).replace([np.inf,-np.inf], np.nan).dropna()
    px = px[px > 0]
    rets = np.log(px / px.shift(1)).replace([np.inf,-np.inf], np.nan).dropna()
    if len(rets) < 50: return None
    try:
        am = arch_model(rets, vol="GARCH", p=1, q=1, mean="Zero", dist="t")
        res = am.fit(disp="off")
        var1 = float(res.forecast(horizon=1, reindex=False).variance.iloc[-1,0])
        ann_vol = float(np.sqrt(var1) * np.sqrt(252))
        return ann_vol if np.isfinite(ann_vol) and ann_vol > 0 else None
    except Exception:
        return None

def attach_garch_risk_index(df_prices: pd.DataFrame, trades_df: pd.DataFrame,
                            base_vol_dec: float = 0.20, lookback: int | None = 252) -> pd.DataFrame:
    df_sorted = df_prices.sort_values(["symbol","date"])
    groups = {sym: g[["date","close"]].reset_index(drop=True) for sym,g in df_sorted.groupby("symbol")}
    out = []
    for _, r in trades_df.iterrows():
        sym = r["symbol"]; entry_date = pd.to_datetime(r["entry_date"])
        if sym not in groups: out.append(np.nan); continue
        g = groups[sym]; hist = g[g["date"] < entry_date]["close"]
        if lookback and lookback > 0: hist = hist.tail(lookback)
        if len(hist) < 51: out.append(np.nan); continue
        ann = garch_volatility_forecast(hist)
        out.append(float(min(1.0, base_vol_dec / ann))) if ann and np.isfinite(ann) and ann>1e-9 else out.append(np.nan)
    t2 = trades_df.copy(); t2["garch_risk_index"] = out
    return t2

def optimize_thresholds_per_symbol_closed(trades_all: pd.DataFrame, *, step: float = 0.05, min_trades: int = 3):
    closed = trades_all[trades_all["exit_date"].notna()].copy()
    if closed.empty:
        return ({s: 0.00 for s in sorted(trades_all["symbol"].unique())},
                pd.DataFrame(columns=["symbol","threshold","trades","total_return_%","win_rate_%"]))
    start_ts = closed["entry_date"].min(); end_ts = closed["exit_date"].max()
    thr_values = np.round(np.arange(0.0, 1.0 + 1e-9, step), 2)
    rows, best_map = [], {}
    def simulate_oaat(cand: pd.DataFrame, thr: float):
        t = cand[cand["garch_risk_index"].ge(thr)].sort_values("entry_date")
        cap = 10000.0; rets=[]
        for _,r in t.iterrows():
            entry = float(r["entry"])
            if pd.notna(r["exit_date"]) and r["exit_date"] <= end_ts:
                exit_px = float(r["exit_price"]); ret = (exit_px/entry -1)*100
            else:
                exit_px = float(r["latest_close"]); ret = (exit_px/entry -1)*100
            cap = cap*(1+ret/100); rets.append(ret)
        total = (cap/10000.0 -1)*100; trades = len(rets)
        win_rate = (pd.Series(rets)>0).mean() if trades else np.nan
        return total, trades, win_rate
    for sym, cand in closed.groupby("symbol"):
        best = None
        for thr in thr_values:
            total, trades, win = simulate_oaat(cand, thr)
            rows.append({"symbol":sym,"threshold":thr,"trades":trades,"total_return_%":total,"win_rate_%":(win*100 if pd.notna(win) else np.nan)})
            if trades < min_trades: continue
            candidate = (total, trades, -thr, thr)
            if (best is None) or (candidate > best): best = candidate
        best_map[sym] = (best[3] if best is not None else 0.00)
    return best_map, pd.DataFrame(rows)
