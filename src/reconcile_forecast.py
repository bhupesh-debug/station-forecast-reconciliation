"""Bottom-up vs top-down vs MinT reconciliation on a rolling-origin backtest.

Hierarchy: enterprise (total) -> 17 stations. Base forecasts are Holt-Winters
ETS per series. MinT follows Wickramasuriya et al. (2019) with a shrinkage
covariance estimate of in-sample one-step residuals.
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")
HORIZON, N_FOLDS = 6, 8
METHODS = ["base", "bottom_up", "top_down", "mint"]


def load_wide(path="data/synthetic/station_monthly.csv"):
    df = pd.read_csv(path, parse_dates=["month"])
    return df.pivot(index="month", columns="station", values="revenue")


def summing_matrix(n):
    return np.vstack([np.ones((1, n)), np.eye(n)])


def base_forecast(y, h):
    m = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add",
                             seasonal_periods=12).fit()
    return np.asarray(m.forecast(h)), np.asarray(y - m.fittedvalues)


def shrink_cov(res, lam=0.5):
    """Shrink the sample covariance toward its diagonal."""
    S = np.cov(res, rowvar=False)
    return (1 - lam) * S + lam * np.diag(np.diag(S))


def reconcile(base, res, S, method, props=None):
    """base: (h, 1+n) forecasts [total, stations]. Returns (h, 1+n)."""
    if method == "base":
        return base
    if method == "bottom_up":
        return (S @ base[:, 1:].T).T
    if method == "top_down":
        return (S @ (props[:, None] * base[:, [0]].T)).T
    if method == "mint":
        Wi = np.linalg.inv(shrink_cov(res))
        G = np.linalg.inv(S.T @ Wi @ S) @ S.T @ Wi
        return (S @ G @ base.T).T
    raise ValueError(method)


def mape(a, f):
    return float(np.mean(np.abs((a - f) / a)) * 100)


def backtest(wide):
    n = wide.shape[1]; S = summing_matrix(n)
    series = pd.concat([wide.sum(axis=1).rename("TOTAL"), wide], axis=1)
    T = len(series)
    origins = [T - HORIZON * (N_FOLDS - i) for i in range(N_FOLDS)]
    ent = {m: [] for m in METHODS}; stn = {m: [] for m in METHODS}; gaps = []
    for o in origins:
        train, test = series.iloc[:o], series.iloc[o:o + HORIZON]
        fc, rs = zip(*[base_forecast(train[c], HORIZON) for c in series.columns])
        base, res = np.column_stack(fc), np.column_stack(rs)
        last = train.iloc[-24:, 1:].sum()
        props = (last / last.sum()).values
        for m in METHODS:
            rec = reconcile(base, res, S, m, props)
            ent[m].append(mape(test["TOTAL"].values, rec[:, 0]))
            stn[m].append(np.mean([mape(test.iloc[:, j + 1].values, rec[:, j + 1]) for j in range(n)]))
        gaps.append((np.abs(base[:, 1:].sum(axis=1) - base[:, 0]) / base[:, 0] * 100).mean())
    r = lambda d: {m: round(float(np.mean(v)), 2) for m, v in d.items()}
    return {"folds": N_FOLDS, "horizon_months": HORIZON, "enterprise_mape": r(ent),
            "station_mape": r(stn), "mean_base_incoherence_pct": round(float(np.mean(gaps)), 2)}


def flag_stations(wide, z=2.0):
    """Flag stations whose latest YoY growth deviates from the peer set."""
    yoy = wide.pct_change(12).iloc[-1]
    zs = (yoy - yoy.mean()) / yoy.std()
    return pd.DataFrame({"yoy_growth": yoy, "zscore": zs, "flag": zs.abs() > z}).sort_values("zscore")


def main():
    wide = load_wide()
    res = backtest(wide)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/reconciliation_results.json").write_text(json.dumps(res, indent=2))
    flag_stations(wide).to_csv("reports/station_flags.csv")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="mint", help="all methods are always compared")
    ap.parse_args(); main()
