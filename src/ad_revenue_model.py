"""Does knowing about political cycles and sports events improve revenue prediction?

Train on 2017-2023, test on 2024-2025 (2024 is an election year). Revenue is indexed
to each station's training mean so R^2 measures shape, not market size.
NOTE: election/sports effects are injected by the synthetic generator, so this
checks the pipeline can recover a known effect, not that the effect exists.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

BASE = ["trend"] + [f"m{i}" for i in range(2, 13)]
DRIVERS = ["is_election_year", "political_window", "major_sports_event"]


def build(df):
    df = df.copy()
    df["trend"] = (df["month"].dt.year - 2017) * 12 + df["month"].dt.month
    for i in range(2, 13):
        df[f"m{i}"] = (df["month"].dt.month == i).astype(int)
    return df


def fit_eval(train, test, features):
    per_station = {}
    pred = pd.Series(index=test.index, dtype=float)
    for s, g in train.groupby("station"):
        m = Ridge(alpha=1e-3).fit(g[features], np.log(g["revenue"]))
        te = test[test.station == s]
        pred.loc[te.index] = np.exp(m.predict(te[features]))
    return pred


def main(path="data/synthetic/station_monthly.csv"):
    df = build(pd.read_csv(path, parse_dates=["month"]))
    train, test = df[df.month < "2024-01-01"], df[df.month >= "2024-01-01"]
    mean = train.groupby("station")["revenue"].mean()
    idx = lambda d, s: s / d["station"].map(mean)
    out = {}
    for name, feats in [("without_drivers", BASE), ("with_drivers", BASE + DRIVERS)]:
        pred = fit_eval(train, test, feats)
        y, p = idx(test, test["revenue"]), idx(test, pred)
        ely = test["is_election_year"] == 1
        out[name] = {"r2_all_test_months": round(float(r2_score(y, p)), 3),
                     "r2_election_year_months": round(float(r2_score(y[ely], p[ely])), 3)}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/ad_revenue_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
