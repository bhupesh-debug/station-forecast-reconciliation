"""Generate synthetic monthly data for 17 station markets.

Injected effects (known by construction, see README "Data and Its Limits"):
  - linear trend + annual seasonality, scaled by market size
  - election years (even years): political ad lift concentrated in Sep-Nov
  - major sports events: revenue lift in event months
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

N_STATIONS = 17
START, END = "2017-01-01", "2025-12-01"


def generate_stations(rng):
    sizes = rng.lognormal(mean=0.0, sigma=0.45, size=N_STATIONS)
    return pd.DataFrame({"station": [f"ST{i+1:02d}" for i in range(N_STATIONS)],
                         "size": sizes / sizes.mean()})


def generate_monthly(rng, stations):
    months = pd.date_range(START, END, freq="MS")
    t = np.arange(len(months))
    seas = 1 + 0.12 * np.sin(2 * np.pi * (months.month - 3) / 12)
    rows = []
    for _, s in stations.iterrows():
        base = 10.0 * s["size"]  # $M / month
        growth = 1 + rng.normal(0.02, 0.008) * t / 12
        election = ((months.year % 2 == 0) & months.month.isin([9, 10, 11])).astype(int)
        election_year = (months.year % 2 == 0).astype(int)
        political = election * rng.uniform(0.25, 0.55)
        sports = (rng.random(len(months)) < 0.15).astype(int)
        sports_lift = sports * rng.uniform(0.05, 0.12)
        noise = rng.normal(0, 0.03, len(months))
        revenue = base * growth * seas * (1 + political + sports_lift + noise)
        opex = revenue * rng.uniform(0.62, 0.72) * (1 + rng.normal(0, 0.02, len(months)))
        rows.append(pd.DataFrame({
            "month": months, "station": s["station"], "revenue": revenue, "opex": opex,
            "is_election_year": election_year, "political_window": election,
            "major_sports_event": sports}))
    return pd.concat(rows, ignore_index=True)


def generate_capex(rng, stations, n=400):
    df = pd.DataFrame({
        "project_id": range(n),
        "station": rng.choice(stations["station"], n),
        "budget_musd": rng.lognormal(0.3, 0.8, n),
        "planned_months": rng.integers(3, 25, n),
        "vendor_count": rng.integers(1, 6, n),
        "scope_changes": rng.poisson(1.2, n),
        "is_regulatory": rng.integers(0, 2, n),
        "prior_overruns_at_station": rng.poisson(1.0, n),
    })
    z = (-2.2 + 0.35 * df.scope_changes + 0.25 * np.log(df.budget_musd)
         + 0.04 * df.planned_months + 0.20 * df.vendor_count
         + 0.30 * df.prior_overruns_at_station - 0.4 * df.is_regulatory)
    df["overrun"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df


def main(seed=42, out="data/synthetic"):
    rng = np.random.default_rng(seed)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    stations = generate_stations(rng)
    monthly = generate_monthly(rng, stations)
    capex = generate_capex(rng, stations)
    monthly.to_csv(out / "station_monthly.csv", index=False)
    capex.to_csv(out / "capex_projects.csv", index=False)
    print(f"Wrote {len(monthly)} station-month rows and {len(capex)} CapEx projects to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args().seed)
