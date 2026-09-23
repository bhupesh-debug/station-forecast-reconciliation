"""Station-level consolidation view. Run: streamlit run src/dashboard.py
(run the pipeline and model scripts first so reports/ exists)"""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Station Forecast Reconciliation", layout="wide")
st.title("Station Forecast Reconciliation")

rec = json.loads(Path("reports/reconciliation_results.json").read_text())
c1, c2 = st.columns(2)
c1.subheader("Enterprise MAPE by method (%)")
c1.bar_chart(pd.Series(rec["enterprise_mape"]))
c2.subheader("Mean station MAPE by method (%)")
c2.bar_chart(pd.Series(rec["station_mape"]))

st.subheader("Stations diverging from peers (latest YoY growth, z-score)")
flags = pd.read_csv("reports/station_flags.csv", index_col=0)
st.dataframe(flags.style.format({"yoy_growth": "{:.1%}", "zscore": "{:.2f}"}))

st.subheader("Revenue history")
df = pd.read_csv("data/synthetic/station_monthly.csv", parse_dates=["month"])
pick = st.multiselect("Stations", sorted(df.station.unique()), default=["ST01", "ST02"])
st.line_chart(df[df.station.isin(pick)].pivot(index="month", columns="station", values="revenue"))
