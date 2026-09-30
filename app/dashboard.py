"""
Prototype operational dashboard (spec Sec.27):

    streamlit run app/dashboard.py

Calls the FastAPI service (app/api.py) rather than loading the model itself, so the dashboard
and any other client share exactly one inference path. Start the API first:

    uvicorn app.api:app --port 8000
    streamlit run app/dashboard.py -- --api http://localhost:8000
"""
import sys
from pathlib import Path

import requests
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

st.set_page_config(page_title="BustCast -- Forecast Bust Detection", layout="wide")

API = st.sidebar.text_input("API base URL", "http://localhost:8000")

st.title("AI-Based Forecast Bust Detection -- Medium-Range Weather Forecasts")
st.caption("NCMRWF / Ministry of Earth Sciences -- prototype dashboard (spec Sec.27)")


@st.cache_data(ttl=60)
def get_init_dates(api, split):
    r = requests.get(f"{api}/init_dates", params={"split": split}, timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def get_forecast(api, init_date, threshold):
    r = requests.post(f"{api}/forecast", json={"init_date": init_date,
                                                 "include_explanations": True,
                                                 "explanation_prob_threshold": threshold}, timeout=60)
    r.raise_for_status()
    return r.json()


col1, col2, col3 = st.columns([2, 2, 2])
with col1:
    split = st.selectbox("Data split to browse", ["test", "calib", "val", "train"], index=0)
try:
    dates = get_init_dates(API, split)
except Exception as e:
    st.error(f"Could not reach the API at {API}. Is `uvicorn app.api:app` running? ({e})")
    st.stop()

if not dates:
    st.warning(f"No init dates available for split='{split}'.")
    st.stop()

with col2:
    init_date = st.selectbox("Forecast initialization date", dates, index=len(dates) - 1)
with col3:
    explain_threshold = st.slider("Explanation flag threshold (P(bust) >)", 0.1, 0.9, 0.5, 0.05)

data = get_forecast(API, init_date, explain_threshold)
df = pd.DataFrame(data["grid_predictions"])

lead = st.select_slider("Lead day", options=sorted(df["lead_day"].unique().tolist()))
sub = df[df["lead_day"] == lead]

st.subheader(f"Forecast issued {init_date} -- Day {lead}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Points shown", len(sub))
m2.metric("Mean bust probability", f"{sub['bust_probability'].mean():.2f}")
m3.metric("Mean confidence", f"{sub['confidence'].mean():.2f}")
m4.metric("High/very-high risk points", int((sub["risk_category"].isin(["high", "very_high"])).sum()))

tab_conf, tab_bust, tab_err, tab_areas, tab_explain = st.tabs(
    ["Forecast Confidence Map", "Bust Probability Map", "Predicted Error Map",
     "Error-Prone Areas", "Explanation Panel"])

with tab_conf:
    fig = px.scatter_geo(sub, lat="latitude", lon="longitude", color="confidence",
                          color_continuous_scale="Greens", scope="asia",
                          hover_data=["point_id", "risk_category"], title=f"Confidence -- Day {lead}")
    fig.update_geos(lataxis_range=[5, 38], lonaxis_range=[65, 100])
    st.plotly_chart(fig, use_container_width=True)

with tab_bust:
    fig = px.scatter_geo(sub, lat="latitude", lon="longitude", color="bust_probability",
                          color_continuous_scale="Reds", scope="asia",
                          hover_data=["point_id", "risk_category"], title=f"P(bust) -- Day {lead}")
    fig.update_geos(lataxis_range=[5, 38], lonaxis_range=[65, 100])
    st.plotly_chart(fig, use_container_width=True)

with tab_err:
    fig = px.scatter_geo(sub, lat="latitude", lon="longitude", color="predicted_error",
                          color_continuous_scale="Magma", scope="asia",
                          hover_data=["point_id"], title=f"Predicted composite error (z) -- Day {lead}")
    fig.update_geos(lataxis_range=[5, 38], lonaxis_range=[65, 100])
    st.plotly_chart(fig, use_container_width=True)

with tab_areas:
    reg = pd.DataFrame(data["regions"])
    reg = reg[reg["lead_day"] == lead].sort_values("mean_bust_probability", ascending=False)
    st.markdown("**Region-wise aggregation** (KMeans-clustered regions over the 900 points)")
    st.dataframe(reg, use_container_width=True, hide_index=True)
    st.markdown("**High-risk points at this lead**")
    st.dataframe(sub[sub["risk_category"].isin(["high", "very_high"])]
                 .sort_values("bust_probability", ascending=False), use_container_width=True, hide_index=True)

with tab_explain:
    ex = pd.DataFrame(data["explanations"])
    ex = ex[ex["lead_day"] == lead] if len(ex) else ex
    if len(ex) == 0:
        st.info("No points crossed the explanation threshold at this lead day. Lower the "
                 "threshold in the sidebar controls above to see more.")
    else:
        for _, row in ex.iterrows():
            with st.expander(f"Point {row['point_id']} -- P(bust)={row['bust_probability']:.2f}"):
                st.write("Dominant meteorological factors (input-gradient saliency, "
                         "see bustcast/explain.py):")
                for f in row["dominant_factors"]:
                    st.markdown(f"- {f}")

st.divider()
st.caption("Predicted error is the calibrated, z-normalised composite error (see config.yaml "
           "`targets`). Confidence = 1 - calibrated P(bust). This is a prototype for operational "
           "review, not an official NCMRWF product.")
