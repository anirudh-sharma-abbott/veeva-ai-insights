import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

st.set_page_config(page_title="TRx/NBRx Forecasting", layout="wide")
st.title("📈 TRx/NBRx Forecasting (Prophet + Plotly)")
st.markdown(
    "*Forecasting prescription trends helps identify future HCP behavior, enabling proactive sales planning. We use Prophet by Meta – a robust time-series forecasting model that handles seasonality, trends, and holiday effects with minimal manual tuning.*"
)


@st.cache_data
def load_data():
    df = pd.read_csv("data/trx_nbrx_trends.csv")
    df["month"] = pd.to_datetime(df["month"])
    return df

df = load_data()
hcp_list = df["hcp_id"].unique()
selected_hcp = st.selectbox("Select HCP", hcp_list)

hcp_data = df[df["hcp_id"] == selected_hcp].sort_values("month")

# Prophet for TRx
trx_df = hcp_data[["month", "trxs"]].rename(columns={"month": "ds", "trxs": "y"})
model_trx = Prophet()
model_trx.fit(trx_df)
future_trx = model_trx.make_future_dataframe(periods=3, freq="MS")
forecast_trx = model_trx.predict(future_trx)

# Prophet for NBRx
nbrx_df = hcp_data[["month", "nbrxs"]].rename(columns={"month": "ds", "nbrxs": "y"})
model_nbrx = Prophet()
model_nbrx.fit(nbrx_df)
future_nbrx = model_nbrx.make_future_dataframe(periods=3, freq="MS")
forecast_nbrx = model_nbrx.predict(future_nbrx)

# Plotly chart
fig = go.Figure()

# TRx
fig.add_trace(go.Scatter(x=trx_df["ds"], y=trx_df["y"],
                         mode="lines+markers", name="TRx - Actual", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=forecast_trx["ds"], y=forecast_trx["yhat"],
                         mode="lines", name="TRx - Forecast", line=dict(dash="dash", color="blue")))
fig.add_trace(go.Scatter(x=forecast_trx["ds"], y=forecast_trx["yhat_upper"],
                         mode="lines", name="TRx - Upper CI", line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=forecast_trx["ds"], y=forecast_trx["yhat_lower"],
                         mode="lines", fill='tonexty', name="TRx CI", line=dict(width=0), fillcolor='rgba(0,0,255,0.1)', showlegend=False))

# NBRx
fig.add_trace(go.Scatter(x=nbrx_df["ds"], y=nbrx_df["y"],
                         mode="lines+markers", name="NBRx - Actual", line=dict(color="green")))
fig.add_trace(go.Scatter(x=forecast_nbrx["ds"], y=forecast_nbrx["yhat"],
                         mode="lines", name="NBRx - Forecast", line=dict(dash="dash", color="green")))
fig.add_trace(go.Scatter(x=forecast_nbrx["ds"], y=forecast_nbrx["yhat_upper"],
                         mode="lines", name="NBRx - Upper CI", line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=forecast_nbrx["ds"], y=forecast_nbrx["yhat_lower"],
                         mode="lines", fill='tonexty', name="NBRx CI", line=dict(width=0), fillcolor='rgba(0,128,0,0.1)', showlegend=False))

fig.update_layout(
    title=f"Forecast for HCP: {selected_hcp}",
    xaxis_title="Month",
    yaxis_title="TRx / NBRx Count",
    legend_title="Legend",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# Display KPIs
# st.markdown("### 📊 Forecasted Values (Next Month)")
# next_month = forecast_trx["ds"].iloc[-3].strftime("%b %Y")
# st.columns(3)[0].metric(f"TRx - {next_month}", f"{forecast_trx['yhat'].iloc[-3]:.1f}")
# st.columns(3)[1].metric(f"NBRx - {next_month}", f"{forecast_nbrx['yhat'].iloc[-3]:.1f}")

# Display KPIs
st.markdown("### 📊 Forecasted Values (Next Month)")

# ✅ Define this BEFORE metrics
next_month = forecast_trx["ds"].iloc[-3].strftime("%b %Y")

col1, col2 = st.columns(2)
with col1:
    st.metric(f"TRx - {next_month}", f"{forecast_trx['yhat'].iloc[-3]:.1f}")
with col2:
    st.metric(f"NBRx - {next_month}", f"{forecast_nbrx['yhat'].iloc[-3]:.1f}")



with st.expander("ℹ️ About this Page"):
    st.markdown("""
    This interactive dashboard forecasts **TRx (Total Prescriptions)** and **NBRx (New-to-Brand Prescriptions)** for individual HCPs using Meta's **Prophet time series model**.

    **Key Features**:
    - 📅 Historical data and 3-month forecast visualization
    - 🧠 Advanced forecasting using Prophet (captures seasonality, trend, uncertainty)
    - 📉 Confidence intervals (shaded area) to express forecast reliability
    - 📊 Hover tooltips, zoom, and pan via Plotly for deeper insight

    **Use Case**:
    - Identify HCPs with potential upward/downward momentum
    - Tailor engagement plans based on expected TRx/NBRx trends
    - Inform sales strategy and territory management

    This demo is part of a larger effort to equip reps and managers with AI-powered tools for smarter decision-making.
    """)