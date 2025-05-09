import streamlit as st
from PIL import Image

# --- Page Setup ---
st.set_page_config(page_title='Veeva AI Insights', layout='wide')

# --- Layout for Logo + Title ---
col_logo, col_title = st.columns([1, 5])
with col_logo:
    logo = Image.open("static/abbott_logo.jpg")  # Update path if needed
    st.image(logo, width=100)
with col_title:
    st.markdown("<h1 style='margin-bottom:0;'>Veeva AI Insights Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='margin-top:0; color: gray;'>AI-Powered Decision Support for Field Excellence</h4>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --- Overview Cards ---
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    st.subheader("HCP Targeting")
    st.markdown("""
    • ML-based prioritization of HCPs  
    • Predicts conversion potential using TRx/NBRx, specialty, and call history  
    """)

with col2:
    st.subheader("Next Best Action Assistant")
    st.markdown("""
    • GPT-based strategy suggestions  
    • Contextual guidance on calls, samples, and coaching  
    • Designed to bridge engagement gaps  
    """)

with col3:
    st.subheader("TRx Forecasting")
    st.markdown("""
    • Visual time-series forecasting using Prophet/ARIMA  
    • Highlights monthly TRx and NBRx trends  
    • Tracks performance at the HCP level  
    """)

with col4:
    st.subheader("Geo Routing Optimization")
    st.markdown("""
    • Route planning with real-road distances (Google Maps API)  
    • Prioritization via urgency scores and TSP heuristics  
    • Minimizes travel, maximizes field efficiency  
    """)

st.markdown("<hr>", unsafe_allow_html=True)

# --- Callout Box ---
st.success("""
This demo showcases how machine learning, GenAI, and real-world routing logic can elevate field decision-making in Veeva CRM.
""")

# --- Footer ---
st.markdown("<div style='text-align: center; color: gray;'>Built by the ADC IT – Commercial Excellence Team</div>", unsafe_allow_html=True)
