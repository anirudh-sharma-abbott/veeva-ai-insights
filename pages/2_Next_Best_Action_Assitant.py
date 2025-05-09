import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Validate key
if not api_key:
    st.error("❌ OpenAI API key not found. Please check your .env file.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Page config
st.set_page_config(page_title="NBA Assistant", layout="wide")
st.title("🤖 Next Best Action (NBA) Assistant - AI Suggestions")

# Load HCP data
@st.cache_data
def load_data():
    return pd.read_csv("data/hcp_profiles.csv")

df = load_data()

# UI for HCP selection
selected_hcp = st.selectbox("Select HCP", df["name"])
hcp_row = df[df["name"] == selected_hcp].iloc[0]

# Show HCP snapshot
st.markdown("### 👤 HCP Snapshot")
st.dataframe(hcp_row.to_frame().T, use_container_width=True)

# Prepare prompt for OpenAI
prompt = f"""
You are a sales enablement AI assistant for a medical device company.
Based on the following HCP profile data, recommend the next best action for the sales rep to take this week.

Be specific and consider if a CLM email, in-person call, sample drop, or coaching material would be most effective.
Explain *why* you are suggesting that action in a short paragraph.

HCP Profile:
{hcp_row.to_dict()}
"""

# AI Recommendation Section
if st.button("🧠 Generate AI Recommendation"):
    with st.spinner("Thinking..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a medical sales strategy assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            suggestion = response.choices[0].message.content
            st.markdown("### ✅ AI Recommendation")
            st.success(suggestion)
        except Exception as e:
            st.error(f"OpenAI Error: {e}")

# --- Page Information Footer ---
st.markdown("---")
st.markdown("#### ℹ️ About This Page")
st.info("""
The **Next Best Action (NBA) Assistant** uses AI to analyze HCP profile data and recommend personalized actions for sales reps.

It leverages:
- HCP segment and specialty
- Recent activity (TRx/NBRx)
- Model-generated **conversion score**
- GPT-4 for recommendation reasoning

This tool is designed to help sales teams make informed decisions, prioritize outreach, and improve HCP engagement outcomes.
""")
