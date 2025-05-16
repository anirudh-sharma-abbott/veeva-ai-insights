import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import plotly.graph_objects as go

# --- Load API key ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OpenAI API key not found.")
    st.stop()

client = OpenAI(api_key=api_key)

# --- Page setup ---
st.set_page_config(page_title="NBA Assistant", layout="wide")
st.title("🤖 Next Best Action (NBA) Assistant - Multi-Agent Chatbot")

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/hcp_profiles.csv")
    df["last_call_date"] = pd.to_datetime(df["last_call_date"], errors="coerce")
    df["days_since_last_call"] = (datetime.now() - df["last_call_date"]).dt.days
    return df

df = load_data()
selected_hcp = st.selectbox("Select HCP", df["name"])

# --- Reset chat history if HCP changes ---
if "selected_hcp" not in st.session_state or st.session_state.selected_hcp != selected_hcp:
    st.session_state.selected_hcp = selected_hcp
    st.session_state.messages = []

hcp_row = df[df["name"] == selected_hcp].iloc[0]
conversion_score = hcp_row["conversion_score"]
trxs_last_month = int(hcp_row["trxs_last_month"])
nbrxs_last_month = int(hcp_row["nbrxs_last_month"])

# --- Display KPI Metrics ---
st.markdown("### 👤 HCP Snapshot")
col1, col2, col3 = st.columns(3)

def get_score_badge(score):
    if score < 0.4:
        color = "#e74c3c"; label = "Low"
    elif score < 0.6:
        color = "#f1c40f"; label = "Medium"
    else:
        color = "#2ecc71"; label = "High"
    return f"""
    <span style="background-color:{color}; color:white; padding:4px 10px; border-radius:8px; font-weight:bold; font-size:0.85rem;">
        {label} Conversion Potential
    </span>"""

col1.metric("🧠 Conversion Score", f"{conversion_score:.2f}")
col1.markdown(get_score_badge(conversion_score), unsafe_allow_html=True)
col2.metric("💊 TRx Last Month", trxs_last_month)
col3.metric("📈 NBRx Last Month", nbrxs_last_month)

# --- HCP Summary ---
st.markdown(f"""
- **Segment**: {hcp_row['segment']}
- **Specialty**: {hcp_row['specialty']}
- **Days Since Last Call**: {hcp_row['days_since_last_call']}
- **Last Call Date**: {hcp_row['last_call_date'].date() if pd.notnull(hcp_row['last_call_date']) else 'N/A'}
""")

# --- Intent Classifier ---
def classify_intent(user_input):
    classification_prompt = """
You are an intent classification agent. Given the user's message, classify it into one of these categories:

- call_strategy
- sample_strategy
- coaching_strategy
- call_trends
- rx_trends
- last_call
- general_question

Only return the category name. No explanation.
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": classification_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0,
        max_tokens=10
    )
    return response.choices[0].message.content.strip().lower()

# --- Prompt Generator ---
def generate_prompt(action_type, hcp_dict):
    return f"""
You are an AI assistant for a medical sales team.

Your task is to evaluate whether a **{action_type}** should be done this week for the following HCP.
Do not treat this as a binary decision. Instead, apply **tiered recommendation logic**:

- If conversion_score > 0.6: Strongly recommend the action.
- If conversion_score is between 0.4–0.6: Soft recommendation, suggest alternate materials or cautious engagement.
- If conversion_score < 0.4: Do not push in-person actions. Suggest low-effort engagement like email nurturing.

Also factor in:
- Days since last interaction (>21 is a good sign to re-engage)
- Recent TRx/NBRx activity (high = good signal)

Conclude your response with a clear **recommendation tier**:
- ✅ Recommended
- 🤔 Optional
- 🟡 Low Priority

Format your reasoning in bullet points, then give a one-line summary.

HCP Profile:
{hcp_dict}
"""

# --- Chatbot Section ---
st.markdown("---")
st.markdown("### 💬 Chat with the AI Assistant")
# st.caption("Ask things like: *What should I do with this HCP?*, *Show trends*, or *When was the last call?*")
st.caption("Ask things like: *What should I do with this HCP?*, *Show call trends*, *Show TRx trends*, or *When was the last call?*")


user_input = st.chat_input("Type your message...")

# --- Show chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Handle user message ---
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            intent = classify_intent(user_input)
            reply = ""

            if intent == "call_strategy":
                prompt = generate_prompt("in-person call", hcp_row.to_dict())
            elif intent == "sample_strategy":
                prompt = generate_prompt("sample drop", hcp_row.to_dict())
            elif intent == "coaching_strategy":
                prompt = generate_prompt("coaching session", hcp_row.to_dict())
            elif intent == "rx_trends":
                st.markdown("Here is the TRx vs NBRx trend for this HCP:")
                dates = pd.date_range(end=datetime.now(), periods=6, freq="M")
                trxs = [int(trxs_last_month * (0.9 + 0.05*i)) for i in range(6)]
                nbrxs = [int(nbrxs_last_month * (0.85 + 0.04*i)) for i in range(6)]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates, y=trxs, mode="lines+markers", name="TRx"))
                fig.add_trace(go.Scatter(x=dates, y=nbrxs, mode="lines+markers", name="NBRx"))
                fig.update_layout(title="TRx vs NBRx Trend", xaxis_title="Month", yaxis_title="Scripts")
                st.plotly_chart(fig, use_container_width=True)

                reply = "📈 TRx and NBRx trends shown above. Let me know if you'd like strategy recommendations."

            elif intent == "call_trends":
                st.markdown("📞 Simulated Call Activity Trend for this HCP:")
                call_dates = pd.date_range(end=datetime.now(), periods=6, freq="M")
                calls = [3, 4, 2, 5, 6, 4]  # Replace with real data if available

                fig = go.Figure()
                fig.add_trace(go.Bar(x=call_dates, y=calls, name="Calls"))
                fig.update_layout(title="Monthly Call Activity", xaxis_title="Month", yaxis_title="Number of Calls")
                st.plotly_chart(fig, use_container_width=True)

                reply = "📊 This chart shows recent call activity. Want to schedule another call?"

            elif intent == "last_call":
                days = hcp_row["days_since_last_call"]
                reply = f"📅 Last call was **{days} days ago**. Re-engagement is recommended if over 21 days."
            elif intent == "general_question":
                reply = "🤖 I'm here to help! Try asking for strategy suggestions or data trends."

            # Handle prompt-based response
            if intent in ["call_strategy", "sample_strategy", "coaching_strategy"]:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful medical sales AI agent."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                reply = response.choices[0].message.content

        except Exception as e:
            reply = f"❌ Error: {str(e)}"

        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

# --- Footer ---
st.markdown("---")
st.info("This is a multi-agent AI prototype that applies reasoning and action via a unified chat interface.")
