import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime

# --- Load API key ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OpenAI API key not found.")
    st.stop()

client = OpenAI(api_key=api_key)

# --- Page setup ---
st.set_page_config(page_title="NBA Assistant", layout="wide")
st.title("🤖 Next Best Action (NBA) Assistant - Multi-Agent AI")

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/hcp_profiles.csv")
    df["last_call_date"] = pd.to_datetime(df["last_call_date"], errors="coerce")
    df["days_since_last_call"] = (datetime.now() - df["last_call_date"]).dt.days
    return df

df = load_data()
selected_hcp = st.selectbox("Select HCP", df["name"])
hcp_row = df[df["name"] == selected_hcp].iloc[0]

# --- KPI Metrics ---
# st.markdown("### 👤 HCP Snapshot")
# col1, col2, col3 = st.columns(3)
# conversion_score = hcp_row["conversion_score"]


# col1.metric("🧠 Conversion Score", f"{conversion_score:.2f}",
#             delta="High" if conversion_score > 0.6 else ("Medium" if conversion_score > 0.4 else "Low"))

#########
# --- Pull value from selected row ---
conversion_score = hcp_row["conversion_score"]
trxs_last_month = int(hcp_row["trxs_last_month"])
nbrxs_last_month = int(hcp_row["nbrxs_last_month"])

# --- Custom HTML badge for conversion score ---
def get_score_badge(score):
    if score < 0.4:
        color = "#e74c3c"  # Red
        label = "Low"
    elif score < 0.6:
        color = "#f1c40f"  # Yellow
        label = "Medium"
    else:
        color = "#2ecc71"  # Green
        label = "High"
    
    return f"""
    <span style="background-color:{color}; 
                 color:white; 
                 padding:4px 10px; 
                 border-radius:8px; 
                 font-weight:bold;
                 font-size:0.85rem;">
        {label} Conversion Potential
    </span>
    """

# --- Display KPI Metrics ---
st.markdown("### 👤 HCP Snapshot")
col1, col2, col3 = st.columns(3)

col1.metric("🧠 Conversion Score", f"{conversion_score:.2f}")
col1.markdown(get_score_badge(conversion_score), unsafe_allow_html=True)

col2.metric("💊 TRx Last Month", trxs_last_month)
col3.metric("📈 NBRx Last Month", nbrxs_last_month)

# --- Additional profile summary ---
st.markdown(f"""
- **Segment**: {hcp_row['segment']}
- **Specialty**: {hcp_row['specialty']}
- **Days Since Last Call**: {hcp_row['days_since_last_call']}
- **Last Call Date**: {hcp_row['last_call_date'].date() if pd.notnull(hcp_row['last_call_date']) else 'N/A'}
""")


########


# col2.metric("💊 TRx Last Month", int(hcp_row["trxs_last_month"]))
# col3.metric("📈 NBRx Last Month", int(hcp_row["nbrxs_last_month"]))

# --- Additional Profile Info ---
st.markdown(f"""
- **Segment**: {hcp_row['segment']}
- **Specialty**: {hcp_row['specialty']}
- **Days Since Last Call**: {hcp_row['days_since_last_call']}
- **Last Call Date**: {hcp_row['last_call_date'].date() if pd.notnull(hcp_row['last_call_date']) else 'N/A'}
""")

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

# --- Strategy Tabs ---
tab1, tab2, tab3 = st.tabs(["📞 Call Strategy", "📧 Sample Strategy", "📚 Coaching Strategy"])
for tab, action in zip([tab1, tab2, tab3], ["in-person call", "sample drop", "coaching session"]):
    with tab:
        st.subheader(f"{action.title()} Recommendation")
        if st.button(f"🧠 Generate {action.title()} Recommendation", key=action):
            with st.spinner("Thinking..."):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a medical sales strategy assistant."},
                            {"role": "user", "content": generate_prompt(action, hcp_row.to_dict())}
                        ],
                        temperature=0.7,
                        max_tokens=500
                    )
                    suggestion = response.choices[0].message.content
                    st.success(suggestion)

                    # --- Suggested Actions ---
                    st.markdown("### 🚀 Suggested Actions")
                    cols = st.columns(3)
                    if cols[0].button("📞 Schedule Call", key=f"call_{action}"):
                        st.info("Call scheduled in Veeva.")
                    if cols[1].button("📧 Send Email", key=f"email_{action}"):
                        st.info("Email drafted.")
                    if cols[2].button("📦 Assign Sample", key=f"sample_{action}"):
                        st.info("Sample request submitted.")
                except Exception as e:
                    st.error(f"OpenAI Error: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("#### ℹ️ About This Page")
st.info("""
This assistant uses GPT-4 and business logic to generate AI-powered decisions for sales reps. 
It applies thresholds (like conversion score and recent interaction) to guide whether an in-person call, sample drop, or coaching session should be taken this week — and provides rationale for each.
""")



import plotly.graph_objects as go

# --- Chatbot Section ---
st.markdown("---")
st.markdown("### 💬 Ask the Assistant")
st.caption("Chat with your AI rep assistant about this HCP. Try: *'Show me TRx trends'* or *'When was the last call?'*")

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Chat input ---
user_input = st.chat_input("Type your question...")

# --- Display messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Handle new user message ---
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- Simulate AI reply ---
    with st.chat_message("assistant"):
        if "trend" in user_input.lower():
            st.markdown("Here is the TRx vs NBRx trend for this HCP:")

            # Create time series dummy data (replace with real history if available)
            dates = pd.date_range(end=datetime.now(), periods=6, freq="M")
            trxs = [int(trxs_last_month * (0.9 + 0.05*i)) for i in range(6)]
            nbrxs = [int(nbrxs_last_month * (0.85 + 0.04*i)) for i in range(6)]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=trxs, mode="lines+markers", name="TRx"))
            fig.add_trace(go.Scatter(x=dates, y=nbrxs, mode="lines+markers", name="NBRx"))
            fig.update_layout(title="TRx vs NBRx Trend", xaxis_title="Month", yaxis_title="Scripts")
            st.plotly_chart(fig, use_container_width=True)

            ai_reply = "📈 TRx and NBRx have shown a steady trend. Consider a follow-up if last call was over 3 weeks ago."
        
        elif "last call" in user_input.lower():
            days = hcp_row['days_since_last_call']
            ai_reply = f"📅 Last call was **{days} days ago**. Re-engagement is recommended if over 21 days."
        
        else:
            ai_reply = f"🤖 I'm still learning to understand this question. Try asking about trends, segment, or last call."

        st.markdown(ai_reply)
        st.session_state.messages.append({"role": "assistant", "content": ai_reply})
