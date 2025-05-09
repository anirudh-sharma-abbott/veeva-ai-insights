
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import plotly.express as px

# Page setup
st.set_page_config(page_title="Smart HCP Targeting", layout="wide")
st.title("🎯 Smart HCP Targeting - ML-Powered Prioritization")

# Load your CMS-based dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/San_Francisco_HCP_TRx___Unique_Drugs_as_NBRx.csv")

df = load_data()

# Sidebar: Score interpretation
with st.sidebar.expander("📊 Score Interpretation Guide"):
    st.markdown("""
    - 🟢 **High (0.8 – 1.0)**: High conversion potential
    - 🟡 **Medium (0.5 – 0.8)**: Moderate priority
    - 🔴 **Low (0.0 – 0.5)**: Low urgency
    """)

# Sidebar: Specialty filter
st.sidebar.header("🔍 Filter HCP List")
specialty = st.sidebar.selectbox("Specialty", ["All"] + sorted(df["Prscrbr_Type"].dropna().unique().tolist()))

# Encode specialty
le = LabelEncoder()
df["specialty_encoded"] = le.fit_transform(df["Prscrbr_Type"].astype(str))

# # Simulate binary target for training
# df["converted"] = (df["NBRx_TRx_Ratio"] > 0.05).astype(int)

# Create a smarter binary target using top 25% as "converted"
ratio_threshold = df["NBRx_TRx_Ratio"].quantile(0.75)
df["converted"] = (df["NBRx_TRx_Ratio"] > ratio_threshold).astype(int)

# Prepare features
X = df[["TRx", "NBRx", "NBRx_TRx_Ratio", "specialty_encoded"]]
y = df["converted"]

# Button: Trigger ML model scoring
if st.button("🔁 Generate Conversion Scores"):
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X, y)
    df["conversion_score"] = model.predict_proba(X)[:, 1]
    st.success("✅ Conversion scores successfully generated using XGBoost.")
else:
    df["conversion_score"] = 0
    st.warning("⚠️ Conversion scores not generated yet. Click the button above to compute.")

# Apply specialty filter
if specialty != "All":
    df = df[df["Prscrbr_Type"] == specialty]

# Table: Top HCPs by score
st.subheader("🔝 Top HCPs by ML Conversion Score")
top_df = df.sort_values("conversion_score", ascending=False).head(20)

st.dataframe(
    top_df[[
        "Prscrbr_First_Name", "Prscrbr_Last_Org_Name", "Prscrbr_Type",
        "TRx", "NBRx", "NBRx_TRx_Ratio", "conversion_score"
    ]].style
    .format({"conversion_score": "{:.3f}", "NBRx_TRx_Ratio": "{:.4f}"})
    .bar(subset=["conversion_score"], color="#90ee90"),
    use_container_width=True
)

# # Conversion Score Breakdown
# st.subheader("📊 Conversion Score Breakdown")

# df["score_bucket"] = pd.cut(df["conversion_score"], bins=[0, 0.5, 0.8, 1.0],
#                             labels=["Low", "Medium", "High"])

# score_counts = df["score_bucket"].value_counts().sort_index().reset_index()
# score_counts.columns = ["Score_Bucket", "Count"]

# fig_score = px.bar(
#     score_counts,
#     x="Score_Bucket",
#     y="Count",
#     text="Count",
#     color="Score_Bucket",
#     title="HCPs by Conversion Potential"
# )
# st.plotly_chart(fig_score, use_container_width=True)

# Conversion Score Breakdown with Percentages
st.subheader("📊 Conversion Score Breakdown")

df["score_bucket"] = pd.cut(df["conversion_score"], bins=[0, 0.5, 0.8, 1.0],
                            labels=["Low", "Medium", "High"])

# Count + percentage
score_counts = df["score_bucket"].value_counts().sort_index().reset_index()
score_counts.columns = ["Score_Bucket", "Count"]
score_counts["Percent"] = (score_counts["Count"] / score_counts["Count"].sum() * 100).round(1)
score_counts["Label"] = score_counts["Count"].astype(str) + " (" + score_counts["Percent"].astype(str) + "%)"

# Bar chart
fig_score = px.bar(
    score_counts,
    x="Score_Bucket",
    y="Count",
    text="Label",
    color="Score_Bucket",
    title="HCPs by Conversion Potential"
)
fig_score.update_traces(textposition="outside")

st.plotly_chart(fig_score, use_container_width=True)

# # High-Potential HCPs by behavior
# st.subheader("📌 High Potential HCPs (Low TRx, High NBRx)")

# df_top_targets = df[
#     (df["TRx"] < 200) &
#     (df["NBRx_TRx_Ratio"] > 0.06) &
#     (df["conversion_score"] > 0.8)
# ]

# if df_top_targets.empty:
#     st.info("ℹ️ No high-priority targets match the current filter.")
# else:
#     fig_focus = px.scatter(df_top_targets, x="TRx", y="NBRx",
#                         color="Prscrbr_Type", size="conversion_score",
#                         hover_data=["Prscrbr_First_Name", "Prscrbr_Last_Org_Name"],
#                         title="Top Target HCPs with High New Rx Behavior")
#     st.plotly_chart(fig_focus, use_container_width=True)

# Top Target HCPs with Better Quadrant Highlighting
st.subheader("📌 High Potential HCPs (Low TRx, High NBRx)")

df_top_targets = df[
    (df["TRx"] < 200) &
    (df["NBRx_TRx_Ratio"] > 0.06) &
    (df["conversion_score"] > 0.8)
]

if df_top_targets.empty:
    st.info("ℹ️ No high-priority targets match the current filter.")
else:
    # Define quadrant cutoffs
    trx_cutoff = 100
    nbrx_cutoff = 8

    # Create the scatter plot
    fig_focus = px.scatter(
        df_top_targets, x="TRx", y="NBRx",
        color="Prscrbr_Type", size="conversion_score",
        hover_data=["Prscrbr_First_Name", "Prscrbr_Last_Org_Name"],
        title="Top Target HCPs with High New Rx Behavior"
    )

    # Add light quadrant background rectangles
    fig_focus.add_shape(type="rect", x0=0, x1=trx_cutoff, y0=nbrx_cutoff, y1=14,
        fillcolor="LightGreen", opacity=0.15, layer="below", line_width=0)

    fig_focus.add_shape(type="rect", x0=trx_cutoff, x1=200, y0=nbrx_cutoff, y1=14,
        fillcolor="LightSkyBlue", opacity=0.15, layer="below", line_width=0)

    # Add vertical and horizontal guide lines
    fig_focus.add_shape(type="line", x0=trx_cutoff, x1=trx_cutoff, y0=0, y1=14,
        line=dict(color="gray", width=1, dash="dash"))

    fig_focus.add_shape(type="line", x0=0, x1=200, y0=nbrx_cutoff, y1=nbrx_cutoff,
        line=dict(color="gray", width=1, dash="dash"))

    st.plotly_chart(fig_focus, use_container_width=True)

    # Clear markdown summary above
    st.markdown("""
    #### 🧠 How to Read This Chart:
    - **X-axis**: Total prescription volume (TRx)
    - **Y-axis**: Number of unique drugs prescribed (NBRx)
    - **Bubble size**: ML-predicted conversion score
    - **Color**: HCP specialty

    ##### Visual Quadrants:
    - 🟩 **Top-left (light green)**: Low TRx, High NBRx → _Hidden Gems_ — try outreach.
    - 🟦 **Top-right (light blue)**: High TRx, High NBRx → _Key Accounts_ — protect & grow.
    """)

# Explanation section
with st.expander("ℹ️ How This Works"):
    st.markdown("""
    - This page uses CMS Part D prescriber data to identify high-value HCPs.
    - A machine learning model (XGBoost) is trained live using:
      - TRx (Total Claims)
      - NBRx (Unique Drugs Prescribed)
      - NBRx / TRx Ratio
      - Specialty
    - HCPs are scored from **0 (low)** to **1 (high)** conversion potential.
    - Visuals highlight HCPs who may be receptive to outreach or education.
    """)

# Overfitting disclaimer and future scope
with st.expander("⚠️ Model Limitations & Future Scope"):
    st.markdown("""
    ### ⚠️ Current Limitation: Model Overfitting
    - The current ML model is trained on a **proxy label** derived from NBRx/TRx ratio.
    - Since the same feature is also part of the input, the model may **overfit**, especially with small or homogenous datasets.
    - This results in **inflated scores** (e.g., many HCPs having scores close to 1.0), which may not generalize.

    ### 🚀 Future Scope
    - **Better labeling** using real-world adoption events (e.g., rep call response, sample request, drug adoption).
    - **Additional features** like:
        - Recent rep interactions
        - Geographic region / zip-level targeting
        - Practice size or patient panel
        - Drug category focus
    - **Model robustness**:
        - Cross-validation and AUC reporting
        - Explore logistic regression, LightGBM for interpretability
        - Deploying a live retraining pipeline

    This PoC demonstrates the workflow, but real-world deployment should include **rigorous model evaluation, bias checks, and ground-truth labels**.
    """)