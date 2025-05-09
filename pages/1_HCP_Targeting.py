import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from scipy.special import expit
import plotly.graph_objects as go

# Page setup
st.set_page_config(page_title="Smart HCP Targeting", layout="wide")
st.title("Smart HCP Targeting - ML-Powered Prioritization")
st.markdown(
    "*Data sourced from [CMS](https://data.cms.gov/). Model: XGBoost (Extreme Gradient Boosting) is an ensemble learning algorithm that builds a series of decision trees where each new tree corrects the errors of the previous ones. It uses gradient descent optimization and regularization techniques to improve predictive accuracy while preventing overfitting.*"
)


# Load enriched dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/Updated_HCP_Conversion_Data.csv")

df = load_data()


# Fix the 'converted' column to ensure both classes are present
# threshold = df["NBRx_TRx_Ratio"].median()
# df["converted"] = (df["NBRx_TRx_Ratio"] > threshold).astype(int)
threshold = df["NBRx_TRx_Ratio"].quantile(0.75)
df["converted"] = (df["NBRx_TRx_Ratio"] > threshold).astype(int)


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




# Encode categorical features
le_specialty = LabelEncoder()
df["specialty_encoded"] = le_specialty.fit_transform(df["Prscrbr_Type"].astype(str))

le_zip = LabelEncoder()
df["zip_encoded"] = le_zip.fit_transform(df["Zip_Region"].astype(str))

le_size = LabelEncoder()
df["practice_size_encoded"] = le_size.fit_transform(df["Practice_Size"].astype(str))

# Prepare features for model
X = df[["TRx", "NBRx", "NBRx_TRx_Ratio", "Recent_Rep_Visits", "specialty_encoded", "zip_encoded", "practice_size_encoded"]]
y = df["converted"]

# Button: Trigger ML model scoring
if st.button("Generate Conversion Scores"):
    # Check if both classes (0 and 1) are present in the label
    if y.nunique() < 2:
        st.error("❌ Cannot train model: The 'converted' column must contain both 0 and 1 values.")
        st.write("Label distribution:", y.value_counts())
        st.stop()

    # Proceed with training
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X, y)

    # Clip conversion scores between 0.05 and 0.95
    # raw_scores = model.predict_proba(X)[:, 1]
    # # df["conversion_score"] = np.clip(raw_scores, 0.05, 0.95)
    # # Add controlled noise to XGBoost predictions for realistic spread
    # noise = np.random.normal(loc=0, scale=0.05, size=len(raw_scores))  # mean 0, stddev 0.05
    # noisy_scores = raw_scores + noise
    # df["conversion_score"] = np.clip(noisy_scores, 0.05, 0.95)
    # Predict and apply rank-based transformation for smoother spread
    raw_scores = model.predict_proba(X)[:, 1]

    # Scale to 0–1 before sigmoid (stretches out input for more granularity)
    scaled_raw = MinMaxScaler().fit_transform(raw_scores.reshape(-1, 1)).flatten()

    # Apply sigmoid with slight amplification
    z_scores = (scaled_raw - 0.5) * 6  # widen the sigmoid input range
    sigmoid_scores = expit(z_scores)

    df["conversion_score"] = np.clip(sigmoid_scores, 0.05, 0.95)




    st.success("✅ Conversion scores successfully generated using XGBoost.")




    # --- Feature Importance using Plotly ---
    st.subheader("Feature Importance (XGBoost)")
    importance_vals = model.feature_importances_
    feature_names = X.columns

    fig_imp = go.Figure(go.Bar(
        x=importance_vals,
        y=feature_names,
        orientation='h',
        marker=dict(color='lightblue'),
    ))
    fig_imp.update_layout(
        title="XGBoost Feature Importance (Gain-Based)",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=350
    )
    st.plotly_chart(fig_imp, use_container_width=True)
else:
    df["conversion_score"] = 0
    st.warning("Conversion scores not generated yet. Click the button above to compute.")

# Apply specialty filter
if specialty != "All":
    df = df[df["Prscrbr_Type"] == specialty]

# Table: Top HCPs
st.subheader("🔝 Top HCPs by ML Conversion Score")
top_df = df.sort_values("conversion_score", ascending=False).head(20)
st.dataframe(
    top_df[[
        "Prscrbr_First_Name", "Prscrbr_Last_Org_Name", "Prscrbr_Type",
        "TRx", "NBRx", "NBRx_TRx_Ratio", "Recent_Rep_Visits",
        "Zip_Region", "Practice_Size", "conversion_score"
    ]].style
    .format({"conversion_score": "{:.3f}", "NBRx_TRx_Ratio": "{:.4f}"})
    .bar(subset=["conversion_score"], color="#90ee90"),
    use_container_width=True
)

# Conversion score breakdown
st.subheader("Conversion Score Breakdown")
df["score_bucket"] = pd.cut(df["conversion_score"], bins=[0, 0.5, 0.8, 1.0],
                            labels=["Low", "Medium", "High"])
score_counts = df["score_bucket"].value_counts().sort_index().reset_index()
score_counts.columns = ["Score_Bucket", "Count"]
score_counts["Percent"] = (score_counts["Count"] / score_counts["Count"].sum() * 100).round(1)
score_counts["Label"] = score_counts["Count"].astype(str) + " (" + score_counts["Percent"].astype(str) + "%)"

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

# Score Distribution
st.subheader("Conversion Score Distribution")
if df["conversion_score"].max() > 0:
    fig_dist = px.histogram(
        df, x="conversion_score", nbins=30,
        marginal="violin",
        title="Distribution of ML-Predicted Conversion Scores",
        opacity=0.7, color_discrete_sequence=["#636EFA"]
    )
    fig_dist.update_layout(xaxis_title="Conversion Score", yaxis_title="Number of HCPs")
    st.plotly_chart(fig_dist, use_container_width=True)

# High potential targets
st.subheader("High Potential HCPs (Low TRx, High NBRx)")
df_top_targets = df[
    (df["TRx"] < 200) &
    (df["NBRx_TRx_Ratio"] > 0.06) &
    (df["conversion_score"] > 0.8)
]
if df_top_targets.empty:
    st.info("No high-priority targets match the current filter.")
else:
    trx_cutoff, nbrx_cutoff = 100, 8
    fig_focus = px.scatter(
        df_top_targets, x="TRx", y="NBRx",
        color="Prscrbr_Type", size="conversion_score",
        hover_data=["Prscrbr_First_Name", "Prscrbr_Last_Org_Name"],
        title="Top Target HCPs with High New Rx Behavior"
    )
    fig_focus.add_shape(type="rect", x0=0, x1=trx_cutoff, y0=nbrx_cutoff, y1=14,
                        fillcolor="LightGreen", opacity=0.15, layer="below", line_width=0)
    fig_focus.add_shape(type="rect", x0=trx_cutoff, x1=200, y0=nbrx_cutoff, y1=14,
                        fillcolor="LightSkyBlue", opacity=0.15, layer="below", line_width=0)
    fig_focus.add_shape(type="line", x0=trx_cutoff, x1=trx_cutoff, y0=0, y1=14,
                        line=dict(color="gray", width=1, dash="dash"))
    fig_focus.add_shape(type="line", x0=0, x1=200, y0=nbrx_cutoff, y1=nbrx_cutoff,
                        line=dict(color="gray", width=1, dash="dash"))
    st.plotly_chart(fig_focus, use_container_width=True)
    st.markdown("""
    #### How to Read This Chart:
    - **X-axis**: Total prescription volume (TRx)  
    - **Y-axis**: Number of unique drugs prescribed (NBRx)  
    - **Bubble size**: ML-predicted conversion score  
    - **Color**: HCP specialty  
    """)

# Explanation and model notes
with st.expander("How This Works"):
    st.markdown("""
    - This dashboard uses enriched CMS Part D prescriber data.  
    - The model uses: TRx, NBRx, NBRx/TRx, rep visits, specialty, zip region, practice size  
    - XGBoost trains live and scores HCPs with 0–1 conversion potential  
    - Visuals highlight who to engage or retain
    """)

with st.expander("Model Limitations & Future Scope"):
    st.markdown("""
    - Labels are still a proxy; not actual adoption
    - Explore A/B test feedback or call outcomes as future labels
    - Consider SHAP values for interpretability
    """)
