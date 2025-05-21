
#  Veeva AI Insights

**AI-powered decision support tools for Field Operations and Commercial Excellence within Veeva CRM.**

This project showcases a suite of AI/ML modules designed to address real-world operational needs at Abbott Diabetes Care (ADC), such as HCP targeting, prescription forecasting, and Next Best Action recommendations — all tailored to integrate with Veeva CRM in a compliant, scalable manner.

##  Modules Overview

### 1. Smart HCP Targeting
Prioritize HCPs based on urgency scores, prescription behavior, and visit recency. Supports ML-based scoring using XGBoost or Logistic Regression.

### 2. TRx/NBRx Forecasting
Predict future prescription volumes using Prophet (time-series model). Outputs include confidence intervals and trend visualizations per HCP.

### 3. Next Best Action Assistant
Recommends optimal next steps (Call, Sample Drop, Coaching) based on HCP behavior and recent engagement. Uses GPT-4 to generate contextual suggestions.

### 4. Route Optimization
Optimized geo-routing of top 20 HCPs using Google Maps Distance Matrix API and clustering logic. Reduces travel time for sales reps.

##  Tech Stack

- **Frontend**: Streamlit
- **ML Models**: XGBoost, Prophet, Logistic Regression, GPT-4 (via OpenAI API)
- **Geospatial**: Google Maps API, KMeans clustering
- **Data**: Synthetic CMS data (Part D prescribers), customizable CSV inputs

##  Repository Structure

```bash
veeva-ai-insights/
│
├── 1_HCP_Targeting.py          # HCP scoring & prioritization module
├── 2_TRx_NBRx_Forecast.py      # Prescription forecasting module
├── 3_NBA_Assistant.py          # GPT-based Next Best Action recommender
├── 4_Geo_Routing_Advanced.py   # Route optimization with clustering + maps
│
├── data/                       # Sample datasets (CMS, TRx/NBRx, profiles)
├── static/                     # Logos and assets
├── utils/                      # Helper functions, plotters, etc.
└── requirements.txt            # All required Python packages
```

##  Setup Instructions

```bash
# Clone the repo
git clone https://github.com/anirudh-sharma-abbott/veeva-ai-insights.git
cd veeva-ai-insights

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run a module
streamlit run 1_HCP_Targeting.py
```

>  Note: To run the geo-routing module, you’ll need a valid Google Maps API key. Store it in a `.env` file as `GOOGLE_MAPS_API_KEY=your_key_here`.

##  Collaboration

This MVP is part of an internal AI/ML initiative to explore operational enhancements in Veeva CRM. For feedback, improvements, or integration discussions, please reach out.

---

###  Maintainer  
**Anirudh Sharma**  
Business Analyst | AI/ML Enthusiast | Abbott Diabetes Care  
anirudhs9411@gmail.com

