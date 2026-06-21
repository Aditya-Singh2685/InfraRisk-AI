import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os 
import shap 
import requests

import plotly.express as px 
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rf_model = joblib.load(
    os.path.join(BASE_DIR, "models", "rf_model.pkl")
)

xgb_model = joblib.load(
    os.path.join(BASE_DIR, "models", "xgb_model.pkl")
)

lgb_model = joblib.load(
    os.path.join(BASE_DIR, "models", "lgb_model.pkl")
)

rf_model = joblib.load("models/rf_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")
lgb_model = joblib.load("models/lgb_model.pkl")


st.set_page_config(
    page_title="InfraRisk AI Dashboard",
    layout="wide"
)

st.title("InfraRisk AI Dashboard")

st.subheader(
    "AI-Powered Infrastructure Credit Risk & Financial Analytics"
)

st.markdown("Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Projects", "500")

with col2:
    st.metric("Avg Credit Risk", "42%")

with col3:
    st.metric("Loan Exposure", "$25M")

with col4:
    st.metric("Default Probability", "18%")
    
data = pd.DataFrame({
    "Project": ["Highway", "Metro", "Bridge", "Solar Plant"],
    "Credit_Risk": [25, 60, 45, 30],
    "Loan_Exposure": [10, 25, 15, 20],
    "Default_Probability": [5, 20, 12, 8]
})

st.subheader("Infrastructure Dataset")
st.dataframe(data, use_container_width=True)

st.subheader("Infrastructure Credit Risk")

fig = px.bar(
    data,
    x="Project",
    y="Credit_Risk",
    color="Credit_Risk",
    text="Credit_Risk"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.subheader("Predict Infrastructure Risk")

st.markdown("---")

st.subheader("Predict Infrastructure Risk")

dscr = st.slider("DSCR", 0.5, 5.0, 2.0)

llcr = st.slider("LLCR", 0.5, 5.0, 2.5)

plcr = st.slider("PLCR", 0.5, 5.0, 2.5)

leverage = st.slider("Leverage", 1.0, 10.0, 5.0)

gdp_growth = st.slider("GDP Growth (%)", 0.0, 15.0, 6.0)

inflation = st.slider("Inflation (%)", 0.0, 15.0, 5.0)

construction_progress = st.slider(
    "Construction Progress (%)",
    0,
    100,
    70
)

contractor_score = st.slider(
    "Contractor Score",
    0,
    100,
    80
)

country_rating = st.slider(
    "Country Rating",
    0,
    100,
    75
)

input_data = pd.DataFrame({
    "DSCR": [dscr],
    "LLCR": [llcr],
    "PLCR": [plcr],
    "Leverage": [leverage],
    "GDP_Growth": [gdp_growth],
    "Inflation": [inflation],
    "Construction_Progress": [construction_progress],
    "Contractor_Score": [contractor_score],
    "Country_Rating": [country_rating]
})

rf_pred = rf_model.predict_proba(input_data)[0][1]

xgb_pred = xgb_model.predict_proba(input_data)[0][1]

lgb_pred = lgb_model.predict_proba(input_data)[0][1]

prediction = (
    rf_pred +
    xgb_pred +
    lgb_pred
) / 3

prediction = prediction * 100

st.subheader("Predicted Credit Risk Score")

st.metric(
    "Default Probability",
    f"{prediction:.2f}%"
)

st.progress(int(prediction))

if prediction < 35:
    st.success("Low Infrastructure Risk")

elif prediction < 70:
    st.warning("Medium Infrastructure Risk")

else:
    st.error("High Infrastructure Risk")
    
st.markdown("AI Risk Insights")

if prediction > 70:
    st.error(
        "AI detected severe financial stress. High probability of infrastructure default."
    )

elif prediction > 40:
    st.warning(
        "Moderate infrastructure risk detected. Monitor leverage and repayment capability."
    )

else:
    st.success(
        "Project financial health is stable with low probability of default."
    )
    
explainer = shap.TreeExplainer(xgb_model)

shap_values = explainer.shap_values(input_data)

shap_importance = pd.DataFrame({
    "Feature": input_data.columns,
    "Importance": np.abs(shap_values).mean(axis=0)
})

shap_importance = shap_importance.sort_values(
    by="Importance",
    ascending=True
)

st.markdown("---")
st.subheader("Scenario Simulator")

scenario = st.selectbox(
    "Select Risk Scenario",
    [
        "Normal Economy",
        "Pandemic",
        "High Inflation",
        "Sovereign Downgrade",
        "Construction Delay",
        "Supply Chain Crisis",
        "Interest Rate Shock",
        "Currency Crash",
        "Political Instability",
        "Energy Crisis"
    ]
)

risk_multiplier = {
    "Normal Economy": 1.00,
    "Pandemic": 1.50,
    "High Inflation": 1.30,
    "Sovereign Downgrade": 1.70,
    "Construction Delay": 1.40,
    "Supply Chain Crisis": 1.35,
    "Interest Rate Shock": 1.45,
    "Currency Crash": 1.60,
    "Political Instability": 1.55,
    "Energy Crisis": 1.25
}

adjusted_risk = prediction * risk_multiplier[scenario]

st.metric(
    "Scenario Adjusted Risk",
    f"{adjusted_risk:.2f}%"
)

st.progress(min(int(adjusted_risk), 100))

st.markdown("---")
st.subheader("Project Locations")

map_data = pd.DataFrame({
    "lat": [28.6139, 19.0760, 13.0827],
    "lon": [77.2090, 72.8777, 80.2707],
    "Project": [
        "Delhi Highway",
        "Mumbai Metro",
        "Chennai Solar Plant"
    ]
})

st.map(map_data)

st.markdown("---")
st.header("Satellite Project Viewer")

import streamlit as st

project = st.selectbox(
    "Select Project",
    ["Mumbai Metro", "Delhi Highway", "Solar Park"]
)

prompt = f"Satellite view of {project}, infrastructure project, realistic aerial imagery"

image_url = f"https://image.pollinations.ai/prompt/{prompt}"

st.image(
    image_url,
    caption=f"AI Generated Satellite View - {project}",
    use_container_width=True
)

st.markdown("---")

st.subheader("Model Ensemble Confidence")

ensemble_df = pd.DataFrame({
    "Model": [
        "RandomForest",
        "XGBoost",
        "LightGBM"
    ],
    "Default Probability": [
    round(rf_pred * 100, 2),
    round(xgb_pred * 100, 2),
    round(lgb_pred * 100, 2)
]
})

fig_models = px.bar(
    ensemble_df,
    x="Model",
    y="Default Probability",
    text="Default Probability",
    color="Default Probability"
)

st.plotly_chart(
    fig_models,
    use_container_width=True
)

fig_models.update_traces(
    texttemplate='%{text:.2f}%',
    textposition='outside'
)

st.markdown("---")

st.subheader("Financial Risk Fusion")

health_score = (
    contractor_score * 0.4 +
    country_rating * 0.3 +
    construction_progress * 0.3
)

health_score = round(health_score, 2)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Infrastructure Health Score",
        health_score
    )

with col2:
    st.metric(
        "Default Probability",
        f"{prediction:.2f}%"
    )

st.markdown("---")
st.header("DSCR Long-Term Projection")

years = [2026, 2027, 2028, 2029, 2030]

dscr_forecast = [
    round(dscr, 2),
    round(dscr * 1.10, 2),
    round(dscr * 1.20, 2),
    round(dscr * 1.25, 2),
    round(dscr * 1.35, 2)
]

fig_dscr = go.Figure()

fig_dscr.add_trace(
    go.Scatter(
        x=years,
        y=dscr_forecast,
        mode="lines+markers",
        name="DSCR Forecast"
    )
)

fig_dscr.update_layout(
    title="Debt Service Coverage Ratio Forecast",
    xaxis_title="Year",
    yaxis_title="DSCR",
    xaxis=dict(
        tickmode="array",
        tickvals=years,
        ticktext=[str(y) for y in years]
    )
)

st.plotly_chart(fig_dscr, use_container_width=True)

latest_dscr = dscr_forecast[-1]

if latest_dscr > 2:
    st.success("Projected DSCR remains healthy over the next 5 years.")

elif latest_dscr > 1.2:
    st.warning("Projected DSCR is moderate. Monitor debt servicing ability.")

else:
    st.error("Projected DSCR indicates potential financial stress.")

@st.cache_data
def get_world_bank_data(indicator):
    url = f"https://api.worldbank.org/v2/country/IND/indicator/{indicator}?format=json"

    try:
        response = requests.get(url)
        data = response.json()[1][:5]

        years = []
        values = []

        for item in reversed(data):
            years.append(int(item["date"]))
            values.append(item["value"])

        return years, values

    except:
        return [2021, 2022, 2023, 2024, 2025], [0, 0, 0, 0, 0]
    
st.markdown("---")
st.header("Macroeconomic Indicator Panel")

years_macro = [2021, 2022, 2023, 2024, 2025]

years, gdp_growth = get_world_bank_data(
    "NY.GDP.MKTP.KD.ZG"
)

_, inflation = get_world_bank_data(
    "FP.CPI.TOTL.ZG"
)

col1, col2 = st.columns(2)

with col1:

    fig_gdp = go.Figure()

    fig_gdp.add_trace(
        go.Scatter(
            x=years_macro,
            y=gdp_growth,
            mode="lines+markers",
            name="GDP Growth"
        )
    )

    fig_gdp.update_layout(
    title="GDP Growth Trend (%)",
    xaxis=dict(
        tickmode="array",
        tickvals=years_macro,
        ticktext=[str(y) for y in years_macro]
    )
)

    st.plotly_chart(fig_gdp, use_container_width=True)

with col2:

    fig_inf = go.Figure()

    fig_inf.add_trace(
        go.Scatter(
            x=years_macro,
            y=inflation,
            mode="lines+markers",
            name="Inflation"
        )
    )

    fig_inf.update_layout(
    title="Inflation Trend (%)",
    xaxis=dict(
        tickmode="array",
        tickvals=years_macro,
        ticktext=[str(y) for y in years_macro]
    )
)
    st.plotly_chart(fig_inf, use_container_width=True)
    
st.success(
    "Live Data Source: World Bank API"
)
    
    
st.header("ESG Risk Assessment")

env_score = st.slider(
    "Environmental Score",
    0,100,70
)

soc_score = st.slider(
    "Social Score",
    0,100,75
)

gov_score = st.slider(
    "Governance Score",
    0,100,80
)

esg_score = round(
    (env_score + soc_score + gov_score)/3,
    2
)

st.metric(
    "ESG Risk Score",
    esg_score
)
    
    
st.markdown("---")
st.header("Portfolio Risk Heatmap")

heatmap_data = pd.DataFrame({
    "Project": [
        "Metro",
        "Highway",
        "Airport",
        "Solar",
        "Bridge",
        "Port",
        "Rail"
    ],
    "Risk Score": [
        75,
        65,
        55,
        30,
        45,
        70,
        50
    ]
})

fig_heatmap = px.imshow(
    [heatmap_data["Risk Score"]],
    labels=dict(x="Project", color="Risk"),
    x=heatmap_data["Project"],
    y=["Portfolio"],
    text_auto=True,
    aspect="auto"
)

st.plotly_chart(
    fig_heatmap,
    use_container_width=True
)

st.header("Climate Risk Monitor")

region = st.selectbox(
    "Select Project Region",
    ["Mumbai", "Delhi", "Chennai", "Kolkata"]
)

climate_risk = {
    "Mumbai": 78,
    "Delhi": 52,
    "Chennai": 85,
    "Kolkata": 70
}

risk_score = climate_risk[region]

st.metric("Climate Risk Score", f"{risk_score}/100")

if risk_score > 75:
    st.error("High climate exposure detected.")
elif risk_score > 50:
    st.warning("Moderate climate exposure.")
else:
    st.success("Low climate exposure.")
    
st.subheader("Climate Risk Trend")

climate_years = [2021, 2022, 2023, 2024, 2025]

climate_data = {
    "Mumbai": [65, 68, 72, 75, 78],
    "Delhi": [45, 47, 50, 51, 52],
    "Chennai": [70, 75, 78, 82, 85],
    "Kolkata": [60, 63, 66, 68, 70]
}

trend_df = pd.DataFrame({
    "Year": climate_years,
    "Risk": climate_data[region]
})

fig_climate = px.line(
    trend_df,
    x="Year",
    y="Risk",
    markers=True,
    title=f"{region} Climate Risk Trend"
)

st.plotly_chart(fig_climate, use_container_width=True)

st.header("Interest Rate Monitor")

interest_rate = 6.5

st.metric(
    "Current Interest Rate",
    f"{interest_rate}%"
)

st.subheader("Flood Risk Index")

flood_risk = {
    "Mumbai": 85,
    "Delhi": 35,
    "Chennai": 78,
    "Kolkata": 82
}

score = flood_risk[region]

st.metric("Flood Risk Score", f"{score}/100")

if score > 75:
    st.error("High flood vulnerability detected.")
elif score > 50:
    st.warning("Moderate flood vulnerability.")
else:
    st.success("Low flood vulnerability.")
    
st.subheader("Heatwave Exposure")

heatwave = {
    "Mumbai": 65,
    "Delhi": 88,
    "Chennai": 72,
    "Kolkata": 60
}

heat_score = heatwave[region]

st.metric("Heatwave Exposure", f"{heat_score}/100")

st.subheader("Carbon Risk Assessment")

carbon = {
    "Mumbai": 62,
    "Delhi": 74,
    "Chennai": 55,
    "Kolkata": 68
}

carbon_score = carbon[region]

st.metric("Carbon Risk Score", f"{carbon_score}/100")
 
st.markdown("---")
st.header(" Infrastructure Dependency Network")

G = nx.Graph()

edges = [
    ("Metro", "Highway"),
    ("Metro", "Airport"),
    ("Airport", "Port"),
    ("Solar", "Rail")
]

G.add_edges_from(edges)

pos = nx.spring_layout(
    G,
    seed=42,
    k=1.5,
    iterations=100
)

edge_x = []
edge_y = []

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]

    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    line=dict(width=2),
    hoverinfo='none',
    mode='lines'
)

node_x = []
node_y = []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

node_colors = [
    75, 65, 55, 30, 45, 70
]

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode='markers+text',
    text=list(G.nodes()),
    textposition="top center",
    hoverinfo='text',
    marker=dict(
        size=40,
        color=node_colors,
        colorscale="Reds",
        showscale=True,
        colorbar=dict(title="Risk")
),
textfont=dict(size=16)
)

fig_network = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        title="Infrastructure Risk Dependency Network",
        showlegend=False,
        hovermode='closest',
        height=600,
        template="plotly_dark"
    )
)

st.plotly_chart(fig_network, use_container_width=True)

st.info(
    "GNN-inspired dependency graph showing interconnected infrastructure assets."
)


st.header("Monte Carlo Risk Simulation")

simulated_risk = np.clip(
    np.random.normal(
    prediction,
    10,
    1000
),
0,
100
)

fig_mc = px.histogram(
    simulated_risk,
    nbins=30,
    title="Monte Carlo Risk Distribution"
)

st.plotly_chart(
    fig_mc,
    use_container_width=True
)
 
st.markdown("---")
st.header("Real SHAP Explainability")

explainer = shap.TreeExplainer(xgb_model)

shap_values = explainer.shap_values(input_data)

shap_importance = pd.DataFrame({
    "Feature": input_data.columns,
    "Importance": np.abs(shap_values).mean(axis=0)
})

shap_importance = shap_importance.sort_values(
    by="Importance",
    ascending=True
)

fig_shap = px.bar(
    shap_importance,
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    title="SHAP Feature Contribution"
)

st.plotly_chart(fig_shap, use_container_width=True)

top_feature = shap_importance.iloc[-1]["Feature"]

if prediction < 5:
    st.success(
        f"AI Assessment: Project risk is LOW ({prediction:.2f}%). "
        f"Primary driver: {top_feature}."
    )

elif prediction < 15:
    st.warning(
        f"AI Assessment: Project risk is MODERATE ({prediction:.2f}%). "
        f"Primary driver: {top_feature}."
    )

else:
    st.error(
        f"AI Assessment: Project risk is HIGH ({prediction:.2f}%). "
        f"Primary driver: {top_feature}."
    )
    
st.subheader("Investment Recommendation")

if prediction < 5:
    st.success("Recommendation: INVEST")
    st.write("Project shows strong financial stability and low default risk.")

elif prediction < 15:
    st.warning("Recommendation: WATCHLIST")
    st.write("Project risk is manageable but requires monitoring.")

else:
    st.error("Recommendation: HIGH RISK")
    st.write("Project exhibits elevated default probability.")   

# AI Explanation

top_feature = shap_importance.iloc[-1]["Feature"]

st.success(
    f"""
    AI Explanation:

    The model identified **{top_feature}** as the strongest contributor
    to infrastructure credit risk.

    Higher values of this feature have the largest impact on the
    predicted default probability.

    Project currently shows a low-risk profile with strong financial stability.
    """
)
st.header("Executive Summary")

st.info(
    f"""
    Default Probability: {prediction:.2f}%

    ESG Score: {esg_score:.1f}

    Climate Risk: {risk_score}/100

    Primary Risk Driver: {top_feature}

    Recommendation:
    {"INVEST" if prediction < 5 else "WATCHLIST" if prediction < 15 else "HIGH RISK"}
    """
)

st.markdown("---")
st.header("Portfolio Ranking Engine")

portfolio_df = pd.DataFrame({
    "Project": [
        "Metro",
        "Highway",
        "Solar Park",
        "Airport",
        "Port"
    ],
    "Risk": [
        2.4,
        8.5,
        1.2,
        5.7,
        3.5
    ],
    "ESG": [
        75,
        60,
        82,
        70,
        78
    ],
    "Climate": [
        78,
        70,
        45,
        65,
        55
    ]
})

portfolio_df["Investment Score"] = (
    portfolio_df["ESG"] * 0.4
    + (100 - portfolio_df["Climate"]) * 0.3
    + (100 - portfolio_df["Risk"] * 10) * 0.3
)

portfolio_df = portfolio_df.sort_values(
    by="Investment Score",
    ascending=False
).reset_index(drop=True)

portfolio_df.index += 1

st.dataframe(
    portfolio_df,
    use_container_width=True
)

best_project = portfolio_df.iloc[0]["Project"]
best_score = portfolio_df.iloc[0]["Investment Score"]

st.success(
    f"Best Investment Opportunity: "
    f"{best_project} "
    f"(Score: {best_score:.1f})"
)


st.subheader("Top Ranked Assets")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Rank 1",
        portfolio_df.iloc[0]["Project"]
    )

with col2:
    st.metric(
        "Rank 2",
        portfolio_df.iloc[1]["Project"]
    )

with col3:
    st.metric(
        "Rank 3",
        portfolio_df.iloc[2]["Project"]
    )

st.info(
    "Portfolio ranking combines ESG quality, "
    "climate exposure and default risk into a unified score."
)

top_feature = shap_importance.iloc[-1]["Feature"]

if prediction < 5:
    recommendation = "INVEST"
elif prediction < 15:
    recommendation = "WATCHLIST"
else:
    recommendation = "HIGH RISK"

climate_score = climate_risk

st.header("Risk Report Generator")

if st.button("Generate Risk Report"):

    pdf_file = "risk_report.pdf"

    c = canvas.Canvas(pdf_file)

    c.line(80, 790, 520, 790)
    c.setFont("Helvetica-Bold", 28)
    c.drawString(150, 800, "INFRARISK AI REPORT")

    c.setFont("Helvetica", 14)

    c.drawString(80, 750, f"Default Probability: {prediction:.2f}%")
    c.drawString(80, 720, f"ESG Score: {esg_score}")
    c.drawString(80, 690, f"Climate Risk Score: {climate_score[region]}/100")
    c.drawString(80, 660, f"Primary Risk Driver: {top_feature}")
    c.drawString(80, 630, f"Investment Recommendation: {recommendation}")
    c.drawString(
    100,
    470,
    f"Generated On: {datetime.now().strftime('%d-%m-%Y')}"
)
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(80, 580, "Executive Summary")
  
    c.setFont("Helvetica", 12)
    c.drawString(
      100,
      550,
      "AI-driven infrastructure credit risk assessment completed."
  )
  
    c.drawString(
      100,
      525,
      "ESG, Climate Risk, Monte Carlo and SHAP analysis included."
  )
  
    c.drawString(
      100,
      500,
      "Generated by InfraRisk AI Platform."
  )
    
    if recommendation == "INVEST":
        c.setFillColor(colors.green)
    elif recommendation == "WATCHLIST":
        c.setFillColor(colors.orange)
    else:
        c.setFillColor(colors.red)

        c.rect(80, 620, 250, 25, fill=1)
        c.setFillColor(colors.white)
        c.drawString(90, 628, recommendation)
    
    
    c.save()

    with open(pdf_file, "rb") as file:
        st.download_button(
            "Download Report",
            file,
            file_name="InfraRisk_Report.pdf"
        )