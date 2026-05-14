import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Cargo Rejection Prediction",
    layout="wide"
)

# =========================
# Load Model
# =========================

model = joblib.load("cargo_rejection_catboost_model.pkl")

# =========================
# Custom Styling
# =========================

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }

    .title {
        font-size: 40px;
        font-weight: bold;
        color: #003366;
        text-align: center;
    }

    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #555555;
        margin-bottom: 30px;
    }

    .stButton>button {
        background-color: #003366;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        height: 50px;
        width: 100%;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Header
# =========================

st.markdown(
    '<p class="title">Cargo Acceptance Rejection Forecasting</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Predict whether cargo shipment will be Accepted or Rejected using Machine Learning</p>',
    unsafe_allow_html=True
)

# =========================
# Layout Columns
# =========================

col1, col2 = st.columns(2)

# =========================
# Left Column Inputs
# =========================

with col1:

    st.subheader("Shipment Information")
    
    # Shipment Date
    shipment_date = st.date_input(
    "Shipment Date"
    )

    airline = st.selectbox(
        "Airline",
        ['Emirates', 'Qatar', 'Lufthansa', 'Air India']
    )

    SHC = st.selectbox(
        "SHC",
        ['DGR', 'PER', 'AVI', 'VAL', 'PIL', 'COL', 'HUM']
    )

    origin = st.selectbox(
        "Origin",
        ['HAM', 'DXB', 'MAA', 'OSL', 'AMS', 'JFK', 'MXP', 'FRA', 'PVG']
    )

    destination = st.selectbox(
        "Destination",
        ['JED']
    )

    packaging_condition = st.selectbox(
        "Packaging Condition",
        ['Good', 'Average', 'Damaged']
    )

    packaging_seal_status = st.selectbox(
        "Packaging Seal Status",
        ['Intact', 'Tampered', 'Broken', 'Missing']
    )

    shipper_company_name = st.selectbox(
        "Shipper Company",
        ['DB Schenker', 'FedEx', 'DHL', 'Maersk', 'Aramex', 'BlueDart', 'UPS', 'Amazon Logistics']
    )

    consignee_name = st.selectbox(
        "Consignee Name",
        ['Gulf Fresh Foods', 'Al Jazeera Trading', 'Jeddah Electronics',
        'Arabian Medical Supply', 'Middle East Retail Group', 'Desert Logistics',
        'Red Sea Imports', 'Saudi Pharma Ltd']
    )

# =========================
# Right Column Inputs
# =========================

with col2:

    st.subheader("Compliance & Security")

    shipper_type = st.selectbox(
        "Shipper Type",
        ['Corporate', 'Retail', 'Agent']
    )

    documentation_status = st.selectbox(
        "Documentation Status",
        ['Complete', 'Pending', 'Missing']
    )

    security_screening_status = st.selectbox(
        "Security Screening Status",
        ['Cleared', 'Manual Inspection Required', 'Pending', 'Failed']
    )

    xray_scan_result = st.selectbox(
        "X-Ray Scan Result",
        ['Clear', 'Suspicious', 'Manual Review Required', 'Restricted Item Detected']
    )

    shipment_priority = st.selectbox(
        "Shipment Priority",
        ['Normal', 'Express', 'VIP']
    )

    cargo_weight_kg = st.number_input(
        "Cargo Weight (kg)",
        min_value=0.0,
        value=100.0
    )

    damage_history_count = st.number_input(
        "Damage History Count",
        min_value=0,
        max_value=50,
        value=0
    )

    compliance_violation_count = st.number_input(
        "Compliance Violation Count",
        min_value=0,
        max_value=50,
        value=0
    )

    shipper_reliability_score = st.number_input(
        "Shipper Reliability Score",
        min_value=0,
        max_value=100,
        value=80
    )

# =========================
# Prediction Button
# =========================

st.markdown("---")

month = shipment_date.month
day = shipment_date.day
day_of_week = shipment_date.weekday()
is_weekend = 1 if day_of_week in [5, 6] else 0

if st.button("Predict Cargo Status"):

# =========================
# Date Feature Engineering
# =========================
    # Create Input DataFrame
    input_data = pd.DataFrame({
        'airline': [airline],
        'SHC': [SHC],
        'origin': [origin],
        'destination': [destination],
        'packaging_condition': [packaging_condition],
        'packaging_seal_status': [packaging_seal_status],
        'shipper_company_name': [shipper_company_name],
        'consignee_name': [consignee_name],
        'shipper_type': [shipper_type],
        'documentation_status': [documentation_status],
        'security_screening_status': [security_screening_status],
        'xray_scan_result': [xray_scan_result],
        'shipment_priority': [shipment_priority],

        'cargo_weight_kg': [cargo_weight_kg],
        'damage_history_count': [damage_history_count],
        'compliance_violation_count': [compliance_violation_count],
        'shipper_reliability_score': [shipper_reliability_score],
        'month': [month],
        'day': [day],
        'day_of_week': [day_of_week],
        'is_weekend': [is_weekend],
        
    })

    input_data = input_data[[       
        'airline',
        'SHC',
        'origin',
        'destination',
        'packaging_condition',
        'packaging_seal_status',
        'shipper_company_name',
        'consignee_name',
        'shipper_type',
        'documentation_status',
        'security_screening_status',
        'xray_scan_result',
        'shipment_priority',
        'cargo_weight_kg',
        'damage_history_count',
        'compliance_violation_count',
        'shipper_reliability_score',
        'month',
        'day',
        'day_of_week',
        'is_weekend',
    ]]
    # =========================
# Convert categorical columns to string
# =========================

    
    # Convert categorical columns to string
    cat_features = input_data.select_dtypes(include=['object']).columns.tolist()

    for col in cat_features:
        input_data[col] = input_data[col].astype(str)

    # Predict probability
    prediction_prob = model.predict_proba(input_data)[:, 1][0]

    # Threshold
    threshold = 0.65

    st.markdown("---")

    # Prediction Output
    if prediction_prob > threshold:

        st.error("❌ Prediction Result: Rejected Cargo")

    else:

        st.success("✅ Prediction Result: Accepted Cargo")

# =========================
# Feature Contribution Analysis
# =========================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<h1 style='color:#2d2d3a; font-size:42px; font-weight:bold;'>
Feature Contribution Analysis
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<h2 style='color:#2d2d3a; font-size:28px; font-weight:bold; margin-bottom:20px;'>
Feature Influence on Current Prediction
</h2>
""", unsafe_allow_html=True)

# Real Feature Importance Values

features = [
    "packaging_seal_status",
    "xray_scan_result",
    "shipper_reliability_score",
    "compliance_violation_count",
    "documentation_status",
    "damage_history_count",
    "cargo_weight_kg",
    "month",
    "security_screening_status",
    "day_of_week"
]

importance = [
    19.705755,
    16.124521,
    14.701518,
    13.606497,
    10.264653,
    7.643616,
    2.821063,
    2.592779,
    2.525383,
    2.070268
]

# Create Plotly Figure
fig = go.Figure()

fig.add_trace(go.Bar(
    x=importance,
    y=features,
    orientation='h',
    text=[f"{round(x,2)}%" for x in importance],
    textposition='outside',
    marker=dict(
        color='#0b66c3'
    )
))

# Layout Styling
fig.update_layout(
    height=500,

    plot_bgcolor='#f3f3f3',
    paper_bgcolor='#f3f3f3',

    xaxis=dict(
        showgrid=False,
        visible=False
    ),

    yaxis=dict(
        autorange='reversed',
        title='',
        tickfont=dict(
            size=15,
            color='#7d8597'
        )
    ),

    font=dict(
        family="Arial",
        size=15,
        color="#2d2d3a"
    ),

    margin=dict(
        l=260,
        r=100,
        t=20,
        b=40
    )
)

# Display Chart
st.plotly_chart(fig, use_container_width=True)
