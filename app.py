import streamlit as st
import pandas as pd
import joblib

# =========================
# Load Model
# =========================

model = joblib.load("cargo_rejection_catboost_model.pkl")

# =========================
# App Title
# =========================

st.set_page_config(page_title="Cargo Rejection Prediction", layout="wide")

st.title("Cargo Acceptance Rejection Forecasting")
st.write("Predict whether a cargo shipment will be Accepted or Rejected.")

# =========================
# User Inputs
# =========================

st.subheader("Shipment Details")

# Categorical Inputs

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
    "Shipper Company Name",
    ['DB Schenker', 'FedEx', 'DHL', 'Maersk', 'Aramex', 'BlueDart', 'UPS', 'Amazon Logistics']
)

consignee_name = st.selectbox(
    "Consignee Name",
    ['Gulf Fresh Foods', 'Al Jazeera Trading', 'Jeddah Electronics',
    'Arabian Medical Supply', 'Middle East Retail Group', 'Desert Logistics',
    'Red Sea Imports', 'Saudi Pharma Ltd']
)

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
    ['Clear', 'Suspicious','Manual Review Required','Restricted Item Detected']
)

shipment_priority = st.selectbox(
    "Shipment Priority",
    ['Normal', 'Express', 'VIP']
)

# Numerical Inputs
cargo_weight_kg = st.number_input(
    "Cargo Weight (kg)",
    min_value=0.0,
    value=0.0
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

if st.button("Predict Cargo Status"):

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
        'shipper_reliability_score': [shipper_reliability_score]
    })

    # Convert categorical columns to string
    cat_features = input_data.select_dtypes(include=['object']).columns.tolist()

    for col in cat_features:
        input_data[col] = input_data[col].astype(str)

    # Predict probability
    prediction_prob = model.predict_proba(input_data)[:, 1][0]

    # Threshold
    threshold = 0.65

    # Prediction
    if prediction_prob > threshold:

        st.error("Prediction: Rejected Cargo")

    else:

        st.success("Prediction: Accepted Cargo")

    # Display probability
    st.subheader("Rejection Probability")

    st.write(f"{prediction_prob:.2f}")