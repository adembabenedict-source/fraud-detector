import streamlit as st
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Fraud Detector", layout="wide")
st.title("Credit Card Fraud Detector - Live Demo")
st.write("Enter transaction details to see fraud risk update instantly")

# Sidebar inputs
st.sidebar.header("Transaction Details")
amount = st.sidebar.slider("Transaction Amount ($)", 0, 5000, 150)
time = st.sidebar.slider("Time since last transaction (hrs)", 0, 48, 5)
location = st.sidebar.selectbox("Location", ["Local", "Different City", "Foreign Country"])
device = st.sidebar.selectbox("Device Used", ["Known Device", "New Device"])
merchant = st.sidebar.selectbox("Merchant Category", ["Grocery", "Electronics", "Travel", "Online Gaming"])

# Simple fraud scoring logic for demo
risk_score = 0
risk_score += amount * 0.01
risk_score += (48 - time) * 0.5  # Recent transactions = higher risk
if location == "Foreign Country": risk_score += 25
if location == "Different City": risk_score += 10
if device == "New Device": risk_score += 20
if merchant in ["Travel", "Online Gaming"]: risk_score += 15

risk_score = min(100, max(0, risk_score))
fraud_prob = risk_score / 100
legit_prob = 1 - fraud_prob

# Main dashboard
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Prediction", "Fraud" if risk_score > 50 else "Legitimate")
with col2:
    st.metric("Fraud Risk", f"{risk_score:.0f}%")
with col3:
    if risk_score > 75:
        st.error("Critical Risk")
    elif risk_score > 50:
        st.warning("High Risk")
    else:
        st.success("Low Risk")

# Live Pie Chart
st.subheader("Fraud Probability")
fig = go.Figure(data=[go.Pie(
    labels=['Likely Fraud', 'Likely Legitimate'],
    values=[fraud_prob, legit_prob],
    hole=0.4,
    marker_colors=['#FF4B4B', '#00CC96']
)])
fig.update_layout(height=400, showlegend=True)
st.plotly_chart(fig, use_container_width=True)

# Risk breakdown bar chart - FIXED BRACKETS
st.subheader("Risk Factor Breakdown")
factors = pd.DataFrame({
    'Factor': ['Amount', 'Time', 'Location', 'Device', 'Merchant'],
    'Risk Contribution': [
        amount * 0.01,
        (48 - time) * 0.5,
        25 if location == "Foreign Country" else 10 if location == "Different City" else 0,
        20 if device == "New Device" else 0,
        15 if merchant in ["Travel", "Online Gaming"] else 0
    ]
})
st.bar_chart(factors.set_index('Factor'))