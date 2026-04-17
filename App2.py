import streamlit as st
import pandas as pd
import joblib

# 🔹 Load trained pipeline model
model = joblib.load("fraud_pipeline.pkl")

# 🔹 App title
st.title("💳 Invoice Fraud Detection")

st.write("Enter invoice details to check fraud risk")

# -----------------------------
# 🔹 INPUT FIELDS
# -----------------------------

invoice_amount = st.number_input("Invoice Amount", min_value=0.0)
invoice_amount_zscore = st.number_input("Invoice Amount Z-Score")

duplicate_invoice_flag = st.selectbox("Duplicate Invoice", [0, 1])
split_invoice_flag = st.selectbox("Split Invoice", [0, 1])
late_night_submission_flag = st.selectbox("Late Night Submission", [0, 1])

supplier_invoice_count_30d = st.number_input("Supplier Invoice Count (30 days)", min_value=0)
supplier_avg_amount_90d = st.number_input("Supplier Avg Amount (90 days)", min_value=0.0)

submission_hour = st.number_input("Submission Hour (0–23)", min_value=0, max_value=23)

# 🔥 DATE INPUT (VERY IMPORTANT)
date = st.date_input("Invoice Date")

# Extract features
year = date.year
month = date.month
day = date.day

# -----------------------------
# 🔹 PREDICTION
# -----------------------------

if st.button("Predict Fraud"):

    # Create input dataframe
    input_df = pd.DataFrame([{
        'invoice_amount': invoice_amount,
        'invoice_amount_zscore': invoice_amount_zscore,
        'duplicate_invoice_flag': duplicate_invoice_flag,
        'split_invoice_flag': split_invoice_flag,
        'late_night_submission_flag': late_night_submission_flag,
        'supplier_invoice_count_30d': supplier_invoice_count_30d,
        'supplier_avg_amount_90d': supplier_avg_amount_90d,
        'submission_hour': submission_hour,

        # 🔥 Required for pipeline
        'year': year,
        'month': month,
        'day': day
    }])

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Output
    if prediction == 1:
        st.error(f"⚠️ Fraud Detected! (Risk Score: {probability:.2f})")
    else:
        st.success(f"✅ Not Fraud (Confidence: {1 - probability:.2f})")