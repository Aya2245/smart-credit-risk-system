import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
os.system("pip install scikit-learn")
# ---------------- PAGE ----------------
st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

# ---------------- FEATURE ENGINEERING ----------------
def feature_engineering(df):
    df = df.copy()

    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]

    df['TOTAL_DELAY'] = 0
    df['UTILIZATION_RATIO'] = df['BILL_AMT1'] / df['LIMIT_BAL'].replace(0, 1)
    df['AVG_BILL'] = df[bill_cols].mean(axis=1)
    df['PAY_RATIO_1'] = df['PAY_AMT1'] / df['BILL_AMT1'].replace(0, 1)

    df['INCOME_UTIL_RATIO'] = df['LIMIT_BAL'] / df['INCOME'].replace(0, 1)
    df['DEFAULT_FLAG'] = df['PREVIOUS_DEFAULT']

    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    return df


# ---------------- LOAD MODEL ----------------
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

data = load_model()

pipeline = data["pipeline"]
feature_order = data["columns"]

# ---------------- UI ----------------
st.title("💳 Credit Risk Predictor")
st.caption("Enhanced Credit Scoring System")
st.divider()

# ---------------- INPUTS ----------------
limit_bal = st.number_input("💰 Credit Limit", value=50000)
age = st.slider("👤 Age", 21, 80, 30)

sex = st.selectbox("⚧ Sex", ["Male", "Female"])
education = st.selectbox("🎓 Education", ["Graduate", "University", "High School"])

payment_behavior = st.selectbox(
    "📅 Payment Behavior",
    ["Always on time", "Mostly on time", "Often delayed", "Rarely pays"]
)

avg_bill = st.number_input("🧾 Average Bill", value=10000)
avg_pay = st.number_input("💵 Average Payment", value=5000)

# 🔥 NEW INPUTS
income = st.number_input("💡 Income", value=30000)
previous_default = st.selectbox("🔥 Previous Default?", ["No", "Yes"])
credit_util = st.slider("📊 Credit Utilization %", 0, 100, 30)

# ---------------- MAPPING ----------------
sex_map = {"Male": 0, "Female": 1}

edu_map = {
    "Graduate": 1,
    "University": 2,
    "High School": 3
}

delay_map = {
    "Always on time": -1,
    "Mostly on time": 0,
    "Often delayed": 2,
    "Rarely pays": 4
}

default_map = {"No": 0, "Yes": 1}

# ---------------- PREDICT ----------------
input_df = None

if st.button("🔍 Analyze Risk", use_container_width=True):

    base_delay = delay_map[payment_behavior]

    input_dict = {
        "LIMIT_BAL": float(limit_bal),
        "AGE": float(age),
        "SEX": float(sex_map[sex]),
        "EDUCATION": float(edu_map[education]),

        # 🔥 NEW FEATURES
        "INCOME": float(income),
        "PREVIOUS_DEFAULT": float(default_map[previous_default]),
        "CREDIT_UTIL_INPUT": float(credit_util),

        # payment history
        "PAY_0": base_delay,
        "PAY_2": base_delay,
        "PAY_3": base_delay,
        "PAY_4": base_delay,
        "PAY_5": base_delay,
        "PAY_6": base_delay,

        # bills
        "BILL_AMT1": float(avg_bill),
        "BILL_AMT2": float(avg_bill),
        "BILL_AMT3": float(avg_bill),
        "BILL_AMT4": float(avg_bill),
        "BILL_AMT5": float(avg_bill),
        "BILL_AMT6": float(avg_bill),

        # payments
        "PAY_AMT1": float(avg_pay),
        "PAY_AMT2": float(avg_pay),
        "PAY_AMT3": float(avg_pay),
        "PAY_AMT4": float(avg_pay),
        "PAY_AMT5": float(avg_pay),
        "PAY_AMT6": float(avg_pay),
    }

    input_df = pd.DataFrame([input_dict])

    try:
        # feature engineering
        input_df_fe = feature_engineering(input_df)

        # align columns
        input_df_fe = input_df_fe.reindex(columns=feature_order, fill_value=0)

        # predict
        pred = pipeline.predict(input_df_fe)

        st.divider()

        if pred[0] == 1:
            st.error("🚩 High Risk Client")
        else:
            st.success("✅ Low Risk Client")

    except Exception as e:
        st.error(f"Error: {e}")

# ---------------- DEBUG ----------------
with st.expander("🔍 Debug"):
    if input_df is not None:
        st.write(input_df)