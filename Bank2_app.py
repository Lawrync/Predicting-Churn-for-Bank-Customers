import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from PIL import Image
import os
# ==== Load saved preprocessor & model ====
# (make sure preprocessor.pkl and xgb_churn_model.pkl are in the same folder)
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("xgb_churn_model.pkl")

# ==== Streamlit App ====
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📊 Customer Churn Prediction App")

st.markdown("""
This app uses a tuned **XGBoost model** to predict whether a bank customer is likely to churn.  
Enter the customer details below and click **Predict**.
""")

# ==== Input form ====
with st.form("churn_form"):
    credit_score = st.number_input("Credit Score", min_value=350, max_value=850, value=600)
    age = st.number_input("Age", min_value=18, max_value=92, value=30)
    tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=10, value=2)
    balance = st.number_input("Balance", min_value=0.0, max_value=250000.0, value=8000.0, step=100.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
    has_card = st.radio("Has Credit Card?", [0, 1], index=1)
    is_active = st.radio("Is Active Member?", [0, 1], index=1)
    salary = st.number_input("Estimated Salary", min_value=0.0, max_value=200000.0, value=60000.0, step=500.0)

    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    card_type = st.selectbox("Card Type", ["SILVER", "GOLD", "PLATINUM", "DIAMOND"])

    submitted = st.form_submit_button("🔮 Predict Churn")

# ==== Prediction ====
if submitted:
    # Prepare data
    sample = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": salary,
        "Geography": geography,
        "Gender": gender,
        "Card Type": card_type
    }

    df = pd.DataFrame([sample])

    # Transform + predict
    X = preprocessor.transform(df)
    proba = model.predict_proba(X)[0][1]
    pred = model.predict(X)[0]

    # Show result
    st.subheader("✅ Prediction Result")
    st.write("**Churn Prediction:**", "🔴 Yes" if pred else "🟢 No")
    st.write("**Churn Probability:**", f"{proba:.2%}")
