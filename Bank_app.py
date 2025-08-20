import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load Model and Scaler
# -----------------------------
@st.cache_resource
def load_model():
    with open("churn_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    return scaler

model = load_model()
scaler = load_scaler()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🏦 Bank Customer Churn Prediction App")
st.write("Predict whether a customer is likely to **Churn (leave)** or **Stay (retain)**.")

# -----------------------------
# Input Fields
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=20, value=5)
    balance = st.number_input("Balance", min_value=0.0, value=50000.0)

with col2:
    num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0)

# Categorical features
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])

# -----------------------------
# Convert Inputs
# -----------------------------
has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active_member = 1 if is_active_member == "Yes" else 0
gender = 1 if gender == "Male" else 0

# Geography encoding
geo_france = 1 if geography == "France" else 0
geo_spain = 1 if geography == "Spain" else 0
geo_germany = 1 if geography == "Germany" else 0

# -----------------------------
# Prepare Input Data
# -----------------------------
input_data = np.array([[credit_score, age, tenure, balance,
                        num_of_products, has_cr_card, is_active_member,
                        estimated_salary, gender, geo_france,
                        geo_spain, geo_germany]])

input_data_scaled = scaler.transform(input_data)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_data_scaled)[0]
    prob = model.predict_proba(input_data_scaled)[0]

    retain_prob = round(prob[0] * 100, 2)
    churn_prob = round(prob[1] * 100, 2)

    # Display results
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("⚠️ The customer is likely to **Churn**.")
    else:
        st.success("✅ The customer is likely to **Stay**.")

    # Show probabilities
    st.write(f"🟢 **Stay:** {retain_prob:.2f}%")
    st.write(f"🔴 **Churn:** {churn_prob:.2f}%")

    # Add interpretation
    if churn_prob > 50:
        st.warning("The customer is at **high risk** of leaving. Consider retention strategies.")
    else:
        st.info("The customer is **likely to stay**, but continue monitoring engagement.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("📊 Built with Streamlit | Customer Churn Prediction Model")
ain__":
    main()
ion==1 else 'Retain'}**")

# Run app
if __name__ == "__main__":
    main()







