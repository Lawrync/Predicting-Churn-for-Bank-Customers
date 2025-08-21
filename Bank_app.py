import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("churn_model.pkl")

st.title("🏦 Customer Churn Prediction")

# Layout with two columns
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("User Inputs")
    credit_score = st.number_input("Credit Score", 300, 900, 600)
    age = st.number_input("Age", 18, 100, 30)
    tenure = st.number_input("Tenure (Years)", 0, 20, 2)
    balance = st.number_input("Balance", 0.0, 250000.0, 8000.0)
    num_products = st.number_input("Number of Products", 1, 4, 1)
    has_card = st.selectbox("Has Credit Card", ["Yes", "No"])
    is_active = st.selectbox("Is Active Member", ["Yes", "No"])
    salary = st.number_input("Estimated Salary", 0.0, 200000.0, 60000.0)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])

    if has_card == "Yes":
        has_card = 1
    else:
        has_card = 0

    if is_active == "Yes":
        is_active = 1
    else:
        is_active = 0

    # Prepare input
    input_data = np.array([[credit_score, age, tenure, balance, num_products,
                            has_card, is_active, salary]])
    
with col2:
    st.subheader("Prediction")
    if st.button("Predict"):
        prob = model.predict_proba(input_data)[0]
        pred = model.predict(input_data)[0]

        if pred == 1:
            result = "Churn"
        else:
            result = "Retain"

        st.write(f"**Predicted Value:** {result}")
        st.write(f"🔴 Probability (Churn): {prob[1]*100:.2f}%")
        st.write(f"🟢 Probability (Retain): {prob[0]*100:.2f}%")
        st.success(f"**Output: {result}**")
_":
    main()
ion==1 else 'Retain'}**")

# Run app
if __name__ == "__main__":
    main()







