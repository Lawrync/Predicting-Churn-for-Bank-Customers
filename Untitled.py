#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import os

# ==== Load model & preprocessor with caching ====
@st.cache(allow_output_mutation=True)
def load_model_and_preprocessor():
    file_dir = os.path.dirname(__file__)  # ensures relative path
    preprocessor_path = os.path.join(file_dir, "preprocessor.pkl")
    model_path = os.path.join(file_dir, "xgb_churn_model.pkl")
    
    try:
        preprocessor = joblib.load(preprocessor_path)
    except FileNotFoundError:
        st.error(f"Preprocessor file not found at {preprocessor_path}")
        return None, None
    
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}")
        return None, None

    return preprocessor, model

# ==== Main app function ====
def main():
    st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
    st.title("📊 Customer Churn Prediction App")
    st.write("Enter customer details below to predict churn probability.")

    # Load preprocessor and model
    preprocessor, model = load_model_and_preprocessor()
    if preprocessor is None or model is None:
        st.stop()  # stop if files not loaded

    # Sidebar for input parameters
    st.sidebar.header("Enter Customer Details")
    credit_score = st.sidebar.number_input("Credit Score", 350, 850, 600)
    age = st.sidebar.number_input("Age", 18, 92, 30)
    tenure = st.sidebar.number_input("Tenure (Years with Bank)", 0, 10, 2)
    balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 8000.0, step=100.0)
    num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4], index=1)
    has_card = st.sidebar.radio("Has Credit Card?", [0, 1], index=1)
    is_active = st.sidebar.radio("Is Active Member?", [0, 1], index=1)
    salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 60000.0, step=500.0)
    geography = st.sidebar.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    card_type = st.sidebar.selectbox("Card Type", ["SILVER", "GOLD", "PLATINUM", "DIAMOND"])

    # Predict button
    if st.sidebar.button("🔮 Predict Churn"):
        # Prepare data for prediction
        sample = pd.DataFrame([{
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
            "Card_Type": card_type
        }])

        try:
            # Transform and predict
            X = preprocessor.transform(sample)
            proba = model.predict_proba(X)[0][1]
            pred = model.predict(X)[0]

            # Display results
            st.subheader("✅ Prediction Result")
            if pred:
                st.error(f"🔴 Customer is likely to churn! Probability: {proba:.2%}")
            else:
                st.success(f"🟢 Customer is unlikely to churn. Probability: {proba:.2%}")
        except ValueError as e:
            st.error(f"Error during prediction. Check input format: {e}")

# ==== Run the app ====
if __name__ == "__main__":
    main()







