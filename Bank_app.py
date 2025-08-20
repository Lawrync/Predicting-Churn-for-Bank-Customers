import streamlit as st
import pickle
import pandas as pd
import os
from PIL import Image

# Load model & preprocessor
MODEL_PATH = "models/churn_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)

# Image folder
IMAGE_FOLDER = "images"
img1 = Image.open(os.path.join(IMAGE_FOLDER, "large-corporates-will-never-be-allowed-to-open-a-bank-in-india-n-vaghul.webp"))
img2 = Image.open(os.path.join(IMAGE_FOLDER, "interior-design-bank-office-employees-600nw-2307454537.webp"))

# Main function
def main():
    st.set_page_config(page_title="Bank Customer Churn Prediction", layout="wide")

    # Layout with two columns
    col1, col2 = st.columns([2, 3])

    with col1:
        st.image(img1, caption="Bank Exterior", use_container_width=True)
        st.image(img2, caption="Bank Interior", use_container_width=True)

    with col2:
        st.title("🏦 Bank Customer Churn Prediction App")
        st.write("Enter customer details to predict whether they are likely to churn or stay.")

        # Inputs side-by-side
        c1, c2 = st.columns(2)
        with c1:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            tenure = st.number_input("Tenure (years)", min_value=0, max_value=20, value=5)
            balance = st.number_input("Balance", min_value=0.0, value=50000.0, format="%.2f")

        with c2:
            num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
            has_cr_card = st.selectbox("Has Credit Card", [0, 1])
            is_active_member = st.selectbox("Is Active Member", [0, 1])
            estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, format="%.2f")

        # Combine inputs
        input_data = pd.DataFrame([[
            credit_score, age, tenure, balance, num_products,
            has_cr_card, is_active_member, estimated_salary
        ]], columns=[
            "CreditScore", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
        ])

        # 🔹 Prediction button & logic (this is what you were missing)
        if st.button("Predict"):
            input_processed = preprocessor.transform(input_data)
            prediction = model.predict(input_processed)[0]
            prob = model.predict_proba(input_processed)[0]

            churn_prob = prob[1] * 100
            retain_prob = prob[0] * 100

            if prediction == 1:
                st.error(f"**Predicted Value: Churn** ❌")
            else:
                st.success(f"**Predicted Value: Retain** ✅")

            st.write(f"🔴 Probability (Churn): {churn_prob:.2f}%")
            st.write(f"🟢 Probability (Retain): {retain_prob:.2f}%")
            st.info(f"**Final Output: {'Churn' if prediction==1 else 'Retain'}**")

# Run app
if __name__ == "__main__":
    main()







