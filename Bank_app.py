import streamlit as st
import pickle
import numpy as np
import os

# ---------------------------
# Paths to model + preprocessor
# --------------------------
MODEL_PATH = os.path.join("models", "churn_model.pkl")
PREPROCESSOR_PATH = os.path.join("models", "preprocessor.pkl")

if not os.path.exists(MODEL_PATH):  # fallback if in root
    MODEL_PATH = "churn_model.pkl"
if not os.path.exists(PREPROCESSOR_PATH):
    PREPROCESSOR_PATH = "preprocessor.pkl"

# ---------------------------
# Load model + preprocessor
# ---------------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)
    return model, preprocessor

model, preprocessor = load_model()

# ---------------------------
# Streamlit App
# ---------------------------
def main():
    st.set_page_config(page_title="Bank Customer Churn Prediction", layout="wide")

    st.title("🏦 Bank Customer Churn Prediction")
    st.markdown("Enter customer details below to predict **whether they will Churn or Stay**.")

    # --- Organize inputs in 3 columns
    st.subheader("👤 Customer Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=50, value=5)
        balance = st.number_input("Balance", min_value=0.0, max_value=250000.0, value=50000.0)
    with col3:
        num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=200000.0, value=50000.0)

    # --- Create input array
    input_data = np.array([[
        credit_score,
        geography,
        gender,
        age,
        tenure,
        balance,
        num_products,
        1 if has_cr_card == "Yes" else 0,
        1 if is_active_member == "Yes" else 0,
        estimated_salary
    ]], dtype=object)

    # --- Prediction button
    if st.button("🔮 Predict Churn", use_container_width=True):
        input_processed = preprocessor.transform(input_data)
        prediction = model.predict(input_processed)[0]
        prob = model.predict_proba(input_processed)[0]

        churn_prob = prob[1] * 100
        retain_prob = prob[0] * 100

        # Highlight output
        if prediction == 1:
            st.error(f"### ❌ Customer is likely to CHURN\nChurn Probability: **{churn_prob:.2f}%**")
        else:
            st.success(f"### ✅ Customer is likely to STAY\nRetention Probability: **{retain_prob:.2f}%**")

        # Show probability breakdown
        st.progress(churn_prob / 100)
        st.write(f"🔴 **Churn:** {churn_prob:.2f}%")
        st.write(f"🟢 **Stay:** {retain_prob:.2f}%")


if __name__ == "__main__":
    main()
ion==1 else 'Retain'}**")

# Run app
if __name__ == "__main__":
    main()







