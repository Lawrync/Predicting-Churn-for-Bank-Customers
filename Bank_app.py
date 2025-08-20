import streamlit as st
import pickle
import numpy as np
import os
from PIL import Image

# -------------------------
# Paths for model & assets
# -------------------------
MODEL_PATH = "models/churn_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
IMAGE_FOLDER = "images"

# -------------------------
# Load model & preprocessor
# -------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)

# -------------------------
# App Config
# -------------------------
st.set_page_config(page_title="Bank Customer Churn Prediction", layout="wide")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Customer Prediction"])

# -------------------------
# Home Page
# -------------------------
if page == "Home":
    st.title("🏦 Bank Customer Churn Prediction App")

    img1_path = os.path.join(IMAGE_FOLDER, "large-corporates-will-never-be-allowed-to-open-a-bank-in-india-n-vaghul.webp")
    img2_path = os.path.join(IMAGE_FOLDER, "interior-design-bank-office-employees-600nw-2307454537.webp")

    try:
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img1, caption="Banking Sector")
        with col2:
            st.image(img2, caption="Bank Office Interior")
    except Exception as e:
        st.warning("⚠️ Images not found. Please check the 'images' folder.")

    st.markdown("""
    This app predicts **whether a bank customer will churn or stay** using their profile details.
    
    - **Churn** → The customer is likely to leave the bank ❌  
    - **Retain** → The customer is likely to stay ✅  

    Navigate to **Customer Prediction** from the sidebar to try it out!
    """)

# -------------------------
# Prediction Page
# -------------------------
elif page == "Customer Prediction":
    st.title("🔍 Customer Churn Prediction")

    # Wide form layout
    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=50, value=5)

        with col2:
            balance = st.number_input("Balance", min_value=0.0, value=50000.0, step=1000.0)
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
            has_cr_card = st.selectbox("Has Credit Card", [0, 1])

        with col3:
            is_active = st.selectbox("Is Active Member", [0, 1])
            estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0, step=1000.0)
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Male", "Female"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Arrange input into model format
        input_data = np.array([[credit_score, geography, gender, age, tenure, balance,
                                num_products, has_cr_card, is_active, estimated_salary]])

        # Preprocess
        input_processed = preprocessor.transform(input_data)

        # Predict
        prediction = model.predict(input_processed)[0]
        prob = model.predict_proba(input_processed)[0]

        churn_prob = prob[1] * 100
        retain_prob = prob[0] * 100

        # Results
        if prediction == 1:
            st.error(f"**Predicted Value: Churn** ❌")
        else:
            st.success(f"**Predicted Value: Retain** ✅")

        st.write(f"🔴 Probability (Churn): {churn_prob:.2f}%")
        st.write(f"🟢 Probability (Retain): {retain_prob:.2f}%")

        st.info(f"**Final Output: {'Churn' if prediction == 1 else 'Retain'}**")
 'Retain'}**")      st.write(f"🟢 **Stay:** {retain_prob:.2f}%")


if __name__ == "__main__":
    main()
ion==1 else 'Retain'}**")

# Run app
if __name__ == "__main__":
    main()







