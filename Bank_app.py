import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Load Model & Preprocessor
# -----------------------------
MODEL_PATH = "model.pkl"
PREPROCESSOR_PATH = "preprocessor.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Bank Customer Churn Prediction", layout="wide")

def main():
    st.title("🏦 Bank Customer Churn Prediction App")
    st.markdown("Predict whether a customer will **churn** or **stay** based on their information.")

    # Layout: Two columns
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=20, value=5)
        balance = st.number_input("Balance", min_value=0.0, max_value=250000.0, value=50000.0)

    with col2:
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_cr_card = st.selectbox("Has Credit Card?", [0, 1])
        is_active_member = st.selectbox("Active Member?", [0, 1])
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=250000.0, value=70000.0)

    # Extra categorical features
    col3, col4 = st.columns(2)
    with col3:
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    with col4:
        gender = st.selectbox("Gender", ["Male", "Female"])

    # Prepare input
    input_data = pd.DataFrame([{
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary,
        "Geography": geography,
        "Gender": gender
    }])

    # -----------------------------
    # Prediction
    # -----------------------------
    if st.button("🔍 Predict"):
        input_processed = preprocessor.transform(input_data)
        prediction = model.predict(input_processed)[0]
        prob = model.predict_proba(input_processed)[0]

        churn_prob = prob[1] * 100
        retain_prob = prob[0] * 100

        if prediction == 1:
            st.error("❌ **Predicted Value: Churn**")
        else:
            st.success("✅ **Predicted Value: Retain**")

        st.write(f"🔴 **Churn Probability:** {churn_prob:.2f}%")
        st.write(f"🟢 **Retain Probability:** {retain_prob:.2f}%")

        st.info(f"**Final Output:** {'Churn' if prediction == 1 else 'Retain'}")

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    main()
push origin main
Retain): {retain_prob:.2f}%")

        st.info(f"**Final Output: {'Churn' if prediction == 1 else 'Retain'}**")
 'Retain'}**")      st.write(f"🟢 **Stay:** {retain_prob:.2f}%")


if __name__ == "__main__":
    main()
ion==1 else 'Retain'}**")

# Run app
if __name__ == "__main__":
    main()







