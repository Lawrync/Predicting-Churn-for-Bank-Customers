import streamlit as st
import pandas as pd
from PIL import Image
import os
from sklearn.ensemble import RandomForestClassifier

# --- Streamlit page config ---
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# --- Image paths ---
IMAGE_FOLDER = "images"
img1_path = os.path.join(IMAGE_FOLDER, "large-corporates-will-never-be-allowed-to-open-a-bank-in-india-n-vaghul.webp")
img2_path = os.path.join(IMAGE_FOLDER, "interior-design-bank-office-employees-600nw-2307454537.webp")

img1 = Image.open(img1_path)
img2 = Image.open(img2_path)

# --- Display images ---
col1, col2, col3 = st.columns([1, 2, 1])
col1.image(img1, use_container_width=True)
col2.image(img2, use_container_width=True)
col3.write("")

# --- Load and preprocess data ---
def load_data():
    return pd.read_csv("Customer-Churn-Records.csv")

def preprocess_data(df):
    y = df["Exited"]
    X = df.drop("Exited", axis=1)
    
    # Encode categorical columns
    categorical_cols = ["Geography", "Gender", "Card Type"]
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    preprocessor = None  # Placeholder if you want to save/use a preprocessor
    return X, y, preprocessor

# --- Train model ---
def get_trained_model(X, y):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

# --- Main app ---
def main():
    st.title("💳 Customer Churn Prediction")

    # Load and preprocess
    data = load_data()
    X, y, preprocessor = preprocess_data(data)

    # Train model
    model = get_trained_model(X, y)

    # Sidebar for user input
    st.sidebar.title("Enter Customer Information")
    credit_score = st.sidebar.slider("Credit Score", 300, 900, 600)
    age = st.sidebar.slider("Age", 18, 100, 30)
    tenure = st.sidebar.slider("Tenure (Years with Bank)", 0, 10, 3)
    balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 50000.0, step=100.0)
    num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4], index=0)
    has_card = st.sidebar.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active = st.sidebar.selectbox("Is Active Member?", ["Yes", "No"])
    salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 60000.0, step=500.0)
    geography = st.sidebar.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    card_type = st.sidebar.selectbox("Card Type", ["DIAMOND", "GOLD", "PLATINUM", "SILVER"])

    # Prepare input
    input_data = pd.DataFrame([{
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": salary,
        "Geography": geography,
        "Gender": gender,
        "Card Type": card_type
    }])

    # Encode categorical columns to match training
    input_data = pd.get_dummies(input_data)
    # Add missing columns with 0
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[X.columns]  # reorder columns

    # Predict button
    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")
        st.write(f"**Predicted Value:** {'Churned' if prediction == 1 else 'Retained'}")
        st.write(f"**Predicted Probability:** {probability:.2%} (Churn) | {1-probability:.2%} (Retain)")

if __name__ == "__main__":
    main()






