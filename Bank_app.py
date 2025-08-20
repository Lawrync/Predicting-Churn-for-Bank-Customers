import streamlit as st
import pandas as pd
from PIL import Image

# Set wide layout
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Load images
img1 = Image.open("large-corporates-will-never-be-allowed-to-open-a-bank-in-india-n-vaghul.jpg")
img2 = Image.open("interior-design-bank-office-employees-600nw-2307454537.jpg")

# Display images: first on left, second in center
col1, col2, col3 = st.columns([1, 2, 1])
col1.image(img1, use_column_width=True)
col2.image(img2, use_column_width=True)
col3.write("")  # empty for spacing

# --- Placeholder functions ---
def load_data():
    # Replace with actual loading code
    return pd.read_csv("Customer-Churn-Records.csv")

def preprocess_data(df):
    # Replace with actual preprocessing code
    X = df.drop("Exited", axis=1)
    y = df["Exited"]
    preprocessor = None  # Replace with your trained preprocessor
    return X, y, preprocessor

def get_trained_model(X, y):
    # Replace with loading your trained model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# Main app
def main():
    st.title("💳 Customer Churn Prediction")

    # Load and preprocess data
    data = load_data()
    X, y, preprocessor = preprocess_data(data)

    # Load trained model
    model = get_trained_model(X, y)

    # Sidebar for input
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

    # Predict button
    if st.button("Predict"):
        if preprocessor:
            input_processed = preprocessor.transform(input_data)
        else:
            input_processed = input_data  # Skip if no preprocessor

        prediction = model.predict(input_processed)[0]
        probability = model.predict_proba(input_processed)[0][1]

        st.subheader("Prediction Result")
        st.write(f"**Predicted Value:** {'Churned' if prediction == 1 else 'Retained'}")
        st.write(f"**Predicted Probability:** {probability:.2%} (Churn) | {1-probability:.2%} (Retain)")

if __name__ == "__main__":
    main()






