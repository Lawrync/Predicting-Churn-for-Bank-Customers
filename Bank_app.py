import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from PIL import Image
import os

# --- Display images side by side ---
IMAGE_FOLDER = "images"
img1 = Image.open(os.path.join(IMAGE_FOLDER, "large-corporates-will-never-be-allowed-to-open-a-bank-in-india-n-vaghul.webp"))
img2 = Image.open(os.path.join(IMAGE_FOLDER, "interior-design-bank-office-employees-600nw-2307454537.webp"))

col1, col2 = st.columns([1, 1])
col1.image(img1, use_container_width=True)
col2.image(img2, use_container_width=True)

# --- Load dataset ---
@st.cache_data
def load_data():
    return pd.read_csv("Customer-Churn-Records.csv")

# --- Preprocess data ---
@st.cache_data
def preprocess_data(df):
    X = df.drop([
        'RowNumber', 'CustomerId', 'Surname', 'Exited',
        'Complain', 'Satisfaction Score', 'Point Earned'
    ], axis=1)
    y = df['Exited']

    numeric_features = [
        'CreditScore', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
    ]
    categorical_features = ['Geography', 'Gender', 'Card Type']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )

    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor

# --- Train model ---
@st.cache_data
def train_model(X, y):
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42
    )
    model.fit(X, y)
    return model

# --- Main app ---
def main():
    st.title("💳 Customer Churn Prediction")

    # Load and preprocess data
    data = load_data()
    X_processed, y, preprocessor = preprocess_data(data)

    # Train model 
    model = train_model(X_processed, y)

    # Sidebar input
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

    # Transform and predict
    input_processed = preprocessor.transform(input_data)
    prediction = model.predict(input_processed)[0]
    probability = model.predict_proba(input_processed)[0][1]

    # Display result
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"Customer is likely to churn. Probability: {probability:.2%}")
    else:
        st.success(f"Customer is not likely to churn. Probability: {1 - probability:.2%}")

if __name__ == "__main__":
    main()
f"**Final Output: {'Churn' if prediction==1 else 'Retain'}**")

if __name__ == "__main__":
    main()
":
    main()
ion==1 else 'Retain'}**")

# Run app
if __name__ == "__main__":
    main()







