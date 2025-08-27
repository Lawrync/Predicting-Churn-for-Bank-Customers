import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from PIL import Image
import os

# --- Display one image (centered & not full page) ---
IMAGE_FOLDER = "images"
img1 = Image.open(os.path.join(IMAGE_FOLDER, "large-corporates-will-never-be-allowed-to-open-a-bank-in-india-n-vaghul.webp"))

col1, col2, col3 = st.columns([1, 2, 1])  # center the image
with col2:
    st.image(img1, width=350)  # fixed width so it looks neat

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

       # Sidebar input (split into 2 columns)
    st.sidebar.title("Enter Customer Information")
    col1, col2 = st.sidebar.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600, step=1)
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=3, step=1)
        balance = st.number_input("Balance", min_value=0.0, max_value=250000.0, value=50000.0, step=100.0)
        num_products = st.selectbox("Products", [1, 2, 3, 4], index=0)

    with col2:
        has_card = st.selectbox("Has Card?", ["Yes", "No"])
        is_active = st.selectbox("Active Member?", ["Yes", "No"])
        salary = st.number_input("Salary", min_value=0.0, max_value=200000.0, value=60000.0, step=500.0)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        card_type = st.selectbox("Card Type", ["DIAMOND", "GOLD", "PLATINUM", "SILVER"])

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
    st.subheader("✅ Prediction Result")
    if prediction == 1:
        st.error(f"Customer is likely to churn. Probability: {probability:.2%}")
    else:
        st.success(f"Customer is not likely to churn. Probability: {1 - probability:.2%}")

    # Final output message
    st.markdown(f"**Final Output: {'Churn' if prediction==1 else 'Retain'}**")


if __name__ == "__main__":
    main()

