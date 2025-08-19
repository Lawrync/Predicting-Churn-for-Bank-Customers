import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("Customer-Churn-Records.csv")

# Preprocess the data
def preprocess_data(df):
    X = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited',
                 'Complain', 'Satisfaction Score', 'Point Earned'], axis=1)
    y = df['Exited']

    numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance',
                        'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                        'EstimatedSalary']
    categorical_features = ['Geography', 'Gender', 'Card Type']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )

    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor

# Train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = XGBClassifier(objective="binary:logistic", eval_metric="auc", random_state=42)
    model.fit(X_train, y_train)
    return model

# Main app
def main():
    st.title("💳 Customer Churn Prediction")

    # Load and preprocess data
    data = load_data()
    X, y, preprocessor = preprocess_data(data)

    # Train model
    model = train_model(X, y)

    # Sidebar inputs
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
    prediction = model.predict(input_processed)
    probability = model.predict_proba(input_processed)[0][1]

    # Display prediction
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error(f" Customer is likely to churn. Probability: {probability:.2%}")
    else:
        st.success(f" Customer is not likely to churn. Probability: {probability:.2%}")

if __name__ == "__main__":
    main()







