import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from PIL import Image
import os

# --- Display images side by side (Header Banner) ---
IMAGE_FOLDER = "images"
img1 = Image.open(os.path.join(IMAGE_FOLDER, "bank_building.jpg"))
img2 = Image.open(os.path.join(IMAGE_FOLDER, "bank_office.jpg"))
col1, col2 = st.columns([1,1])
col1.image(img1, use_container_width=True)
col2.image(img2, use_container_width=True)

# --- Load dataset ---
@st.cache_data
def load_data():
    return pd.read_csv("Customer-Churn-Records.csv")

# --- Preprocess data ---
@st.cache_data
def preprocess_data(df):
    X = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited', 
                 'Complain', 'Satisfaction Score', 'Point Earned'], axis=1)
    y = df['Exited']
    
    numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 
                        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
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
    model = XGBClassifier(objective="binary:logistic", eval_metric="auc", random_state=42)
    model.fit(X, y)
    return model

# --- Main app ---
def main():
    st.title("🏦 Customer Churn Prediction")

    # Load and preprocess data
    data = load_data()
    X_processed, y, preprocessor = preprocess_data(data)
    model = train_model(X_processed, y)

    # Two-column layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Enter Customer Information")
        credit_score = st.number_input("Credit Score", 300, 900, 600)
        age = st.number_input("Age", 18, 100, 30)
        tenure = st.number_input("Tenure (Years with Bank)", 0, 10, 3)
        balance = st.number_input("Balance", 0.0, 250000.0, 50000.0, step=100.0)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)
        has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
        salary = st.number_input("Estimated Salary", 0.0, 200000.0, 60000.0, step=500.0)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        card_type = st.selectbox("Card Type", ["DIAMOND", "GOLD", "PLATINUM", "SILVER"])

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

    with col2:
        st.subheader("Prediction Result")

        # Optional image in prediction section
        churn_img = Image.open(os.path.join(IMAGE_FOLDER, "customer_churn.png"))
        st.image(churn_img, use_container_width=True)

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

if __name__ == "__main__":
    main()





