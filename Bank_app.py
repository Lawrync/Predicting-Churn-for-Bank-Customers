import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from PIL import Image
import os

# --- Set background image using CSS ---
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = "data:image/png;base64," + (data.encode("base64") if hasattr(data, "encode") else data)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{data.encode('base64') if hasattr(data,'encode') else ''}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call background function
add_bg_from_local("images/large-corporates-will-never-be-allowed-to-open-a-bank-in-india-n-vaghul.webp")

# --- Load dataset ---
@st.cache_data
def load_data():
    return pd.read_csv("Customer-Churn-Records.csv")

data = load_data()

# --- Sidebar layout with two columns ---
st.sidebar.title("📊 Input Features")

col1, col2 = st.sidebar.columns(2)

with col1:
    tenure = st.number_input("Tenure", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0, value=50)

with col2:
    total_charges = st.number_input("Total Charges", min_value=0, value=600)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

# --- Dummy model training for demo ---
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Column transformer (categorical + numerical)
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(preprocessor.fit_transform(X_train), y_train)

# --- Predict ---
pred = model.predict(preprocessor.transform(X_test))[0]
proba = model.predict_proba(preprocessor.transform(X_test))[0][1]

# --- Show result ---
st.subheader("✅ Prediction Result")
st.write("**Churn Prediction:**", "🔴 Yes" if pred == 1 else "🟢 No")
st.write("**Churn Probability:**", f"{proba:.2%}")

