{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "c68e7fdd-f2f1-487f-a4f8-31bebaa17360",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-08-17 18:18:53.910 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\USER\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "# streamlit_app.py\n",
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import joblib\n",
    "\n",
    "# Load saved preprocessor and model\n",
    "preprocessor = joblib.load(\"preprocessor.pkl\")\n",
    "model = joblib.load(\"xgb_churn_model.pkl\")\n",
    "\n",
    "st.set_page_config(page_title=\"Customer Churn Predictor\", layout=\"centered\")\n",
    "st.title(\"📊 Customer Churn Prediction App\")\n",
    "\n",
    "st.write(\"Enter customer details below to predict churn probability.\")\n",
    "\n",
    "# Collect inputs\n",
    "credit_score = st.number_input(\"Credit Score\", min_value=350, max_value=850, value=600)\n",
    "age = st.number_input(\"Age\", min_value=18, max_value=92, value=30)\n",
    "tenure = st.number_input(\"Tenure (Years with Bank)\", min_value=0, max_value=10, value=2)\n",
    "balance = st.number_input(\"Balance\", min_value=0.0, max_value=250000.0, value=8000.0)\n",
    "num_products = st.selectbox(\"Number of Products\", [1, 2, 3, 4], index=1)\n",
    "has_card = st.selectbox(\"Has Credit Card?\", [0, 1], index=1)\n",
    "is_active = st.selectbox(\"Is Active Member?\", [0, 1], index=1)\n",
    "salary = st.number_input(\"Estimated Salary\", min_value=0.0, max_value=200000.0, value=60000.0)\n",
    "\n",
    "geography = st.selectbox(\"Geography\", [\"France\", \"Spain\", \"Germany\"])\n",
    "gender = st.selectbox(\"Gender\", [\"Female\", \"Male\"])\n",
    "card_type = st.selectbox(\"Card Type\", [\"SILVER\", \"GOLD\", \"PLATINUM\", \"DIAMOND\"])\n",
    "\n",
    "# Prediction button\n",
    "if st.button(\"🔮 Predict Churn\"):\n",
    "    # Prepare data\n",
    "    sample = {\n",
    "        \"CreditScore\": credit_score,\n",
    "        \"Age\": age,\n",
    "        \"Tenure\": tenure,\n",
    "        \"Balance\": balance,\n",
    "        \"NumOfProducts\": num_products,\n",
    "        \"HasCrCard\": has_card,\n",
    "        \"IsActiveMember\": is_active,\n",
    "        \"EstimatedSalary\": salary,\n",
    "        \"Geography\": geography,\n",
    "        \"Gender\": gender,\n",
    "        \"Card Type\": card_type\n",
    "    }\n",
    "\n",
    "    df = pd.DataFrame([sample])\n",
    "\n",
    "    # Preprocess & predict\n",
    "    X = preprocessor.transform(df)\n",
    "    proba = model.predict_proba(X)[0][1]\n",
    "    pred = model.predict(X)[0]\n",
    "\n",
    "    # Show results\n",
    "    st.subheader(\"✅ Prediction Result\")\n",
    "    st.write(\"Churn Prediction:\", \"🔴 Yes\" if pred else \"🟢 No\")\n",
    "    st.write(\"Churn Probability:\", f\"{proba:.2%}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a73e3778-85cb-4980-b980-987b888c35a1",
   "metadata": {},
   "outputs": [],
   "source": [
    "shoould i push to github then i deploy"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
