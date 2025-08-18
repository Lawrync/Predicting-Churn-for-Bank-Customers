{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dd5bad24-3297-49cd-af28-33fdbc6a4ffe",
   "metadata": {},
   "outputs": [],
   "source": [
    "# streamlit_app.py\n",
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import joblib\n",
    "\n",
    "# ==== Load model & preprocessor with caching ====\n",
    "@st.cache_resource\n",
    "def load_model_and_preprocessor():\n",
    "    preprocessor = joblib.load(\"preprocessor.pkl\")\n",
    "    model = joblib.load(\"xgb_churn_model.pkl\")\n",
    "    return preprocessor, model\n",
    "\n",
    "# ==== Main app function ====\n",
    "def main():\n",
    "    st.set_page_config(page_title=\"Customer Churn Predictor\", layout=\"centered\")\n",
    "    st.title(\"📊 Customer Churn Prediction App\")\n",
    "    st.write(\"Enter customer details below to predict churn probability.\")\n",
    "\n",
    "    # Load preprocessor and model\n",
    "    preprocessor, model = load_model_and_preprocessor()\n",
    "\n",
    "    # Sidebar for input parameters\n",
    "    st.sidebar.header(\"Enter Customer Details\")\n",
    "    credit_score = st.sidebar.number_input(\"Credit Score\", 350, 850, 600)\n",
    "    age = st.sidebar.number_input(\"Age\", 18, 92, 30)\n",
    "    tenure = st.sidebar.number_input(\"Tenure (Years with Bank)\", 0, 10, 2)\n",
    "    balance = st.sidebar.number_input(\"Balance\", 0.0, 250000.0, 8000.0, step=100.0)\n",
    "    num_products = st.sidebar.selectbox(\"Number of Products\", [1, 2, 3, 4], index=1)\n",
    "    has_card = st.sidebar.radio(\"Has Credit Card?\", [0, 1], index=1)\n",
    "    is_active = st.sidebar.radio(\"Is Active Member?\", [0, 1], index=1)\n",
    "    salary = st.sidebar.number_input(\"Estimated Salary\", 0.0, 200000.0, 60000.0, step=500.0)\n",
    "    geography = st.sidebar.selectbox(\"Geography\", [\"France\", \"Spain\", \"Germany\"])\n",
    "    gender = st.sidebar.selectbox(\"Gender\", [\"Female\", \"Male\"])\n",
    "    card_type = st.sidebar.selectbox(\"Card Type\", [\"SILVER\", \"GOLD\", \"PLATINUM\", \"DIAMOND\"])\n",
    "\n",
    "    # Predict button\n",
    "    if st.sidebar.button(\"🔮 Predict Churn\"):\n",
    "        # Prepare data for prediction\n",
    "        sample = pd.DataFrame([{\n",
    "            \"CreditScore\": credit_score,\n",
    "            \"Age\": age,\n",
    "            \"Tenure\": tenure,\n",
    "            \"Balance\": balance,\n",
    "            \"NumOfProducts\": num_products,\n",
    "            \"HasCrCard\": has_card,\n",
    "            \"IsActiveMember\": is_active,\n",
    "            \"EstimatedSalary\": salary,\n",
    "            \"Geography\": geography,\n",
    "            \"Gender\": gender,\n",
    "            \"Card_Type\": card_type\n",
    "        }])\n",
    "\n",
    "        # Transform and predict\n",
    "        X = preprocessor.transform(sample)\n",
    "        proba = model.predict_proba(X)[0][1]\n",
    "        pred = model.predict(X)[0]\n",
    "\n",
    "        # Display results\n",
    "        st.subheader(\"✅ Prediction Result\")\n",
    "        if pred:\n",
    "            st.error(f\"🔴 Customer is likely to churn! Probability: {proba:.2%}\")\n",
    "        else:\n",
    "            st.success(f\"🟢 Customer is unlikely to churn. Probability: {proba:.2%}\")\n",
    "\n",
    "# ==== Run the app ====\n",
    "if __name__ == \"__main__\":\n",
    "    main()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "288ce62e-db50-481a-a5ea-da50c111fb8b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Traceback (most recent call last):\n",
      "  File \"C:\\Users\\USER\\anaconda3\\Lib\\site-packages\\nbconvert\\filters\\markdown_mistune.py\", line 24, in <module>\n",
      "    from mistune import (  # type:ignore[attr-defined]\n",
      "  File \"C:\\Users\\USER\\anaconda3\\Lib\\site-packages\\mistune\\__init__.py\", line 4, in <module>\n",
      "    from .renderers import AstRenderer, HTMLRenderer\n",
      "ImportError: cannot import name 'AstRenderer' from 'mistune.renderers' (C:\\Users\\USER\\anaconda3\\Lib\\site-packages\\mistune\\renderers\\__init__.py)\n",
      "\n",
      "During handling of the above exception, another exception occurred:\n",
      "\n",
      "Traceback (most recent call last):\n",
      "  File \"C:\\Users\\USER\\anaconda3\\Scripts\\jupyter-nbconvert-script.py\", line 6, in <module>\n",
      "    from nbconvert.nbconvertapp import main\n",
      "  File \"C:\\Users\\USER\\anaconda3\\Lib\\site-packages\\nbconvert\\__init__.py\", line 7, in <module>\n",
      "    from .exporters import (\n",
      "  File \"C:\\Users\\USER\\anaconda3\\Lib\\site-packages\\nbconvert\\exporters\\__init__.py\", line 4, in <module>\n",
      "    from .html import HTMLExporter\n",
      "  File \"C:\\Users\\USER\\anaconda3\\Lib\\site-packages\\nbconvert\\exporters\\html.py\", line 29, in <module>\n",
      "    from nbconvert.filters.markdown_mistune import IPythonRenderer, MarkdownWithMath\n",
      "  File \"C:\\Users\\USER\\anaconda3\\Lib\\site-packages\\nbconvert\\filters\\markdown_mistune.py\", line 39, in <module>\n",
      "    from mistune import (  # type: ignore[attr-defined]\n",
      "  File \"C:\\Users\\USER\\anaconda3\\Lib\\site-packages\\mistune\\__init__.py\", line 4, in <module>\n",
      "    from .renderers import AstRenderer, HTMLRenderer\n",
      "ImportError: cannot import name 'AstRenderer' from 'mistune.renderers' (C:\\Users\\USER\\anaconda3\\Lib\\site-packages\\mistune\\renderers\\__init__.py)\n"
     ]
    }
   ],
   "source": [
    "!jupyter nbconvert --to script churn_prediction.ipynb\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "16e90959-4665-46ae-b860-3c998fb1e2d9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: mistune==2.0.4 in c:\\users\\user\\anaconda3\\lib\\site-packages (2.0.4)Note: you may need to restart the kernel to use updated packages.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "pip install mistune==2.0.4\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "a8220361-d211-4e9b-9122-a4e50ac8ce5b",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (2530953675.py, line 1)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  Cell \u001b[1;32mIn[11], line 1\u001b[1;36m\u001b[0m\n\u001b[1;33m    jupyter nbconvert --to script churn_prediction.ipynb\u001b[0m\n\u001b[1;37m            ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": [
    "jupyter nbconvert --to script churn_prediction.ipynb\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b6771c64-251a-4aef-ba1f-5a5aba3ee725",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Traceback (most recent call last):\n",
      "  File \"C:\\Users\\USER\\anaconda3\\Lib\\site-packages\\nbconvert\\filters\\markdown_mistune.py\", line 24, in <module>\n",
      "    from mistune import (  # type:ignore[attr-defined]\n",
      "  File \"C:\\Users\\USER\\anaconda3\\Lib\\site-packages\\mistune\\__init__.py\", line 4, in <module>\n",
      "    from .renderers import AstRenderer, HTMLRenderer\n",
      "ImportError: cannot import name 'AstRenderer' from 'mistune.renderers' (C:\\Users\\USER\\anaconda3\\Lib\\site-packages\\mistune\\renderers\\__init__.py)\n",
      "\n",
      "During handling of the above exception, another exception occurred:\n",
      "\n",
      "Traceback (most recent call last):\n",
      "  File \"C:\\Users\\USER\\anaconda3\\Scripts\\jupyter-nbconvert-script.py\", line 6, in <module>\n",
      "    from nbconvert.nbconvertapp import main\n",
      "  File \"C:\\Users\\USER\\anaconda3\\Lib\\site-packages\\nbconvert\\__init__.py\", line 7, in <module>\n",
      "    from .exporters import (\n",
      "  File \"C:\\Users\\USER\\anaconda3\\Lib\\site-packages\\nbconvert\\exporters\\__init__.py\", line 4, in <module>\n",
      "    from .html import HTMLExporter\n",
      "  File \"C:\\Users\\USER\\anaconda3\\Lib\\site-packages\\nbconvert\\exporters\\html.py\", line 29, in <module>\n",
      "    from nbconvert.filters.markdown_mistune import IPythonRenderer, MarkdownWithMath\n",
      "  File \"C:\\Users\\USER\\anaconda3\\Lib\\site-packages\\nbconvert\\filters\\markdown_mistune.py\", line 39, in <module>\n",
      "    from mistune import (  # type: ignore[attr-defined]\n",
      "  File \"C:\\Users\\USER\\anaconda3\\Lib\\site-packages\\mistune\\__init__.py\", line 4, in <module>\n",
      "    from .renderers import AstRenderer, HTMLRenderer\n",
      "ImportError: cannot import name 'AstRenderer' from 'mistune.renderers' (C:\\Users\\USER\\anaconda3\\Lib\\site-packages\\mistune\\renderers\\__init__.py)\n"
     ]
    }
   ],
   "source": [
    "!jupyter nbconvert --to script churn_prediction.ipynb\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5041c585-e100-4d7e-a5ad-c2c46ca94b45",
   "metadata": {},
   "outputs": [],
   "source": []
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
