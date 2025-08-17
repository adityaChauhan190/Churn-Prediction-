import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("churn_model.pkl")

# App configuration
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")

# Custom CSS for modern UI
st.markdown("""
    <style>
        /* Global font */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Main background */
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 2rem;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            padding: 2rem 1rem;
            border-right: 1px solid #eee;
        }

        /* Prediction Card */
        .prediction-card {
            padding: 2rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.75);
            backdrop-filter: blur(15px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.08);
            text-align: center;
            margin-bottom: 2rem;
        }

        .prediction-card h2 {
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
        }

        .probability {
            font-size: 1.2rem;
            font-weight: 500;
            color: #333;
        }

        /* Buttons */
        .stButton button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
            border-radius: 12px;
            border: none;
            transition: 0.3s;
        }
        .stButton button:hover {
            background: linear-gradient(90deg, #5a67d8 0%, #6b46c1 100%);
            transform: translateY(-2px);
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📉 Customer Churn Prediction Dashboard")
st.markdown("A professional tool to analyze and predict the probability of customer churn.")

# Layout: two columns
col1, col2 = st.columns([1,2])

with col1:
    st.subheader("🔧 Enter Customer Details")

    credit_score = st.number_input("Credit Score", 300, 900, 650)
    country = st.selectbox("Country", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 35)
    tenure = st.slider("Tenure (Years)", 0, 10, 3)
    balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
    products_number = st.slider("Number of Products", 1, 4, 1)
    credit_card = st.selectbox("Has Credit Card?", [0, 1])
    active_member = st.selectbox("Active Member?", [0, 1])
    estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 70000.0)

    predict_btn = st.button("🔮 Predict Churn")

with col2:
    if predict_btn:
        sample = pd.DataFrame([{
            "credit_score": credit_score,
            "country": country,
            "gender": gender,
            "age": age,
            "tenure": tenure,
            "balance": balance,
            "products_number": products_number,
            "credit_card": credit_card,
            "active_member": active_member,
            "estimated_salary": estimated_salary
        }])
        
        pred = model.predict(sample)[0]
        proba = model.predict_proba(sample)[:,1][0]

        # Show Prediction Card
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        if pred == 1:
            st.markdown("<h2>🚨 Customer is LIKELY to Churn</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2>✅ Customer will Stay</h2>", unsafe_allow_html=True)
        st.markdown(f"<p class='probability'>Churn Probability: <b>{round(proba*100, 2)}%</b></p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Feature Importance
        st.subheader("📊 Feature Importances")
        importances = model.named_steps['rf_clf'].feature_importances_
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10,5))
        feat_imp.plot(kind='bar', ax=ax, color="#667eea", edgecolor="black")
        plt.title("Feature Importance Ranking")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
