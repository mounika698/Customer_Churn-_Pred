# app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import base64
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# Load model and data
model = joblib.load("model.pkl")
df = pd.read_csv("customer_churn.csv")

# Encode for stats
df_encoded = df.copy()
df_encoded['Gender'] = df_encoded['Gender'].map({'Male': 0, 'Female': 1})
df_encoded['Subscription Type'] = df_encoded['Subscription Type'].map({'Basic': 0, 'Standard': 1, 'Premium': 2})
df_encoded['Contract Length'] = df_encoded['Contract Length'].map({'Monthly': 0, 'Quarterly': 1, 'Annual': 2})

X = df_encoded.drop(["CustomerID", "Churn"], axis=1)
y = df_encoded["Churn"]
y_pred = model.predict(X)

# Navigation
st.set_page_config(page_title="Customer Churn App", layout="wide")
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Customer Churn Prediction", "Data Visualization", "Model Stats & Status"])

# ---------------- Page 1: Prediction ----------------
if page == "Customer Churn Prediction":
    st.title("📉 Customer Churn Prediction")

    age = st.slider("Age", 18, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.number_input("Tenure (months)", 0, 60, 12)
    usage = st.number_input("Usage Frequency", 0, 50, 10)
    support = st.number_input("Support Calls", 0, 20, 2)
    payment_delay = st.number_input("Payment Delay (days)", 0, 30, 5)
    sub_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
    total_spend = st.number_input("Total Spend", 0, 2000, 500)
    last_interaction = st.number_input("Last Interaction (days ago)", 0, 30, 5)

    # Encode inputs
    gender_encoded = 0 if gender == "Male" else 1
    sub_encoded = {"Basic": 0, "Standard": 1, "Premium": 2}[sub_type]
    contract_encoded = {"Monthly": 0, "Quarterly": 1, "Annual": 2}[contract]

    features = [[age, gender_encoded, tenure, usage, support, payment_delay, sub_encoded, contract_encoded, total_spend, last_interaction]]
    prediction = model.predict(features)[0]

    st.write("### Churn Prediction")
    result_text = "Customer is likely to stay. ✅" if prediction == 0 else "Customer is likely to churn. ⚠️"
    st.success(result_text if prediction == 0 else result_text)

    # Log to CSV
    log_data = {
        "timestamp": [datetime.now()],
        "age": [age],
        "gender": [gender],
        "tenure": [tenure],
        "usage": [usage],
        "support_calls": [support],
        "payment_delay": [payment_delay],
        "subscription_type": [sub_type],
        "contract_length": [contract],
        "total_spend": [total_spend],
        "last_interaction": [last_interaction],
        "prediction": [result_text]
    }
    log_df = pd.DataFrame(log_data)
    log_df.to_csv("prediction_logs.csv", mode='a', header=not os.path.exists("prediction_logs.csv"), index=False)

    # Download Report
    st.markdown("### 📋 Full Prediction Summary")
    summary = f"""
Customer Churn Prediction Report

🕒 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🧍 Age: {age}
🚻 Gender: {gender}
📅 Tenure (months): {tenure}
📈 Usage Frequency: {usage}
📞 Support Calls: {support}
⏳ Payment Delay (days): {payment_delay}
📦 Subscription Type: {sub_type}
📝 Contract Length: {contract}
💰 Total Spend: ${total_spend}
📊 Last Interaction (days ago): {last_interaction}

📣 Prediction: {result_text}
    """
    st.text_area("📋 Full Report", summary, height=300)
    b64 = base64.b64encode(summary.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="churn_prediction.txt">📥 Download Full Report</a>'
    st.markdown(href, unsafe_allow_html=True)

# ---------------- Page 2: Data Visualization ----------------
elif page == "Data Visualization":
    st.title("📊 Data Visualization")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        churn_count = df['Churn'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(churn_count, labels=["Stayed", "Churned"], autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

    with col2:
        st.subheader("Contract Type vs Churn")
        fig2 = plt.figure(figsize=(6,4))
        sns.countplot(data=df, x="Contract Length", hue="Churn")
        st.pyplot(fig2)

    st.subheader("Correlation Heatmap")
    corr = df_encoded.drop(['CustomerID'], axis=1).corr()
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

# ---------------- Page 3: Model Stats & Status ----------------
elif page == "Model Stats & Status":
    st.title("📈 Model Statistics & Status")

    st.subheader("✅ Accuracy")
    acc = accuracy_score(y, y_pred)
    st.metric("Model Accuracy", f"{acc*100:.2f}%")

    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    st.subheader("📋 Classification Report")
    report = classification_report(y, y_pred, target_names=["Stayed", "Churned"])
    st.text(report)

    st.subheader("⭐ Feature Importances")
    importances = model.feature_importances_
    feature_names = X.columns
    sorted_idx = np.argsort(importances)[::-1]
    fig_imp, ax_imp = plt.subplots(figsize=(10, 5))
    sns.barplot(x=importances[sorted_idx], y=feature_names[sorted_idx], ax=ax_imp)
    st.pyplot(fig_imp)

