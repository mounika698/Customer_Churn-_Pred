# Customer Churn Prediction Web App

An interactive web app built with **Streamlit** that predicts whether a customer will churn (leave) or stay, using a trained **Random Forest** machine learning model.

---

## 🔥 Features

- **Predict churn** by entering customer details through a clean and intuitive interface.
- **Download** full prediction reports with all entered details.
- Explore **data visualizations** to understand trends in the customer dataset.
- View **model performance metrics** including accuracy, precision, and recall.
- Simple **navigation** between Prediction, Visualization, and Statistics pages.
- **Logs** predictions into a CSV for tracking over time.

---

## 🚀 Quick Start — How to Run Locally

1. **Clone the repository:**

   ```bash
   git clone https://github.com/mounika698/churn-pred.git
   cd churn-prediction


```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment (macOS/Linux)
source venv/bin/activate

# Activate virtual environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run apps.py
