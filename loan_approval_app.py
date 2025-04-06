import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Paths
MODEL_PATH = "xgb_credit_model.pkl"
SCALER_PATH = "scaler.pkl"
DATA_PATH = "sample_lendingclub_data.csv"

# Cache training
@st.cache_resource
def train_and_save_model():
    df = pd.read_csv(DATA_PATH)

    # Clean and derive
    df = df.dropna(subset=['loan_amnt', 'int_rate', 'annual_inc', 'dti',
                           'delinq_2yrs', 'revol_util', 'loan_status',
                           'grade', 'term', 'home_ownership'])
    df['loan_status'] = df['loan_status'].apply(lambda x: 1 if x == 'Fully Paid' else 0)
    df['past_defaults'] = np.random.randint(0, 3, df.shape[0])
    df['monthly_salary'] = df['annual_inc'] / 12
    df['available_balance'] = df['monthly_salary'] * np.random.uniform(0.3, 0.8, df.shape[0])

    df = pd.get_dummies(df, columns=['term', 'grade', 'home_ownership'])

    y = df['loan_status']
    X = df.drop(columns=['loan_status'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_scaled, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return model, scaler, X.columns.tolist()

# Load model
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    model, scaler, expected_features = train_and_save_model()
else:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    df = pd.read_csv(DATA_PATH)
    df['past_defaults'] = np.random.randint(0, 3, df.shape[0])
    df['monthly_salary'] = df['annual_inc'] / 12
    df['available_balance'] = df['monthly_salary'] * 0.5
    df = pd.get_dummies(df, columns=['term', 'grade', 'home_ownership'])
    expected_features = [col for col in df.columns if col != 'loan_status']

explainer = shap.Explainer(model)

# Streamlit UI
st.title("💰 Smart Loan Approval System")
st.markdown("Upload an applicant's financial record (CSV) to evaluate loan eligibility.")

uploaded_file = st.file_uploader("📂 Upload Applicant CSV", type=["csv"])
threshold = st.slider("Set approval threshold", 0.0, 1.0, 0.6, 0.01)

if uploaded_file:
    applicant_df = pd.read_csv(uploaded_file)

    try:
        # Derive new fields
        applicant_df['past_defaults'] = 0
        applicant_df['monthly_salary'] = applicant_df['annual_inc'] / 12
        applicant_df['available_balance'] = applicant_df['monthly_salary'] * 0.5

        # One-hot encode
        applicant_df = pd.get_dummies(applicant_df)

        # Add missing columns
        for col in expected_features:
            if col not in applicant_df:
                applicant_df[col] = 0

        # Reorder
        applicant_df = applicant_df[expected_features]

        # Scale
        applicant_scaled = scaler.transform(applicant_df)

        # Predict
        prob = model.predict_proba(applicant_scaled)[0][1]
        decision = "✅ Loan Approved" if prob >= threshold else "❌ Loan Rejected"

        # Result
        st.subheader("📊 Prediction Result")
        st.write(f"**Approval Score:** `{prob:.2f}`")
        st.write(f"**Decision:** `{decision}`")

        # SHAP
        shap_values = explainer(applicant_scaled)
        st.subheader("🔍 Feature Impact (SHAP)")
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(bbox_inches='tight')

        # Suggestions
        if prob < threshold:
            st.subheader("💡 Suggestions to Improve Credit Score")
            st.markdown("- Pay off existing debts to reduce DTI")
            st.markdown("- Avoid missing loan payments")
            st.markdown("- Increase income or reduce expenses")
            st.markdown("- Maintain low credit utilization (revol_util)")

        # Display data
        st.subheader("🧾 Applicant's Processed Data")
        st.dataframe(applicant_df)

    except Exception as e:
        st.error("🚨 Error during processing. Please check your CSV format.")
        st.code(str(e))

