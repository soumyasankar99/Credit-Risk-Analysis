# 💰 Smart Loan Approval System - Credit Risk Analysis using XGBoost & SHAP


A machine learning-powered web application that predicts whether a loan should be **approved or rejected** based on the applicant’s financial history. The system not only delivers a prediction but also **explains the rationale** behind each decision using SHAP (SHapley Additive exPlanations) for transparency and trust.

---

## 🚀 Project Objectives

- ✅ **Predict Credit Risk** using an XGBoost classifier.
- 📈 **Analyze Financial Behavior** with key features like income, DTI, credit utilization, defaults, etc.
- 🔍 **Explain Model Decisions** using SHAP visualizations for better interpretability.
- 🖥️ **Interactive Streamlit App** for uploading applicant data and generating real-time predictions.
- 🎛️ **User-defined Approval Threshold** to customize decision sensitivity.
- 💡 **Suggestions to Improve Credit Score** for rejected applicants.

---

## 📊 Key Features

| Feature | Description |
|--------|-------------|
| 📂 CSV Upload | Upload single or batch applicant data for prediction |
| 🔎 SHAP Explainability | Waterfall plot shows key features influencing approval |
| ⚖️ Risk Threshold | Adjust the risk threshold to control loan approval leniency |
| 🧮 Feature Engineering | Derives `monthly_salary`, `available_balance`, and `past_defaults` |
| 🧠 Model | Trained XGBoost classifier with `StandardScaler` |
| 🔐 Secure and Transparent | ML with human-friendly explanations for fair decisioning |

---

## 📁 Project Structure

📦 loan-approval-system/ │ 

├── loan_approval_app.py # Main Streamlit app 

├── sample_lendingclub_data.csv  # Raw training data (LendingClub format) 

├── xgb_credit_model.pkl  # Trained XGBoost model 

├── scaler.pkl  # Fitted StandardScaler 

├── updated_applicant_data.csv  # Sample applicant data for predictions 

├── requirements.txt  # Python dependencies 

└── README.md \



---

## 🧠 How it Works

1. **Train Model (Cached)**  
   - Cleans raw LendingClub data  
   - Adds new features: `past_defaults`, `monthly_salary`, `available_balance`  
   - Encodes categorical fields (grade, term, home ownership)  
   - Scales numerical features and fits an XGBoost classifier  
   - Saves model and scaler for reuse

2. **Applicant Prediction Flow**  
   - User uploads a CSV file containing new applicant details  
   - Features are derived, encoded, and scaled to match training format  
   - Model predicts the probability of loan approval  
   - SHAP values explain the prediction using waterfall plots

<img width="960" alt="Loan-1" src="https://github.com/user-attachments/assets/db9c6a0e-c190-49e5-8a17-e270744f16ea" />

---

## 📦 Installation & Running

### 🔧 Prerequisites
- Python 3.8+
- pip
- Recommended: virtualenv

### 📥 Install Dependencies

```bash
pip install -r requirements.txt
▶️ Run the App
bash

streamlit run loan_approval_app.py


📈 SHAP Visualization
Understand which features influenced the model decision with an interactive SHAP waterfall plot.

<img width="638" alt="Loan-2" src="https://github.com/user-attachments/assets/ba835e1a-a172-4059-a6d8-e337b3d12df3" />


Example:

High income ✅

Low credit utilization ✅

High DTI ❌
→ Final Score = 0.67
→ Loan Approved (Threshold: 0.60)
<img width="960" alt="Loan-3" src="https://github.com/user-attachments/assets/6d920a8e-5fb0-4f0c-b7dd-52187cbe7a3d" />

<img width="960" alt="Loan-4" src="https://github.com/user-attachments/assets/6a26c935-3ca3-4221-aca6-439dc41e9bb0" />

<img width="960" alt="Loan-5" src="https://github.com/user-attachments/assets/645b58c0-fd2d-48ce-9db1-b689c3d90e5e" />


📧 Optional Add-ons (Coming Soon)
✅ Batch Prediction Support

📊 Dashboard for Loan Statistics

📤 Email Prediction Result

📁 Export SHAP Explanation Report as PDF

✨ Tech Stack
Python

XGBoost

SHAP

Pandas / NumPy / scikit-learn

Streamlit (Web Interface)

Joblib (Model Persistence)

📚 Inspiration
This project is inspired by real-world credit risk evaluation systems used by banks and fintechs to automate and explain loan decisions with fairness and transparency.

📜 License
MIT License

🤝 Contributing
Contributions are welcome! Feel free to open issues or PRs for:

More robust SHAP visualizations

Additional preprocessing for real-world data

UI improvements for better UX

🙋‍♂️ Author
Soumya Sankar 
Data Engineer | ML Enthusiast | Data Engineer

LinkedIn : https://www.linkedin.com/in/soumyasankar99/

GitHub : https://github.com/soumyasankar99

⭐ If you like this project, don’t forget to Star the repo and share!
