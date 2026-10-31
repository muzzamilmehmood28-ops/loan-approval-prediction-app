# loan-approval-prediction-app
# 🏦 Loan Approval Prediction Web Application

An end-to-end Machine Learning web application designed to predict whether a loan application will be approved or rejected based on applicant financial profiles, credit history, and employment details. 

Built as part of the **Devixo Solutions AI/ML Internship Task**.

🚀 **Live Interactive App:** [Click Here to Access the Live Streamlit App](https://loan-approval-prediction-app-sth5lpztnd67ecnbuenfuq.streamlit.app/)

---

## 📌 Project Overview

Loan approval decision-making requires evaluating complex applicant attributes while minimizing financial default risk. This project automates and streamlines that evaluation process by training, tuning, and deploying a high-accuracy classification model wrapped in an interactive multi-page web interface.

### Key Features
- **Exploratory Data Analysis (EDA):** Insights into distribution of income, loan amounts, and credit history impact.
- **Automated Data Preprocessing:** Preprocessing pipelines handling missing value imputation, categorical encoding, and feature scaling.
- **Model Evaluation & Tuning:** Compared 4 baseline machine learning algorithms and optimized the best model using `GridSearchCV`.
- **Interactive Web Interface:** Streamlit app allowing users to input applicant details and receive real-time approval predictions with confidence scores.

---

## 🛠️ Tech Stack & Tools

- **Programming Language:** Python 3.10+
- **Machine Learning & Data Science:** Pandas, NumPy, Scikit-Learn, Joblib
- **Web Framework:** Streamlit
- **Development Environment:** VS Code, Git, GitHub
- **Deployment Platform:** Streamlit Community Cloud

---

## 📊 Model Building & Performance Evaluation

We evaluated four different classification models to find the optimal balance between accuracy, precision, and recall for risk assessment:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier** *(Selected & Fine-Tuned)*
4. **Gradient Boosting / XGBoost Classifier**

### Hyperparameter Tuning
Using `GridSearchCV`, the **Random Forest Classifier** was fine-tuned over hyperparameter grids (tree depth, number of estimators, min samples split). The best-performing estimator was saved using `joblib` into `models/best_model.pkl` for deployment.

---

## 📁 Repository Structure

```text
loan-approval-prediction-app/
│
├── data/
│   └── loan_data.csv          # Raw and processed dataset files
├── models/
│   └── best_model.pkl         # Saved






🚀 How to Run Locally
1. Clone the Repository
git clone [https://github.com/muzzamilmehmood28-ops/loan-approval-prediction-app.git](https://github.com/muzzamilmehmood28-ops/loan-approval-prediction-app.git)
cd loan-approval-prediction-app


2. Set Up Virtual Environment & Install Dependencies
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt




3. Run the Model Training Pipeline (Optional)
Bash
python loanproject.py


4. Launch the Streamlit App
Bash
streamlit run app.py


