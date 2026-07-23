import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

if "login" not in st.session_state:
    st.session_state.login = False

st.set_page_config(
    page_title="Smart Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)

model = joblib.load("employee_attrition_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

df = pd.read_csv("employee_attrition.csv")
history_file = "prediction_history.csv"

if not os.path.exists(history_file):
    history = pd.DataFrame(
        columns=[
            "Date",
            "Time",
            "Prediction",
            "Confidence"
        ]
    )
    history.to_csv(history_file, index=False)

st.title("🏢 Smart Employee Attrition Prediction")
st.write("Machine Learning based Employee Attrition Prediction System")

# ---------------- LOGIN ----------------

if not st.session_state.login:

    st.title("🔐 HR Login Portal")

    username = st.text_input("Username")

    password = st.text_input(
        "Password",
        type="password"
    )

    login = st.button("Login")

    if login:

        if username == "Rani" and password == "rani123":

            st.session_state.login = True
            st.rerun()

        else:

            st.error("Invalid Username or Password")

    st.stop()

page = st.sidebar.selectbox(
    "📌 Menu",
    [
        "🏠 Home",
        "📊 Dashboard",
        "🤖 Prediction",
        "📂 Upload CSV",
        "📜 Prediction History",
        "⚙ Settings",
        "ℹ About"
    ]
)

if st.sidebar.button("🚪 Logout"):

    st.session_state.login = False

    st.rerun()

# ---------------- HOME ----------------
if page == "🏠 Home":

    st.header("Welcome")
    st.write("This project predicts Employee Attrition.")
    st.write("Algorithms Used")
    st.write("""
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - KNN
    - Gradient Boosting
    """)

# ---------------- DASHBOARD ----------------
elif page == "📊 Dashboard":

    st.header("📊 Employee Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Employees", len(df))
    with col2:
        st.metric("Total Features", len(df.columns))

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Attrition Count")
    st.bar_chart(df["Attrition"].value_counts())

    st.subheader("Department Distribution")
    st.bar_chart(df["Department"].value_counts())

    st.subheader("Gender Distribution")
    st.bar_chart(df["Gender"].value_counts())

    st.subheader("OverTime Distribution")
    st.bar_chart(df["OverTime"].value_counts())

# ---------------- PREDICTION ----------------
elif page == "🤖 Prediction":

    st.header("Employee Attrition Prediction")

    user_input = {}

    feature_list = [
        'Age', 'BusinessTravel', 'DailyRate', 'Department',
        'DistanceFromHome', 'Education', 'EducationField',
        'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
        'JobInvolvement', 'JobLevel', 'JobRole',
        'JobSatisfaction', 'MaritalStatus',
        'MonthlyIncome', 'MonthlyRate',
        'NumCompaniesWorked', 'OverTime',
        'PercentSalaryHike', 'PerformanceRating',
        'RelationshipSatisfaction',
        'StockOptionLevel',
        'TotalWorkingYears',
        'TrainingTimesLastYear',
        'WorkLifeBalance',
        'YearsAtCompany',
        'YearsInCurrentRole',
        'YearsSinceLastPromotion',
        'YearsWithCurrManager'
    ]

    for feature in feature_list:
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[feature]):
            user_input[feature] = st.number_input(
                feature,
                value=float(df[feature].median())
            )
        else:
            user_input[feature] = st.selectbox(
                feature,
                sorted(df[feature].dropna().unique())
            )

    predict = st.button("Predict Employee")

    if predict:

        # keep exact training column order
        input_df = pd.DataFrame([user_input], columns=feature_list)

        for col in input_df.columns:
            if col in encoders:
                input_df[col] = encoders[col].transform(input_df[col])

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)

        st.divider()

        if prediction[0] == 1:
            st.error("⚠ Employee is likely to leave the company.")
        else:
            st.success("✅ Employee is likely to stay in the company.")

        st.subheader("Prediction Confidence")
        st.progress(float(np.max(probability)))
        st.write(f"Confidence : {np.max(probability) * 100:.2f}%")
        st.subheader("Prediction Summary")

        result = "Leave" if prediction[0] == 1 else "Stay"

        summary = pd.DataFrame({
            "Prediction": [result],
            "Confidence (%)": [round(np.max(probability) * 100, 2)]
        })
        # Save Prediction History
        now = datetime.now()

        history = pd.read_csv(history_file)

        new_record = pd.DataFrame({
            "Date": [now.strftime("%d-%m-%Y")],
            "Time": [now.strftime("%I:%M %p")],
            "Prediction": [result],
            "Confidence": [round(np.max(probability) * 100, 2)]
        })

        history = pd.concat([history, new_record], ignore_index=True)

        history.to_csv(history_file, index=False)

        st.dataframe(summary)

        csv = summary.to_csv(index=False)

        st.download_button(
            label="📥 Download Report",
            data=csv,
            file_name="prediction.csv",
            mime="text/csv"
        )

elif page == "📂 Upload CSV":

    st.header("📂 Batch Employee Prediction")

    uploaded_file = st.file_uploader(
        "Upload Employee CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        upload_df = pd.read_csv(uploaded_file)

        st.success("File Uploaded Successfully")

        st.dataframe(upload_df.head())

elif page == "📜 Prediction History":

    st.header("📜 Prediction History")

    history = pd.read_csv(history_file)

    if history.empty:
        st.info("No Prediction History Available")
    else:
        st.dataframe(history)

        csv = history.to_csv(index=False)

        st.download_button(
            label="📥 Download History",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )

elif page == "⚙ Settings":

    st.header("Settings")

    st.write("Theme : Light")

    st.write("Version : 1.0")   
# ---------------- ABOUT ----------------
elif page == "ℹ About":

    st.header("About Project")
    st.write("""
### Smart Employee Attrition Prediction

This project predicts whether an employee is likely to leave the company using Machine Learning.

### Dataset
IBM HR Analytics Employee Attrition Dataset

### Algorithms Used
- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- Gradient Boosting

### Best Model
Random Forest Classifier

### Technologies
- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Joblib
""")
