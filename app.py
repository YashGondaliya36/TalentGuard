# app.py
import streamlit as st
import pandas as pd
from src.TalentGuard.pipeline.prediction_pipeline import EmployeeRetentionRiskPredictor

# Initialize the prediction pipeline
pipeline = EmployeeRetentionRiskPredictor()

# Streamlit app title
st.title("Employee Retention Risk Prediction")

# Input fields for user input based on the given features
st.header("Input Data")

# Collecting input values
satisfaction_level = st.number_input("Satisfaction Level (e.g., 0.1 to 1.0)", min_value=0.0, max_value=1.0)
last_evaluation = st.number_input("Last Evaluation (e.g., 0.1 to 1.0)", min_value=0.0, max_value=1.0)
number_project = st.number_input("Number of Projects (e.g., 1 to 10)", min_value=1, step=1)
average_montly_hours = st.number_input("Average Monthly Hours (e.g., 100 to 400)", min_value=100, max_value=400, step=1,)
time_spend_company = st.number_input("Time Spent at Company (Years, e.g., 1 to 20)", min_value=1, max_value=20, step=1)
work_accident = st.selectbox("Work Accident (0 = No, 1 = Yes)", options=[0, 1])
promotion_last_5years = st.selectbox("Promotion in Last 5 Years (0 = No, 1 = Yes)", options=[0, 1])
Department = st.selectbox("Department", options=['sales', 'technical', 'support', 'IT', 'product_mng', 'marketing',
                                                  'RandD', 'accounting', 'hr', 'management'])
salary = st.selectbox("Salary Level", options=["low", "medium", "high"])

# Create a DataFrame for the input data
input_data = pd.DataFrame({
    "satisfaction_level": [satisfaction_level],
    "last_evaluation": [last_evaluation],
    "number_project": [number_project],
    "average_montly_hours": [average_montly_hours],
    "time_spend_company": [time_spend_company],
    "Work_accident": [work_accident],
    "promotion_last_5years": [promotion_last_5years],
    "Department": [Department],
    "salary": [salary]
})

# Button to make predictions
if st.button("Predict"):
    prediction = pipeline.predict(input_data)
    if prediction == 1:
        st.warning("Prediction: Employee is likely to leave the job.")
    else:
        st.success("Prediction: Employee is likely to stay in the job.")
