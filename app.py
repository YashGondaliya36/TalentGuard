import streamlit as st
import pandas as pd
from src.TalentGuard.pipeline.prediction_pipeline import EmployeeRetentionRiskPredictor

# Configure the page
st.set_page_config(
    page_title="TalentGuard - Employee Retention Prediction",
    page_icon="👥",
    layout='centered'
)

# Initialize the prediction pipeline
pipeline = EmployeeRetentionRiskPredictor()

# Create a sidebar for navigation
page = st.sidebar.selectbox(
    "Navigate to", 
    ["TelentGuard", "Risk Prediction", "Key Insights"]
)

# Introduction Page
if page == "TelentGuard":
    st.header("🎯 TalentGuard - Employee Retention Prediction",divider='rainbow')
    

    st.markdown("""
    ### Why Employee Retention Matters?
    Employee turnover is a critical challenge that organizations face today. It can:
    - 🏢 Disrupt team dynamics and productivity
    - 📉 Impact company culture and morale
    
    ### What TalentGuard Does?
    TalentGuard uses machine learning to:
    - **🔍 Predict** which employees are at risk of leaving
    - **💡 Highlight** critical factors influencing employee departures, based on exploratory data analysis (EDA).
    
    ### How It Works?
    Our model analyzes various factors including:
    - 📊 Performance evaluations
    - ⏰ Working hours
    - 💼 Project involvement
    - 📈 Satisfaction levels
    - 💵 Compensation
    """)


# Prediction Page
elif page == "Risk Prediction":
    st.header("🔮 Employee Retention Prediction",divider='rainbow')
    
    col1, col2 = st.columns(2)
    
    with col1:
        satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.1)
        last_evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.1)
        number_project = st.number_input("Number of Projects",min_value=0)
        average_montly_hours = st.number_input("Average Monthly Hours",min_value=0)
        time_spend_company = st.number_input("Years at Company",min_value= 0,step=3)
    
    with col2:
        Work_accident = st.selectbox("Work Accident", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        promotion_last_5years = st.selectbox("Promoted in Last 5 Years", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        Department = st.selectbox("Department", ['sales', 'technical', 'support', 'IT', 'product_mng', 'marketing',
                                               'RandD', 'accounting', 'hr', 'management'])
        salary = st.selectbox("Salary Level", ["low", "medium", "high"])

    # Create input data
    input_data = pd.DataFrame({
        "satisfaction_level": [satisfaction_level],
        "last_evaluation": [last_evaluation],
        "number_project": [number_project],
        "average_montly_hours": [average_montly_hours],
        "time_spend_company": [time_spend_company],
        "Work_accident": [Work_accident],
        "promotion_last_5years": [promotion_last_5years],
        "Department": [Department],
        "salary": [salary]
    })

    if st.button("Predict", type="primary"):
        prediction = pipeline.predict(input_data)
        
        if prediction == 1:
            st.error("⚠️ High Risk: Employee is likely to leave")
        else:
            st.success("✅ Low Risk: Employee is likely to stay")

# Insights Page
elif page == "Key Insights":
    st.header("📊 Key Insights on Employee Retention",divider='green')
    
    st.markdown("""
    ### 🎯 Key Factors Affecting Employee Retention
    
    #### 1. Workload Balance
    - 🔴 **Underworked** employees (<150 hours/month) tend to leave
    - 🔴 **Overworked** employees (>250 hours/month) tend to leave
    
    #### 2. Performance Evaluation
    - ⚠️ Both **extremely high and low** Performance emplyees tend to leave
    - ✅ Balanced evaluations show better retention
    
    #### 3. Salary Impact
    - 💰 **Low to medium** salary brackets show high risk to leave job
    - 💼 Competitive compensation is crucial for retention
    
    #### 4. Project Engagement
    - 📊 Employees with **2, 6, or 7** projects show highest risk
    - 🎯 Optimal project load: 3-4 projects
    
    #### 5. Tenure Considerations
    - ⏳ **4-5 years** at company is a critical period
    - 🔄 Career growth opportunities crucial at this stage
    
    ### 📈 Top 3 Predictive Factors
    1. **Employee Satisfaction** (Strongest indicator)
    2. **Years at Company**
    3. **Performance Evaluation**
    """)

# Add a footer
st.sidebar.markdown("---")
st.sidebar.markdown("[Made with ❤️ by Yash Gondaliya](https://www.linkedin.com/in/yash-gondaliya-02427a260)")
