import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# Load model and expected columns
# ---------------------------
model_bundle = joblib.load("student_model.pkl")
model = model_bundle["model"]
expected_columns = model_bundle["columns"]

st.set_page_config(page_title="🎓 Student Performance Predictor", layout="centered")
st.title("🎓 Student Pass/Fail Predictor")
st.markdown("Provide the student's academic behavior profile to predict Pass or Fail.")

# ---------------------------
# Input Form
# ---------------------------
with st.form("student_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 10, 30, 18)
        gender = st.selectbox("Gender", ["Male", "Female"])
        study_hours = st.slider("Study Hours/Week", 0, 40, 10)
        online_courses = st.slider("Online Courses Completed", 0, 20, 2)
        participation = st.selectbox("Participation in Discussions", ["Low", "Medium", "High"])
        assignment_completion = st.slider("Assignment Completion Rate (%)", 0, 100, 85)

    with col2:
        exam_score = st.slider("Exam Score (%)", 0, 100, 70)
        attendance = st.slider("Attendance Rate (%)", 0, 100, 80)
        tech_use = st.selectbox("Use of Educational Tech", ["Yes", "No"])
        stress_level = st.selectbox("Self-Reported Stress Level", ["Low", "Medium", "High"])
        social_media = st.slider("Time on Social Media (hrs/week)", 0, 50, 10)
        sleep_hours = st.slider("Sleep Hours per Night", 0, 12, 7)

    submitted = st.form_submit_button("📊 Predict")

# ---------------------------
# Manual encoding (based on expected training columns)
# ---------------------------
user_data = {
    "Age": age,
    "Study_Hours_per_Week": study_hours,
    "Online_Courses_Completed": online_courses,
    "Assignment_Completion_Rate (%)": assignment_completion,
    "Exam_Score (%)": exam_score,
    "Attendance_Rate (%)": attendance,
    "Time_Spent_on_Social_Media (hours/week)": social_media,
    "Sleep_Hours_per_Night": sleep_hours,

    # One-hot encoding
    "Gender_Male": 1 if gender == "Male" else 0,
    "Gender_Female": 1 if gender == "Female" else 0,

    f"Participation_in_Discussions_{participation}": 1,
    f"Self_Reported_Stress_Level_{stress_level}": 1,
    f"Use_of_Educational_Tech_{tech_use}": 1
}

# ---------------------------
# Prepare model input DataFrame
# ---------------------------
input_df = pd.DataFrame(columns=expected_columns)
input_df.loc[0] = 0

for col, val in user_data.items():
    if col in input_df.columns:
        input_df.at[0, col] = val

input_df = input_df.drop(columns=input_df.filter(regex="^Student_ID_").columns, errors='ignore')

# ---------------------------
# Prediction
# ---------------------------
if submitted:
    st.write("🔍 Input Sent to Model:", input_df)
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🎯 Prediction: {'✅ Pass' if prediction == 1 else '❌ Fail'}")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")