import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor | EdTech ML Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #1E88E5;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        border-radius: 8px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #1565C0;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_student_model.joblib")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

pipeline = load_model()

# Header Section
st.markdown("<div class='main-header'>🎓 Student Performance Prediction ML Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>ML Bubble 2026 | Army Institute of Technology (AIT), Pune | TE-BE Advanced Track</div>", unsafe_allow_html=True)

if pipeline is None:
    st.error("⚠️ Model file `best_student_model.joblib` not found! Please run `python main.py` first to train and save the model.")
    st.stop()

# Sidebar for Input Controls
st.sidebar.header("📋 Student Profile Inputs")

st.sidebar.subheader("1. Academic History")
g1 = st.sidebar.slider("Period 1 Marks (G1: 0-20)", 0, 20, 12)
g2 = st.sidebar.slider("Period 2 Marks (G2: 0-20)", 0, 20, 13)
failures = st.sidebar.selectbox("Past Class Failures", [0, 1, 2, 3])

st.sidebar.subheader("2. Study Habits & Attendance")
studytime = st.sidebar.selectbox("Weekly Study Time", options=[1, 2, 3, 4], 
                                 format_func=lambda x: {1: "< 2 hours", 2: "2 - 5 hours", 3: "5 - 10 hours", 4: "> 10 hours"}[x], index=1)
absences = st.sidebar.slider("School Absences", 0, 50, 4)
schoolsup = st.sidebar.selectbox("Extra Educational Support", ["yes", "no"], index=1)
famsup = st.sidebar.selectbox("Family Educational Support", ["yes", "no"], index=0)
paid = st.sidebar.selectbox("Extra Paid Classes", ["yes", "no"], index=1)
higher = st.sidebar.selectbox("Wants Higher Education", ["yes", "no"], index=0)
internet = st.sidebar.selectbox("Internet Access at Home", ["yes", "no"], index=0)

st.sidebar.subheader("3. Demographic & Socio-Economic")
sex = st.sidebar.selectbox("Sex", ["F", "M"])
age = st.sidebar.slider("Age", 15, 22, 17)
address = st.sidebar.selectbox("Address Type", options=["U", "R"], format_func=lambda x: "Urban" if x == "U" else "Rural")
famsize = st.sidebar.selectbox("Family Size", options=["LE3", "GT3"], format_func=lambda x: "≤ 3 members" if x == "LE3" else "> 3 members")
pstatus = st.sidebar.selectbox("Parents Cohabitation Status", options=["T", "A"], format_func=lambda x: "Together" if x == "T" else "Apart")
medu = st.sidebar.selectbox("Mother Education", [0, 1, 2, 3, 4], format_func=lambda x: ["None", "Primary", "5th-9th", "Secondary", "Higher"][x], index=2)
fedu = st.sidebar.selectbox("Father Education", [0, 1, 2, 3, 4], format_func=lambda x: ["None", "Primary", "5th-9th", "Secondary", "Higher"][x], index=2)
mjob = st.sidebar.selectbox("Mother Job", ["teacher", "health", "services", "at_home", "other"])
fjob = st.sidebar.selectbox("Father Job", ["teacher", "health", "services", "at_home", "other"])
reason = st.sidebar.selectbox("Reason for School Choice", ["home", "reputation", "course", "other"])
guardian = st.sidebar.selectbox("Guardian", ["mother", "father", "other"])
traveltime = st.sidebar.selectbox("Travel Time to School", [1, 2, 3, 4], format_func=lambda x: {1: "< 15 min", 2: "15-30 min", 3: "30-60 min", 4: "> 60 min"}[x])
activities = st.sidebar.selectbox("Extra-curricular Activities", ["yes", "no"])
nursery = st.sidebar.selectbox("Attended Nursery School", ["yes", "no"])
romantic = st.sidebar.selectbox("In Romantic Relationship", ["yes", "no"])
famrel = st.sidebar.slider("Family Relationship Quality (1-5)", 1, 5, 4)
freetime = st.sidebar.slider("Free Time After School (1-5)", 1, 5, 3)
goout = st.sidebar.slider("Going Out with Friends (1-5)", 1, 5, 3)
dalc = st.sidebar.slider("Workday Alcohol (1-5)", 1, 5, 1)
walc = st.sidebar.slider("Weekend Alcohol (1-5)", 1, 5, 1)
health = st.sidebar.slider("Current Health Status (1-5)", 1, 5, 4)

# Build Input Dataframe
input_data = pd.DataFrame([{
    'school': 'GP', 'sex': sex, 'age': age, 'address': address,
    'famsize': famsize, 'Pstatus': pstatus, 'Medu': medu, 'Fedu': fedu,
    'Mjob': mjob, 'Fjob': fjob, 'reason': reason, 'guardian': guardian,
    'traveltime': traveltime, 'studytime': studytime, 'failures': failures,
    'schoolsup': schoolsup, 'famsup': famsup, 'paid': paid,
    'activities': activities, 'nursery': nursery, 'higher': higher,
    'internet': internet, 'romantic': romantic, 'famrel': famrel,
    'freetime': freetime, 'goout': goout, 'Dalc': dalc, 'Walc': walc,
    'health': health, 'absences': absences, 'G1': g1, 'G2': g2
}])

# Main Panel Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📌 Selected Student Profile Summary")
    summary_df = pd.DataFrame({
        "Attribute": ["G1 Score", "G2 Score", "Weekly Study Time", "Absences", "Past Failures", "Mother Edu", "Father Edu", "Internet Access"],
        "Value": [f"{g1} / 20", f"{g2} / 20", f"{studytime} Level", f"{absences} days", f"{failures}", f"Level {medu}", f"Level {fedu}", internet.upper()]
    })
    st.table(summary_df)

with col2:
    st.subheader("🎯 Real-Time Predictive Assessment")
    
    if st.button("🔮 Predict Final Grade (G3)"):
        prediction = pipeline.predict(input_data)[0]
        predicted_score = float(np.clip(prediction, 0, 20))
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric(label="Predicted Final Grade (G3 Score)", value=f"{predicted_score:.2f} / 20")
        
        # Risk Badge Classification
        if predicted_score >= 14:
            st.success("🟢 **LOW ACADEMIC RISK**: Student exhibits strong performance trajectory. High probability of academic distinction!")
        elif predicted_score >= 10:
            st.warning("🟡 **MODERATE ACADEMIC RISK**: Student exhibits average performance. Periodic academic monitoring recommended.")
        else:
            st.error("🔴 **HIGH ACADEMIC RISK**: Student at risk of failure! Immediate academic intervention & counseling required.")
        
        st.progress(predicted_score / 20.0)

st.markdown("---")
st.subheader("📊 Model Comparative Performance Summary")

benchmarks_df = pd.DataFrame({
    "Model Architecture": ["Linear Regression (Selected)", "Gradient Boosting Regressor", "Random Forest Regressor"],
    "MAE (Mean Absolute Error)": [0.7672, 0.8235, 0.8250],
    "RMSE (Root Mean Squared Error)": [0.9674, 1.0318, 1.0395],
    "R² Score (Accuracy Fit)": ["95.22%", "94.57%", "94.48%"],
    "Deployment Status": ["★ Best Model", "Candidate", "Candidate"]
})
st.table(benchmarks_df)

st.caption("Developed for ML Bubble 2026 | Army Institute of Technology (AIT), Pune | Track: TE-BE Advanced")
