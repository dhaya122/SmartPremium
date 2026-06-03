import os
import streamlit as st
import joblib
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "model.pkl")
preprocessor_path = os.path.join(BASE_DIR, "models", "preprocessor.pkl")

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)


st.title("💰 SmartPremium Insurance Predictor")

id_val = st.number_input("Customer ID", min_value=1, value=1)

age = st.number_input("Age", min_value=18, max_value=100, value=30)

gender = st.selectbox("Gender", ["Female", "Male"])

income = st.number_input("Annual Income", min_value=0.0, value=50000.0)

marital_status = st.selectbox(
    "Marital Status",
    ["Married", "Divorced", "Single"]
)

dependents = st.number_input(
    "Number of Dependents",
    min_value=0,
    value=0
)

education = st.selectbox(
    "Education Level",
    ["Bachelor's", "Master's", "High School", "PhD"]
)

occupation = st.selectbox(
    "Occupation",
    ["Self-Employed", "Employed", "Unemployed"]
)

health_score = st.number_input(
    "Health Score",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

location = st.selectbox(
    "Location",
    ["Urban", "Rural", "Suburban"]
)

policy_type = st.selectbox(
    "Policy Type",
    ["Premium", "Comprehensive", "Basic"]
)

previous_claims = st.number_input(
    "Previous Claims",
    min_value=0,
    value=0
)

vehicle_age = st.number_input(
    "Vehicle Age",
    min_value=0,
    value=5
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=700
)

insurance_duration = st.number_input(
    "Insurance Duration (Years)",
    min_value=1,
    value=3
)

policy_start_date = st.text_input(
    "Policy Start Date",
    "2023-01-01"
)

customer_feedback = st.selectbox(
    "Customer Feedback",
    ["Poor", "Average", "Good"]
)

smoking_status = st.selectbox(
    "Smoking Status",
    ["No", "Yes"]
)

exercise_frequency = st.selectbox(
    "Exercise Frequency",
    ["Weekly", "Monthly", "Daily", "Rarely"]
)

property_type = st.selectbox(
    "Property Type",
    ["House", "Apartment", "Condo"]
)


if st.button("Predict Premium"):

    input_data = {
        "id": id_val,
        "Age": age,
        "Gender": gender,
        "Annual Income": income,
        "Marital Status": marital_status,
        "Number of Dependents": dependents,
        "Education Level": education,
        "Occupation": occupation,
        "Health Score": health_score,
        "Location": location,
        "Policy Type": policy_type,
        "Previous Claims": previous_claims,
        "Vehicle Age": vehicle_age,
        "Credit Score": credit_score,
        "Insurance Duration": insurance_duration,
        "Policy Start Date": policy_start_date,
        "Customer Feedback": customer_feedback,
        "Smoking Status": smoking_status,
        "Exercise Frequency": exercise_frequency,
        "Property Type": property_type,
    }

    df = pd.DataFrame([input_data])

    processed = preprocessor.transform(df)

    prediction = model.predict(processed)[0]

    st.success(f"Predicted Premium: ₹ {prediction:.2f}")