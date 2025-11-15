import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="❤️ Heart Disease Prediction", layout="centered")

# Load your trained files
model = joblib.load("logisticRegression_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below to predict the risk of heart disease.")

# ---------------- UI FORM -------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, 45)
        resting_bp = st.number_input("Resting BP", 50, 250, 120)
        cholesterol = st.number_input("Cholesterol", 50, 600, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", [0, 1])

    with col2:
        max_hr = st.number_input("Max Heart Rate", 50, 250, 150)
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)

        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    submitted = st.form_submit_button("Predict ❤️")

# ---------------- Processing -------------------
if submitted:
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    # Show Result
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
