import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

st.set_page_config(page_title="Cardiovascular Disease Prediction")
st.title("❤️ Cardiovascular Disease Prediction")
st.write("Enter patient details to predict heart disease")

# Inputs - standard heart dataset features
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 20, 90, 50)
    sex = st.selectbox("Sex (0=Female, 1=Male)", [0,1])
    cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
    trestbps = st.number_input("Resting BP", 90, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar >120 (0=No,1=Yes)", [0,1])

with col2:
    restecg = st.selectbox("Resting ECG (0-2)", [0,1,2])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina (0=No,1=Yes)", [0,1])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope (0-2)", [0,1,2])
    ca = st.selectbox("CA vessels (0-4)", [0,1,2,3,4])
    thal = st.selectbox("Thal (0-3)", [0,1,2,3])

if st.button("Predict"):
    data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)

    if pred[0] == 1:
        st.error("⚠️ High Risk of Heart Disease detected!")
    else:
        st.success("✅ Low Risk - No Heart Disease")
        st.balloons()
