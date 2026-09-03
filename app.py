import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("Cardio Predictor - Heart Disease Prediction")

age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex (1=Male, 0=Female)", [0, 1])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting BP", 90, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 (1=Yes, 0=No)", [0, 1])
restecg = st.selectbox("Rest ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", 70, 210, 150)
exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("CA (No. of vessels)", [0, 1, 2, 3])
thal = st.selectbox("Thal", [0, 1, 2, 3])

if st.button("Predict"):
    data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    pred = model.predict(data)

    if pred[0] == 1:
        st.error("⚠️ High Risk of Heart Disease detected!")
    else:
        st.success("✅ Low Risk - Heart is Healthy!")
