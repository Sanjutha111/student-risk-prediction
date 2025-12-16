import streamlit as st
import requests

st.title("Student Risk Prediction")

studytime = st.selectbox("Study Time (in hours)", [1, 2, 3, 4,5,6,7,8])
absences = st.number_input("Absences", min_value=0)
failures = st.selectbox("Past Failures", [0, 1, 2, 3,4,5])

if st.button("Predict"):
    payload = {
        "studytime": studytime,
        "absences": absences,
        "failures": failures
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=payload
    )

    result = response.json()

    if result["at_risk"] == 1:
        st.error("Student is at risk")
    else:
        st.success("Student is not at risk")
