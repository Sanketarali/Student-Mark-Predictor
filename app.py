import streamlit as st
import joblib

model = joblib.load('model_joblib_test')

st.title("Student Mark Prediction")

number_courses = st.number_input("number_courses")
time_study= st.number_input(" time_study")



if st.button("Predict"):
   
   
    result = model.predict([[number_courses ,time_study]])
    st.write("Marks:" , result[0])
