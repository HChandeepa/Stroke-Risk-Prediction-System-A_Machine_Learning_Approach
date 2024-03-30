import streamlit as st
import pickle
import pandas as pd
import numpy as np
from PIL import Image

# Load the pre-trained model
model = pickle.load(open('model.sav', 'rb'))

# Title and sidebar header
st.title('Stroke Prediction')
st.sidebar.header('Enter Your Data')
image = Image.open('stroke.jpg')
st.image(image, '')

# Function to gather user input
def user_report():
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female', 'Other'])
    age = st.sidebar.number_input('Age', min_value=0, step=1)
    hypertension = st.sidebar.selectbox('Hypertension', ['Yes', 'No'])
    heart_disease = st.sidebar.selectbox('Heart Disease', ['Yes', 'No'])
    ever_married = st.sidebar.selectbox('Ever Married', ['Yes', 'No'])
    work_type = st.sidebar.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'Children', 'Never_worked'])
    bmi = st.sidebar.number_input('BMI', min_value=0.0, step=0.1)
    smoking_status = st.sidebar.selectbox('Smoking Status', ['Unknown', 'Never smoked', 'formerly smoked', 'Smokes'])


    # Map categorical variables to numerical values
    gender_mapping = {'Male': 0, 'Female': 1, 'Other': 2}
    ever_married_mapping = {'Yes': 0, 'No': 1}
    work_type_mapping = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'Children': 3, 'Never_worked': 4}
    smoking_status_mapping = {'Unknown': 3, 'Never smoked': 1, 'formerly smoked': 0, 'Smokes': 2}

    user_report_data = {
        'hypertension': 1 if hypertension == 'Yes' else 0,
        'heart_disease': 1 if heart_disease == 'Yes' else 0,
        'ever_married': ever_married_mapping[ever_married],
        'work_type': work_type_mapping[work_type],
        'bmi': bmi,
        'smoking_status': smoking_status_mapping[smoking_status],
        'age': age,
        'gender': gender_mapping[gender],
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# Display user data
user_data = user_report()
st.header('Patient Data')
st.write(user_data)

# Predict stroke risk
if st.button('Predict Stroke Risk'):
    try:
        stroke_prediction = model.predict(user_data)
        probability = model.predict_proba(user_data)[:,1]
        if stroke_prediction == 1:
            st.subheader('Patient is at risk of stroke with a probability of {:.2f}%'.format(probability[0] * 100))
        else:
            st.subheader('Patient is not at risk of stroke with a probability of {:.2f}%'.format((1 - probability[0]) * 100))
    except Exception as e:
        st.error('An error occurred during prediction. Please check your input data and try again.')
        st.error(str(e))
