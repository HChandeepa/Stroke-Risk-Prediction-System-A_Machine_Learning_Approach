import streamlit as st
import pandas as pd
import pickle

DATASET_PATH = "stroke data.csv"
LOG_MODEL_PATH = "model.pkl"

def main():
    @st.cache_data(persist=True)
    def load_dataset() -> pd.DataFrame:
        stroke_df = pd.read_csv(DATASET_PATH)
        return stroke_df
    
    def user_input_features() -> pd.DataFrame:
        gender = st.sidebar.selectbox('Gender',options=["Male", "Female", "Other"])
        age = st.sidebar.number_input('Enter your Age', 0, 100, 0)
        hypertension = st.sidebar.selectbox('Hypertension', options=["No", "Yes"])
        heart_disease = st.sidebar.selectbox('Heart Disease', options=["No", "Yes"])
        ever_married = st.sidebar.selectbox('Ever Married', options=["No", "Yes"])
        work_type = st.sidebar.selectbox('Work Type', options=["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        Residence_type = st.sidebar.selectbox('Residence Type',options=["Urban","Rural"])
        avg_glucose_level = st.sidebar.number_input('Enter Glucose Level',0.0,300.0,0.0)
        bmi = st.sidebar.number_input('BMI (weight/height²)', 0.0, 100.0, 0.0, step=0.1)
        smoking_status = st.sidebar.selectbox('Smoking Status', options=["formerly smoked", "never smoked", "smokes", "Unknown"])


        features = pd.DataFrame({
            "age": [age],
            "hypertension": [1 if hypertension == "Yes" else 0],
            "ever_married": [1 if ever_married == "Yes" else 0],
            "work_type": [0 if work_type == "Private" else 1 if work_type == "Self-employed" else 2 if work_type == "Govt_job" else 3 if work_type == "children" else 4],
            "heart_disease": [1 if heart_disease == "Yes" else 0],
            "smoking_status": [0 if smoking_status == "formerly smoked" else 1 if smoking_status == "never smoked" else 2 if smoking_status == "smokes" else 3],
            "bmi": [bmi],
            "gender": [0 if gender == "Male" else 1 if gender == "Female" else 2],
            "Residence_type": [0 if Residence_type == "Urban" else 1],
            "avg_glucose_level": [avg_glucose_level]
        })

        return features
    
    st.set_page_config(
        page_title="Stroke Risk Prediction App",
        page_icon="images/stroke.jpg"
    )

    st.title("Stroke Risk Prediction")
    st.subheader("Are you concerned about the state of your brain? "
"This app will assist you in diagnosing it!")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("images/doctor.png",
                 caption="I'll assist you in diagnosing your risk of stroke! - Dr. Logistic Regression",
                 width=150)
        submit = st.button("Predict")
    with col2:
        st.markdown("""
        Did you know that machine learning models can help you predict the likelihood of experiencing a stroke pretty 
                    accurately? In this app, you can estimate your chance of having a stroke (yes/no) in seconds!
        
        Here, a logistic regression model using an advanced technique
        was constructed using survey data of over 5k individuals in the year 2022.
        This application is based on it because it has demonstrated superior performance, 
        achieving an impressive accuracy of 95%.
        
        To predict your stroke risk, simply follow these steps:        
        1. Enter the parameters that best describe you.
        2. Press the "Predict" button and wait for the result.
            
        **If healthcare professionals are interested in using it, they can 
          incorporate this model into their practice as a supplementary 
          tool for risk assessment and decision-making.**
        
        **Author: Heshan Chandeepa ([GitHub](https://github.com/HChandeepa/Stroke_Prediction_System-Machine_Learning_Approach))**
        
        You can see the steps of building the model, evaluating it, and cleaning the data itself
        on my GitHub repo [here](https://github.com/HChandeepa/Stroke_Prediction_System-Machine_Learning_Approach). 
        """)

    stroke = load_dataset()
    st.sidebar.title("Feature Selection")
    st.sidebar.image("images/brain.jpg", width=100)
   

    input_df = user_input_features()

    # Ensure categorical features are one-hot encoded consistently
    input_df['ever_married'] = input_df['ever_married'].astype(str)
    input_df['work_type'] = input_df['work_type'].astype(str)
    input_df['smoking_status'] = input_df['smoking_status'].astype(str)
    input_df['Residence_type'] = input_df['Residence_type'].astype(str)
    input_df = pd.get_dummies(input_df, columns=['ever_married', 'work_type', 'smoking_status','Residence_type'])

    # Ensure input features match training features
    missing_cols = set(stroke.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0

    input_df = input_df[stroke.columns]  # Reorder columns to match training data

    log_model = pickle.load(open(LOG_MODEL_PATH, "rb"))

    selected_columns = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status']
    new_df = input_df[selected_columns]

    if submit:
        
        prediction_prob = log_model.predict_proba(new_df)
        prediction = log_model.predict(new_df)
        if prediction == 0:
            st.markdown(f"**The probability that you'll have"
                        f" Stroke Risk is {round(prediction_prob[0][1] * 100, 2)}%."
                        f" You are healthy!**")
            st.image("images/doctor ok.jpg",
                     caption="Your Brain seems to be okay! - Dr. Logistic Regression")
        else:
            st.markdown(f"**The probability that you will have"
                        f" Stroke Risk is {round(prediction_prob[0][1] * 100, 2)}%."
                        f" It sounds like you are not healthy.**")
            st.image("images/doctor-bad.jpg",
                     caption="I'm not satisfied with the condition of your Brain! - Dr. Logistic Regression")

    # Display user input features
    with st.sidebar:
        st.header("User Input Features")
        st.write(new_df)

if __name__ == "__main__":
    main()
