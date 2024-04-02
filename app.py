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
        age = st.sidebar.number_input('Enter your Age', 0, 100, 0)
        hypertension = st.sidebar.selectbox('Hypertension', options=[0, 1])
        ever_married = st.sidebar.selectbox('Ever Married', options=[0,1])
        work_type = st.sidebar.selectbox('Work Type', options=[0,1,2,3,4])
        heart_disease = st.sidebar.selectbox('Heart Disease', options=[0,1])
        smoking_status = st.sidebar.selectbox('Smoking Status', options=[0,1,2,3])
        bmi = st.sidebar.number_input('BMI', 0.0, 100.0, 0.0, step=0.1)
        gender = st.sidebar.selectbox('Gender',options=[0,1,2])

        features = pd.DataFrame({
            "age": [age],
            "hypertension": [hypertension],
            "ever_married": [ever_married],
            "work_type": [work_type],
            "heart_disease": [heart_disease],
            "smoking_status": [smoking_status],
            "bmi": [bmi],
            "gender": [gender]
        })

        return features
    
    st.set_page_config(
        page_title="Stroke Risk Prediction App",
        page_icon="stroke.jpg"
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("stroke (1).jpg",
                 caption="I'll help you diagnose your heart health! - Dr. Logistic Regression",
                 width=150)
        submit = st.button("Predict")
    with col2:
        st.markdown("""
        Did you know that machine learning models can help you
        predict heart disease pretty accurately? In this app, you can
        estimate your chance of heart disease (yes/no) in seconds!
        
        Here, a logistic regression model using an undersampling technique
        was constructed using survey data of over 300k US residents from the year 2020.
        This application is based on it because it has proven to be better than the random forest
        (it achieves an accuracy of about 80%, which is quite good).
        
        To predict your heart disease status, simply follow the steps bellow:
        1. Enter the parameters that best describe you;
        2. Press the "Predict" button and wait for the result.
            
        **Keep in mind that this result is not equivalent to a medical diagnosis!
        This model would never be adopted by health care facilities because of its less
        than perfect accuracy, so if you have any problems, consult a human doctor.**
        
        **Author: Kamil Pytlak ([GitHub](https://github.com/kamilpytlak/heart-condition-checker))**
        
        You can see the steps of building the model, evaluating it, and cleaning the data itself
        on my GitHub repo [here](https://github.com/kamilpytlak/data-analyses/tree/main/heart-disease-prediction). 
        """)

    stroke = load_dataset()
    st.sidebar.title("Feature Selection")
    st.sidebar.image("stroke (1).jpg", width=100)
    st.title("Stroke Risk Prediction App")

    input_df = user_input_features()

    # Ensure categorical features are one-hot encoded consistently
    input_df['ever_married'] = input_df['ever_married'].astype(str)
    input_df['work_type'] = input_df['work_type'].astype(str)
    input_df['smoking_status'] = input_df['smoking_status'].astype(str)
    input_df = pd.get_dummies(input_df, columns=['ever_married', 'work_type', 'smoking_status'])

    # Ensure input features match training features
    missing_cols = set(stroke.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0

    input_df = input_df[stroke.columns]  # Reorder columns to match training data

    log_model = pickle.load(open(LOG_MODEL_PATH, "rb"))

    selected_columns = ['age','gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type','bmi','smoking_status']
    new_df = input_df[selected_columns]

    if submit:
        
        prediction_prob = log_model.predict_proba(new_df)
        prediction = log_model.predict(new_df)
        if prediction == 0:
            st.markdown(f"**The probability that you'll have"
                        f" heart disease is {round(prediction_prob[0][1] * 100, 2)}%."
                        f" You are healthy!**")
            st.image("images/heart-okay.jpg",
                     caption="Your heart seems to be okay! - Dr. Logistic Regression")
        else:
            st.markdown(f"**The probability that you will have"
                        f" heart disease is {round(prediction_prob[0][1] * 100, 2)}%."
                        f" It sounds like you are not healthy.**")
            st.image("stroke (1).jpg",
                     caption="I'm not satisfied with the condition of your heart! - Dr. Logistic Regression")

    # Display user input features
    with st.sidebar:
        st.header("User Input Features")
        st.write(new_df)

if __name__ == "__main__":
    main()
