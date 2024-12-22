import streamlit as st
import pickle
import numpy as np

# Load the model
def load_model():
    with open('model.pkl', 'rb') as model:
        return pickle.load(model)

data = load_model()

regressor = data['model']
le_country = data['le_country']
le_education = data['le_education']
le_employment = data['le_employment']

# Prediction page
def show_predict_page():
    st.title('Software Developer Salary Prediction')
    
    st.write("### Please fill in the details below to get the salary prediction.")
    countries = ('United States of America', 
            'Spain', 
            'Canada', 
            'Germany',
            'Australia', 
            'India', 
            'Austria', 
            'Netherlands', 
            'Italy', 
            'Brazil',
            'Sweden', 
            'France', 
            'Poland',
            'United Kingdom of Great Britain and Northern Ireland', 
            'Ukraine',
            'Switzerland')

    education = ('Bachelor’s degree', 'Post grad', 'Less than a Bachelors',
       'Master’s degree')

    employment = ('Full Time', 'Independent', 'Other')
    
    # select boxes
    country = st.selectbox('Country', countries)
    education_level = st.selectbox('Education Level', education)
    employment_status = st.selectbox('Employment Status', employment)

    # experience slider 
    experience = st.slider('Years of Experience', 0, 50, 3)
    
    # Predict button
    predict_button = st.button("Calculate Salary")
    
    if predict_button:
        x = np.array([[country, education_level, employment_status, experience]])
        x[:, 0] = le_country.transform(x[:, 0])
        x[:, 1] = le_education.transform(x[:, 1])
        x[:, 2] = le_employment.transform(x[:, 2])
        x = x.astype(float)
        
        salary = regressor.predict(x)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")