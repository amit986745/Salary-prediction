from itertools import count
import streamlit as st 
import pickle
import numpy as np

def load_model():
    with open(r"C:\Users\AMIT PAREEK\Documents\project deploment\project 1\saved_steps.pkl", 'rb') as file:
        data = pickle.load(file)
    return data
data=load_model()
regressor_loaded = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("Software developer salary prediction")
    st.write("we need some more information to predict")
    
    countries=('Turkey', 'France', 'Russian Federation', 'Netherlands', 'Poland',
       'Spain', 'Italy',
       'United Kingdom of Great Britain and Northern Ireland', 'Germany',
       'India', 'Brazil', 'United States of America', 'Canada',
       'Switzerland', 'Sweden', 'Israel', 'Australia', 'Norway')
    education=('Bachelor’s degree', 'Master’s degree', 'Post grad',
       'Less than a Bachelors')
    
    country=st.selectbox("country",countries)
    education=st.selectbox("Education",education)
    experience=st.slider("Year of Experience",0,15,3)
    ok=st.button("calculate salary")
    if ok:
        X = np.array([[country, education, experience ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)
        salary =regressor_loaded.predict(X)
        st.subheader(f"the estimated salary is{salary[0]:.2f}")
        