import streamlit as st

# import basic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# user-interactive visulization with plotly
import plotly.express as px
import plotly.figure_factory as ff
from io import StringIO


def create_input_features():
    def mapper(string):
        if string == "Yes":
            return 1
        elif string == "No":
            return 0

    st.sidebar.header("Create a datapoint")
    gender = st.sidebar.selectbox("Gender", ("Male", "Female", "Other"))
    hypertension = st.sidebar.selectbox("Hypertension", ("Yes", "No"))
    heart_disease = st.sidebar.selectbox("Heart Disease", ("Yes", "No"))
    ever_married = st.sidebar.selectbox("Married", ("Yes", "No"))
    work_type = st.sidebar.selectbox(
        "Occupation",
        ("children", "Govt_job", "Never_worked", "Private", "Self-employed"),
    )
    residence_type = st.sidebar.selectbox("Type of Residence", ("Urban", "Rural"))
    smoking_status = st.sidebar.selectbox(
        "Smoking Status", ("formerly smoked", "never smoked", "smokes", "Unknown")
    )
    avg_glucose_level = st.sidebar.slider("Average Glucose Level", 30, 300, 100)
    bmi = st.sidebar.slider("Body Mass Index (BMI)", 1, 100, 20)
    age = st.sidebar.slider("Age", 1, 120, 21)

    data_point = {
        "gender": gender,
        "age": age,
        "hypertension": mapper(hypertension),
        "heart_disease": mapper(heart_disease),
        "ever_married": mapper(ever_married),
        "work_type": work_type,
        "residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status,
    }
    features = pd.DataFrame(data_point, index=[0])

    return features


st.title("Stroke Prediction: Exploratory Data Analysis")

st.markdown(
    """
## 1. Introduction

According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.

This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relavant information about the patient.

"""
)

st.markdown("### 1.1 Data Overview")

uploaded_file = df = st.session_state["df"]
if uploaded_file is not None:
    # To read file as bytes:
    # bytes_data = uploaded_file.getvalue()
    # # st.write(bytes_data)

    # # To convert to a string based IO:
    # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # # st.write(stringio)

    # # To read file as string:
    # string_data = stringio.read()
    # # st.write(string_data)

    # # Can be used wherever a "file-like" object is accepted:
    # dataframe = pd.read_csv(uploaded_file)
    dataframe = uploaded_file
    rows = st.sidebar.slider("Select number of rows to view", 1, 10, 5)
    st.markdown("#### Dataset Snapshot")
    st.write(dataframe.head(rows))

else:
    st.write("Awaiting CSV file to be uploaded.")


if uploaded_file is not None:
    # if st.button("Show Dataset Description"):

    st.markdown("#### Dataset Description")
    st.write(dataframe.describe())

if uploaded_file is not None:

    st.markdown("#### Correlation Plot")

    fig = plt.figure(figsize=(15, 7))

    # Compute correlations
    corr = dataframe.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(corr, mask=mask, cmap="viridis", annot=False)
    st.pyplot(fig)


features = create_input_features()
if st.sidebar.button("Submit?"):
    st.markdown("#### Custom datapoint")
    st.write(features)
