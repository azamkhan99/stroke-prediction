import streamlit as st

# import basic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.helpers import present_prediction, pr_comparison

# user-interactive visulization with plotly
import plotly.express as px
import plotly.figure_factory as ff
from io import StringIO
from streamlit_option_menu import option_menu
import joblib
from imblearn.ensemble import BalancedRandomForestClassifier
from feature_engine.encoding import OneHotEncoder as fe_OneHotEncoder

from utils.helpers import load_pretrained_model

# def create_input_features():


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

# data_point = {
#     "gender": gender,
#     "age": age,
#     "hypertension": mapper(hypertension),
#     "heart_disease": mapper(heart_disease),
#     "ever_married": mapper(ever_married),
#     "work_type": work_type,
#     "residence_type": residence_type,
#     "avg_glucose_level": avg_glucose_level,
#     "bmi": bmi,
#     "smoking_status": smoking_status,
# }
# features = pd.DataFrame(data_point, index=[0])

# return features


def mapper(string):
    if string == "Yes":
        return 1
    elif string == "No":
        return 0


def create_input_features(
    gender,
    hypertension,
    heart_disease,
    ever_married,
    work_type,
    residence_type,
    smoking_status,
    avg_glucose_level,
    bmi,
    age,
):
    data_point = {
        "gender": gender,
        "age": age,
        "hypertension": mapper(hypertension),
        "heart_disease": mapper(heart_disease),
        "ever_married": mapper(ever_married),
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status,
    }
    features = pd.DataFrame(data_point, index=[0])

    st.session_state.datapoint = features


# if st.sidebar.button("Submit?"):
#     features = create_input_features(
#         gender,
#         hypertension,
#         heart_disease,
#         ever_married,
#         work_type,
#         residence_type,
#         smoking_status,
#         avg_glucose_level,
#         bmi,
#         age,
#     )
#     st.markdown("#### Custom datapoint")


if "datapoint" not in st.session_state:
    st.session_state.datapoint = " "


st.sidebar.button(
    "Submit",
    on_click=create_input_features,
    args=(
        gender,
        hypertension,
        heart_disease,
        ever_married,
        work_type,
        residence_type,
        smoking_status,
        avg_glucose_level,
        bmi,
        age,
    ),
)


st.write("Custom datapoint", st.session_state.datapoint)


selected = option_menu(
    menu_title=None,
    options=[
        "Random Forest Classifier",
        "Support Vector Machine",
        "XGBoost",
        "Trained Logistic Regression",
    ],
    # icons = ['house', 'book', 'envelope'],
    orientation="horizontal",
)

inpu = st.session_state.datapoint


# transform the data


if "inpu" not in st.session_state:
    st.session_state.inpu = inpu

if type(inpu) == pd.DataFrame:

    inpu = st.session_state.ohe_enc.transform(inpu)
    inpu["avg_glucose_level_ranked"] = pd.cut(
        inpu["avg_glucose_level"],
        bins=[0, 90, 160, 230, 500],
        labels=["Low", "Normal", "High", "Very High"],
    )
    # fit the encoder
    inpu = st.session_state.encoder.transform(inpu)
    inpu.drop("avg_glucose_level", axis=1, inplace=True)
    inpu["bmi"] = inpu.pop("bmi")

    if selected == "Random Forest Classifier":
        model = joblib.load("models/balanced_randomforest.joblib")

        st.write("Transformed datapoint", inpu)
        present_prediction(model, inpu)


    elif selected == "Support Vector Machine":
        model = load_pretrained_model("models/svm.pkl")
        scaler = load_pretrained_model("models/scaler.pkl")
        st.write("Transformed datapoint", inpu)
        present_prediction(model, inpu)
    elif selected == "XGBoost":
        st.text("...")
    elif selected == "Trained Logistic Regression":
        model = joblib.load("models/trained_lr.joblib")

        st.write("Transformed datapoint", inpu)
        present_prediction(model, inpu)
