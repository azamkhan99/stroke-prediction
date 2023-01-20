# Import the required Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px

st.set_page_config(layout="wide")

# Add a title and intro text
st.title("Stroke Prediction")
st.markdown(
    """This is a web app to allow prediction of stroke given a set of input features.

Begin by uploading a dataset."""
)

# Sidebar setup
st.sidebar.title("Sidebar")

use_sample_data = st.sidebar.radio(
    "Dataset", ("Use Sample Data", "Upload a file containing data")
)

if use_sample_data == "Use Sample Data":
    df = pd.read_csv("Data/healthcare-dataset-stroke-data.csv")
    st.session_state["df"] = df
    if df is not None:
        st.markdown("You may now navigate to Data Analysis Page!")

elif use_sample_data == "Upload a file containing data":

    upload_file = st.sidebar.file_uploader("")

    # Check if file has been uploaded
    if upload_file is not None:
        df = pd.read_csv(upload_file)
        st.session_state["df"] = df
        if df is not None:
            st.markdown("You may now navigate to Data Analysis Page!")
