# Import the required Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import plotly.express as px

st.set_page_config(layout="wide")

image = Image.open("Images/deloitte.png")


col1, col2, col3, col4 = st.columns(4, gap="small")

with col1:
    st.image(image, width=170)

with col2:
    st.title("Stroke Prediction")

# Add a title and intro text

st.markdown(
    """

    This is a web app to allow prediction of stroke given a set of input features.

Begin by uploading a dataset."""
)

# Sidebar setup
st.sidebar.title("Sidebar")

use_sample_data = st.sidebar.radio(
    "Dataset", ("Use Sample Data", "Upload a file containing data")
)

if use_sample_data == "Use Sample Data":
    df = pd.read_csv("Data/healthcare-dataset-stroke-data.csv")
    df.drop("id", axis=1, inplace=True)
    st.session_state["df"] = df
    if df is not None:
        st.markdown("You may now navigate to Data Overview Page!")

elif use_sample_data == "Upload a file containing data":

    upload_file = st.sidebar.file_uploader("")

    # Check if file has been uploaded
    if upload_file is not None:
        df = pd.read_csv(upload_file)
        st.session_state["df"] = df
        if df is not None:
            st.markdown("You may now navigate to Data Overview Page!")
