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
upload_file = st.sidebar.file_uploader("Upload a file containing data")

# Check if file has been uploaded
if upload_file is not None:
    df = pd.read_csv(upload_file)
    st.session_state["df"] = df
