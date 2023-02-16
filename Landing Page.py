# Import the required Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import plotly.express as px

#packages for animations
import json
import requests
from streamlit_lottie import st_lottie





st.set_page_config(layout="wide")

image = Image.open("Images/deloitte.png")

col1, col2, col3, col4 = st.columns(4, gap="small")

with col1:
    st.image(image, width=170)

with col2:
    st.title("Stroke Prediction")

#functions for animations

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_hello = load_lottiefile("animations/welcome.json")


st_lottie(
    lottie_hello,
    speed=0.8,
    reverse=False,
    loop=True,
    quality="low", # medium ; high
    height=300,
    width=600,
    key=None,
)



# Add a title and intro text

st.markdown(
    """

    This web-application allows user to add new samples and upload a file with custom dataset for real-time stroke prediction. 

    This demo will use sample data from Kaggle, but user will be able to upload their own data for Data Overview, Feature Engineering and Prediction.

    *Note: Due to the limitation of long training time, Modeling result is for sample data only.*

"""
)

link = "[Data Source](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)"
st.markdown(link, unsafe_allow_html=True)

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
