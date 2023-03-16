# Import the required Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import seaborn as sns

import plotly.express as px


# packages for animations
import json
import requests
from streamlit_lottie import st_lottie

#Must be called once only as the first streamlit command
st.set_page_config(#page_title='Stroke Prediction',
                   layout="wide",
                   page_icon=Image.open("Images/logo.png"))


########################'Define Colours'##############################
enmax_palette = ["#86BC25", "#C4D600", "#43B02A", "#2C5234"]
color_codes_wanted = ["d_green", "green_1", "green_2", "green_3"]

d_colours = lambda x: enmax_palette[color_codes_wanted.index(x)]
######################################################################


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Sidebar setup
st.sidebar.title("Begin by uploading a dataset")

use_sample_data = st.sidebar.radio(
    "", ("Use Sample Dataset", "Upload Custom CSV File"), index=1
)

if use_sample_data == "Use Sample Dataset":
    upload_file = pd.read_csv("Data/healthcare-dataset-stroke-data.csv")
    df = upload_file
    df.drop("id", axis=1, inplace=True)
    st.session_state["df"] = df


elif use_sample_data == "Upload Custom CSV File":

    upload_file = st.sidebar.file_uploader("")

    # Check if file has been uploaded
    if upload_file is not None:
        df = pd.read_csv(upload_file)
        st.session_state["df"] = df
        if df is not None:
            st.markdown("You may now navigate to Data Overview Page!")

image = Image.open("Images/deloitte.png")
logo = Image.open("Images/logo.png")

col1, col2, col3, col4 = st.columns((1, 1, 5, 2))

with col1:
    st.image(image, use_column_width=True)

with col2:
    st.image(logo ,use_column_width=True)

with col3:
    #st.title("Stroke Prediction")
    lottie_hello = load_lottiefile("animations/welcome.json")
    st_lottie(
        lottie_hello,
        speed=0.8,
        reverse=False,
        loop=True,
        quality="low",  # medium ; high
        height=200,
        width=300,
        key=None,
    )

# Add a title and intro text



if upload_file is None:

    st.subheader("An application for fast and easy data processing, visualisation and real-time stroke prediction.")

    see_data = st.expander("Disclaimer")
    with see_data:
        st.markdown(
        """

        ⚠️**Warning**: This application is designed for skill developement, which does not provide any clinical recommendations.

        *Note:
        This demo uses publically available data from Kaggle and prediction is based on ML models that are previously trained on sample data.

    """
    )
        link = "[Data Source](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)"
        st.markdown(link, unsafe_allow_html=True)

    st.markdown("""Data Scientists: Upload a csv file or use sample dataset""")
    st.markdown("""Clinicians/GPs: Navigate to __Prediction__ Page""")

else:

    row6_spacer1, row6_1, row6_spacer2 = st.columns((0.2, 7.1, 0.2))
    #with row6_1:
        #st.subheader("Currently selected data")

    row3_spacer1, row3_1, row3_spacer2 = st.columns((0.2, 7.1, 0.2))
    with row3_1:
        st.markdown("")
        see_data = st.expander("Click to see sample dataset first 👉")
        with see_data:
            st.dataframe(data=df)
    st.text("")

    pad0, pies, pad1, dist, pad2 = st.columns((0.25, 3, 1, 3, 3))

    with pies:
        labels = df["stroke"].value_counts().index
        values = df["stroke"].value_counts().values
        fig = px.pie(
            df,
            values=values,
            names=labels,
            title="Distribution of Target value",
            hole=0.3,
            color_discrete_sequence=["#86BC25", "#43B02A"]
        )

        fig.update_layout(
            autosize=False,
            width=400,
            height=200,
            margin=dict(l=0, r=25, b=0, t=50, pad=1),
        )
        st.plotly_chart(fig)

        labels = df["gender"].value_counts().index
        values = df["gender"].value_counts().values
        fig = px.pie(
            df,
            values=values,
            names=labels,
            title="Distribution of Population gender",
            hole=0.3,
            color_discrete_sequence=["#86BC25", "#43B02A"]
        )
        fig.update_layout(
            autosize=False,
            width=400,
            height=200,
            margin=dict(l=0, r=25, b=0, t=50, pad=1),
        )
        st.plotly_chart(fig)

    with dist:
        fig = px.histogram(df, x="age", title="Age distribution in dataset",color_discrete_sequence=["#86BC25"])
        fig.update_layout(
            autosize=False,
            width=650,
            height=445,
            margin=dict(l=0, r=25, b=0, t=50, pad=1),
        )
        st.plotly_chart(fig)

    position0, position1, position2 = st.columns((0.25,7,3))

    with position1:
        see_data = st.expander("Woking Group 💗")
        with see_data:
            image = Image.open("Images/Team Structure.png")
            st.image(image)
