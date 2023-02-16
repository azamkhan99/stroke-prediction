# Import the required Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import plotly.express as px


# packages for animations
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


# functions for animations


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
    quality="low",  # medium ; high
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
st.sidebar.title("Begin by uploading a dataset")

use_sample_data = st.sidebar.radio(
    "Dataset", ("Use Sample Data", "Upload a file containing data")
)

if use_sample_data == "Use Sample Data":
    df = pd.read_csv("Data/healthcare-dataset-stroke-data.csv")
    df.drop("id", axis=1, inplace=True)
    st.session_state["df"] = df


elif use_sample_data == "Upload a file containing data":

    upload_file = st.sidebar.file_uploader("")

    # Check if file has been uploaded
    if upload_file is not None:
        df = pd.read_csv(upload_file)
        st.session_state["df"] = df
        if df is not None:
            st.markdown("You may now navigate to Data Overview Page!")

row6_spacer1, row6_1, row6_spacer2 = st.columns((0.2, 7.1, 0.2))
with row6_1:
    st.subheader("Currently selected data:")

# row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3, row2_3, row2_spacer4, row2_4, row2_spacer5   = st.columns((.2, 1.6, .2, 1.6, .2, 1.6, .2, 1.6, .2))
# with row2_1:
#     unique_patients_in_df = df.game_id.nunique()
#     str_patients = "🧍 " + str(unique_patients_in_df) + " Data points"
#     st.markdown(str_patients)
# with row2_2:
#     unique_teams_in_df = len(np.unique(df_data_filtered.team).tolist())
#     t = " Teams"
#     if(unique_teams_in_df==1):
#         t = " Team"
#     str_teams = "🏃‍♂️ " + str(unique_teams_in_df) + t
#     st.markdown(str_teams)
# with row2_3:
#     total_goals_in_df = df_data_filtered['goals'].sum()
#     str_goals = "🥅 " + str(total_goals_in_df) + " Goals"
#     st.markdown(str_goals)
# with row2_4:
#     total_shots_in_df = df_data_filtered['shots_on_goal'].sum()
#     str_shots = "👟⚽ " + str(total_shots_in_df) + " Shots"
#     st.markdown(str_shots)

row3_spacer1, row3_1, row3_spacer2 = st.columns((0.2, 7.1, 0.2))
with row3_1:
    st.markdown("")
    see_data = st.expander("You can click here to see the raw data first 👉")
    with see_data:
        st.dataframe(data=df)
    if df is not None:
        st.markdown("You may now navigate to Data Overview Page!")
st.text("")
