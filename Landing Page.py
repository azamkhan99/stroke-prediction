# Import the required Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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

"""
)

# Sidebar setup
st.sidebar.title("Begin by uploading a dataset")

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

row6_spacer1, row6_1, row6_spacer2 = st.columns((0.2, 7.1, 0.2))
with row6_1:
    st.subheader("Currently selected data:")

# row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3, row2_3, row2_spacer4, row2_4, row2_spacer5   = st.columns((.2, 1.6, .2, 1.6, .2, 1.6, .2, 1.6, .2))
# with row2_1:
#     unique_patients_in_df = df.game_id.nunique()
#     str_patients = "🧍 " + str(unique_patients_in_df) + " People"
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
st.text("")
