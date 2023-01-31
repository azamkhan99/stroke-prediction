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
    dataframe.drop("id", axis=1, inplace=True)
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

    col1, col2 = st.columns(2, gap="small")

    with col1:
        labels = dataframe["ever_married"].value_counts().index
        values = dataframe["ever_married"].value_counts().values
        fig = px.pie(
            df,
            values=values,
            names=labels,
            title="Percentage of dataset ever married",
            hole=0.3,
        )
        st.plotly_chart(fig)
    with col2:
        labels = dataframe["work_type"].value_counts().index
        values = dataframe["work_type"].value_counts().values
        fig = px.pie(
            df,
            values=values,
            names=labels,
            title="Analysis of Patient Occupations",
            hole=0.3,
        )
        st.plotly_chart(fig)


if uploaded_file is not None:

    st.markdown("#### Correlation Plot")

    fig = plt.figure(figsize=(12, 4))

    # Compute correlations
    corr = dataframe.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(corr, mask=mask, cmap="viridis", annot=False)
    st.pyplot(fig)
