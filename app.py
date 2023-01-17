import streamlit as st

#import basic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#user-interactive visulization with plotly
import plotly.express as px
import plotly.figure_factory as ff


#Read data
df = pd.read_csv("Data/healthcare-dataset-stroke-data.csv")
print(f"Dimention of raw data is {df.shape}")


#Get rid of the 'id' column. In general, id column doesn't contribute to prediction.
#id column in this data has been masked by the author for data privacy.
#An alternative way is to replace index with id column. This preserves data information especially for time series data.
df.drop('id',inplace=True,axis=1)
print(f"Dimention of data after dropping the 'id' column is {df.shape}")

st.title("Hello, world!")

st.header("Stroke Prediction: Exploratory Data Analysis")

st.write(df.head(10))

st.header('Dataset Description')


st.write(df.describe())


