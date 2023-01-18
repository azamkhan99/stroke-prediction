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


df = st.session_state["df"]
