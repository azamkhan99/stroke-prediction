# streamlit packages
import streamlit as st
from streamlit_option_menu import option_menu

# import basic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# user-interactive visulization with plotly
import plotly.express as px
import plotly.figure_factory as ff

# to split the datasets
from sklearn.model_selection import train_test_split

# feature engineering imputation
from feature_engine import imputation as mdi

# for one hot encoding with sklearn
from sklearn.preprocessing import OneHotEncoder

# for encoding with feature-engine
from feature_engine.encoding import OneHotEncoder as fe_OneHotEncoder
from feature_engine.encoding import OrdinalEncoder as fe_OrdinalEncoder

# for Q-Q plots
import scipy.stats as stats

# read images
from PIL import Image


# Read data
df = st.session_state["df"]

# drop id
# df.drop("id", inplace=True, axis=1)


########################'Define Colours'##############################
enmax_palette = ["#86BC25", "#C4D600", "#43B02A", "#2C5234"]
color_codes_wanted = ["d_green", "green_1", "green_2", "green_3"]

d_colours = lambda x: enmax_palette[color_codes_wanted.index(x)]


##################Feature Engineering app section###############################
st.title("Feature Engineering")

# Split into Train and Test Data

X = df.drop(["stroke"], axis=1)
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, shuffle=True
)

see_data = st.expander("Split into train and test 👉")
with see_data:
    st.write(f"- X_train: {X_train.shape}")
    st.write(f"- X_test: {X_test.shape}")

see_data = st.expander("Stroke cases in train and test 👉")
with see_data:

    st.write(
        f"- y_train: {round(y_train.value_counts()[1]/len(y_train),3)*100}%"
    )
    st.write(
        f"- y_test: {round(y_test.value_counts()[1]/len(y_test),3)*100}%"
    )

# Create 3 buttons to navigate between 'Categorical Variable Encoding', 'Handling Outliers', 'Deal with Missing Values'
selected = option_menu(
    menu_title=None,
    options=[
        "Categorical Variable Encoding",
        "Handling Outliers",
        "Deal with Missing Value",
    ],
    icons=["check-circle-fill", "exclamation-circle-fill", "question-circle-fill"],
    orientation="horizontal",
)

###########################'Categorical Variable Encoding'###################################
if selected == "Categorical Variable Encoding":
    #st.subheader(f"{selected}")

    # Display text


    
    see_data = st.expander("Apply **One Hot Encoding** to the following columns")
    with see_data:
                st.markdown(
        """
        - gender
        - ever_married
        - work_type
        - Residence_type
        - smoking_status

"""
    )

    # One hot encoding

    ohe_enc = fe_OneHotEncoder(top_categories=None, drop_last=True)

    ohe_enc.fit(X_train)

    if "ohe_enc" not in st.session_state:
        st.session_state.ohe_enc = ohe_enc
    X_train = ohe_enc.transform(X_train)
    X_test = ohe_enc.transform(X_test)

    if "X_train" not in st.session_state:
        st.session_state.X_train = X_train
    if "X_test" not in st.session_state:
        st.session_state.X_test = X_test

    st.markdown("After one hot encoding:")
    st.write(X_train.head())


##########################################'Handling Outliers'##############################
if selected == "Handling Outliers":
    #st.subheader(f"{selected}")
    

    ######################Handling Outliers for bmi##########################################
    st.markdown(
        """
        #### Handling Outliers for bmi
    """
    )
      
    def diagnostic_plots(df, variable):

        # define figure size
        plt.figure(figsize=(10, 3))

        # histogram
        plt.subplot(1, 3, 1)
        sns.distplot(df[variable], bins=30, color=d_colours("d_green"))
        plt.title("Histogram")

        # Q-Q plot
        ax2 = plt.subplot(1, 3, 2)
        res = stats.probplot(df[variable], dist="norm", plot=plt)
        ax2.get_lines()[0].set_color("#86BC25")
        ax2.get_lines()[0].set_linewidth(1)
        plt.ylabel("Variable quantiles")

        # boxplot
        plt.subplot(1, 3, 3)
        sns.boxplot(y=df[variable], color=d_colours("d_green"))
        plt.title("Boxplot")

        plt.show()

    st.set_option("deprecation.showPyplotGlobalUse", False)  # ignore warning
    st.pyplot(diagnostic_plots(X_train, "bmi"))


    def find_skewed_boundaries(df, variable, distance):

        IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)

        upper_boundary = round(df[variable].quantile(0.75) + (IQR * distance), 2)

        return upper_boundary

    ulbmi1 = find_skewed_boundaries(df, "bmi", 1.5)
    bmi_outlier = 0
    for a in df["bmi"].to_numpy():
        if a > ulbmi1:
            bmi_outlier += 1
    ulbmi2 = find_skewed_boundaries(df, "bmi", 3)
    bmi_outlier1 = 0
    for a in df["bmi"].to_numpy():
        if a > ulbmi2:
            bmi_outlier1 += 1

    



    see_data = st.expander("Apply **Capping** to handle outliers")
    with see_data:
         st.write('Replace outliers with maximum allowable value')
         st.write(
        f"The upper bound is found to be **{ulbmi1}**, with **{bmi_outlier}** outliers, which covers the majority of population, including extreme obesity population shown as below:"
    )
         image = Image.open("Images/body-mass-index-bmi-chart.jpg")
         st.image(image, caption="Body Mass Index Classes")
    
         st.markdown(
        """

        Replace outliers with median/mean is another option, but it could lose the representation of the population with large BMI in the model.

        *Note: When doing capping, we tend to cap values both in train and test set. It is important to remember that the capping values MUST be derived from the train set. And then use those same values to cap the variables in the test set.*
"""
    )
        
  
    
    bmi_upper_limit = find_skewed_boundaries(X_train, "bmi", 1.5)
    X_train["bmi"] = np.where(
        X_train["bmi"] > bmi_upper_limit, bmi_upper_limit, X_train["bmi"]
    )
    X_test["bmi"] = np.where(
        X_test["bmi"] > bmi_upper_limit, bmi_upper_limit, X_test["bmi"]
    )

    st.set_option("deprecation.showPyplotGlobalUse", False)  # ignore warning
    st.pyplot(diagnostic_plots(X_train, "bmi"))

    #############################Handling Outliers for avg_glucose_level#########################
    X_train = st.session_state.X_train

    X_test = st.session_state.X_test

    st.markdown(
        """
        #### Handling Outliers for avg_glucose_level
"""
    )

    st.set_option("deprecation.showPyplotGlobalUse", False)  # ignore warning
    st.pyplot(diagnostic_plots(X_train, "avg_glucose_level"))

    see_data = st.expander("Apply **Binning** to handle outliers")
    with see_data:
         st.markdown(
        """
        **Binning** helps handle outliers by placing these values into the lower or higher intervals, which allows the right-tail dense group to be represented in the general population.
        Based on domain knowledge, average glucose level are grouped as following :
        - Low: <90
        - Normal: 90 - 160
        - High: 161 - 230
        - Very High: 231 - 500

"""
    )

    # apply to train data
    X_train["avg_glucose_level_ranked"] = pd.cut(
        X_train["avg_glucose_level"],
        bins=[0, 90, 160, 230, 500],
        labels=["Low", "Normal", "High", "Very High"],
    )
    # apply to test data
    X_test["avg_glucose_level_ranked"] = pd.cut(
        X_test["avg_glucose_level"],
        bins=[0, 90, 160, 230, 500],
        labels=["Low", "Normal", "High", "Very High"],
    )

    fig = plt.figure(figsize=(10, 3))

    sns.countplot(
        x="avg_glucose_level_ranked",
        data=X_train,
        palette={
            d_colours("d_green"),
            d_colours("green_1"),
            d_colours("green_2"),
            d_colours("green_3"),
        },
    )
    st.pyplot(fig)

    st.markdown('''
    The grouped avg_glucose_level is now a categorical variable, which also needs to be encoded as numerical representation.
    '''
    )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        see_data = st.expander("Before **Ordinal Encoding** ")
        with see_data:
            
            st.write(X_train[["avg_glucose_level", "avg_glucose_level_ranked"]].head(20))

    with col2:
        # set up the encoder
        encoder = fe_OrdinalEncoder(
            encoding_method="ordered", variables=["avg_glucose_level_ranked"]
        )

        # fit the encoder
        encoder.fit(X_train, y_train)
        if "encoder" not in st.session_state:
            st.session_state.encoder = encoder

        # transform the data
        X_train = encoder.transform(X_train)
        X_test = encoder.transform(X_test)


        see_data = st.expander("After **Ordinal Encoding** ")
        with see_data:
            
            st.write(X_train[["avg_glucose_level", "avg_glucose_level_ranked"]].head(20))
        #st.write(X_train[["avg_glucose_level", "avg_glucose_level_ranked"]].head(20))

        st.session_state.X_train = X_train
        st.session_state.X_test = X_test


######################################Deal with Missing Values#######################
if selected == "Deal with Missing Value":
    #st.subheader(f"{selected}")

    st.markdown(
        """
        #### Missing Data Imputation for bmi
"""
    )

    bmi_na = df["bmi"].isna().sum()
    bmi_all = df.shape[0]

    st.write(
        f"There are **{round((bmi_na/bmi_all),3)*100}%** missing value in bmi"
    )
    see_data = st.expander("Apply **Median Imputation** to handle missing value")
    with see_data:
        st.markdown(
        """
        Following imputation methods are tested based on AUC score:

        - Mean Imputation
        - Median Imputation
        - Multivariate imputation with KNN

        In summary, univariate imputation (median/mean) and multivariate imputation (KNN) return similar performance. **Univariate imputation** was selected to avoid introducing additional complexity.

        *Note: The imputation values (that is the median/mean) should be calculated using the training set, and the same value should be used to impute the test set. This is to avoid overfitting.*

"""
    )
        
    X_train = st.session_state.X_train

    X_test = st.session_state.X_test

    mean = np.round(X_train.bmi.mean(), 1)
    median = np.round(X_train.bmi.median(), 1)
    

    variance = round(X_train["bmi"].var(), 2)

    X_train["bmi_mean"] = X_train["bmi"].fillna(mean)
    X_train["bmi_median"] = X_train["bmi"].fillna(median)
    variance_mean = round(X_train["bmi_mean"].var(), 2)
    variance_median = round(X_train["bmi_median"].var(), 2)
    
    see_data = st.expander("More information for Mean, Median and Variance")
    with see_data:

        st.write(f"Mean: {mean}")
        st.write(f"Median: {median}")
        st.write("Variance")
        st.write(f"Before Imputation: {variance}")
        st.write(f"After Mean Imputation: {variance_mean}")
        st.write(f"After Median Imputation: {variance_median}")
        

    # check distibution for bmi before imputation
    fig = px.histogram(
        X_train,
        x="bmi",
        marginal="box",
        color_discrete_sequence=["#86BC25"],
        hover_data=X_train,
        width=1000,
        height=800,
    )
    fig.update_layout(title_text="Box Plot and Distribution of bmi Before Imputation")

    st.plotly_chart(fig)

    # check distribution of bmi after mean/median imputation
    x2 = X_train["bmi_median"]
    x3 = X_train["bmi_mean"]

    hist_data = [x2, x3]

    group_labels = ["Median Imputation", "Mean Imputation"]
    colors = ["#86BC25", "#2C5234"]

    # Create distplot with curve_type set to 'normal'
    fig = ff.create_distplot(
        hist_data,
        group_labels,
        show_hist=False,
        colors=colors,
    )

    fig.layout.height = 800
    fig.layout.width = 1200

    # Add title
    fig.update_layout(
        title_text="Compare bmi Distribution after Mean and Median Imputation"
    )
    st.plotly_chart(fig)

    # drop bmi and bmi_mean column
    X_train.drop(["bmi", "bmi_mean"], axis=1, inplace=True)

    # use median from training set to replace missing values in test set to prevent overfitting
    X_test["bmi_median"] = X_test["bmi"].fillna(median)
    X_test.drop(["bmi"], axis=1, inplace=True)
    X_train.rename(columns={"bmi_median": "bmi"}, inplace=True)
    X_test.rename(columns={"bmi_median": "bmi"}, inplace=True)

    X_train.drop("avg_glucose_level", axis=1, inplace=True)
    X_test.drop("avg_glucose_level", axis=1, inplace=True)

    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
