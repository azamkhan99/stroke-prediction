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
from sklearn.metrics import confusion_matrix

from PIL import Image


########################'Define Colours'##############################
enmax_palette = ["#86BC25", "#C4D600", "#43B02A", "#2C5234"]
color_codes_wanted = ["d_green", "green_1", "green_2", "green_3"]

d_colours = lambda x: enmax_palette[color_codes_wanted.index(x)]
######################################################################


st.title("Exploratory Data Analysis")

st.markdown(
    """
## 1. Introduction

According to the World Health Organization (WHO), strokes are the 2nd leading cause of death globally, responsible for approximately :green[11%] of total deaths. Using data provided by [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) , we wanted to understand the predictability of stroke-risk based up various features.

Exploratory Data Analysis is the first step in any forecasting task, encompassing a variety of techniques to detect outliers and anomalies, test assumptions and maximise insight into the dataset.
"""
)

st.markdown("### 1.1 Data Overview")

df = pd.read_csv("Data/healthcare-dataset-stroke-data.csv")
uploaded_file = st.session_state["df"]

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
    rows = st.sidebar.slider("Select number of rows to view", 1, 10, 5)

else:
    st.write("Awaiting CSV file to be uploaded.")

plt.style.use("dark_background")

if uploaded_file is not None:

    col1, col2 = st.columns(2, gap="small")

    with col1:
        st.markdown(
            """
        The provided dataset is a CSV file, and was used to predict whether a patient is susceptible to strokes. Each row in the dataset is a new patient, with features relevant to the forecasting task such as such as: gender, age, bmi, pre-existing health conditions and smoking status. In total there were :green[5110 samples], with :green[12 attributes], a moderately sized dataset.

        Of note, is the imbalance to the dataset, with only :green[4.87%] of the data being incidences of strokes. Thus, care must be taken when building models, as data imbalance can influence how certain algorithms operate.

        An initial step was to generate a correlation plot, to understand how the features/attributes are related and to provide direction to the investigation. Evidently, there is :green[strong correlation between age and stroke-risk].
        """
        )

    with col2:
        st.markdown("#### Dataset Snapshot")
        st.write(dataframe.head(rows))


if uploaded_file is not None:
    col1, col2 = st.columns(2, gap="small")
    with col1:

        st.markdown("#### Feature Correlation")

        fig = plt.figure(figsize=(10, 5))

        # Compute correlations
        corr = dataframe.corr()

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # create deloitte themed colourmaps
        d_cmap = sns.dark_palette("#92d400", input="hex", reverse=False)

        enmax_palette = ["#86BC25", "#C4D600", "#43B02A", "#2C5234"]
        color_codes_wanted = ["d_green", "green_1", "green_2", "green_3"]

        d_colours = lambda x: enmax_palette[color_codes_wanted.index(x)]

        sns.heatmap(corr, mask=mask, cmap=d_cmap, annot=False)
        st.pyplot(fig)

    with col2:
        st.markdown("#### Stroke / Non-Stroke Ratio")
        labels = ["Non-stroke", "Stroke"]  # dataframe["stroke"].value_counts().index
        values = dataframe["stroke"].value_counts().values
        fig = px.pie(
            df,
            values=values,
            names=labels,
            hole=0.3,
            color_discrete_sequence=["#2C5234", "#86BC25"],
        )
        st.plotly_chart(fig)


# split stroke and non-stroke data
stroke = df[df["stroke"] == 1]
not_stroke = df[df["stroke"] == 0]


############################################## 2.1 AGE ####################################################################
st.markdown("## 2. Initial Observations")
st.markdown(
    """### 2.1. Age

Medical research has demonstrated that aging is the most robust non-modifiable risk factor for incident stroke, which doubles every 10 years after age 55 years. Approximately three-quarters of all strokes occur in persons aged ≥65 years. This is reflected within the stroke dataset, with the modal age of stroke incidence being :green[78], as opposed to :green[52] for non-strokes. Thus, :green[age proved to be the strongest predictor] out of all baseline features.
"""
)

link1 = "[Source](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6535078/#:~:text=Aging%20is%20the%20most%20robust,persons%20aged%20%E2%89%A565%20years)"
st.markdown(link1, unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="small")


if uploaded_file is not None:

    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.markdown("#### Age-Stroke Distribution")

        fig = plt.figure(figsize=(8, 5))

        # plot kde graphs
        sns.kdeplot(
            data=stroke,
            x="age",
            fill=True,
            alpha=0.8,
            linewidth=0.7,
            ec="white",
            color=d_colours("green_3"),
        )
        sns.kdeplot(
            data=not_stroke,
            x="age",
            fill=True,
            alpha=0.8,
            linewidth=0.7,
            ec="white",
            color=d_colours("d_green"),
        )

        # formatting
        plt.ylabel("Probability Density")
        plt.xlabel("Age (Years)")
        plt.legend(["Stroke", "Non-Stroke"], bbox_to_anchor=(1.1, 1.1))

        plt.ylabel("Probability Density")
        plt.xlim([0,100])

        st.pyplot(fig)

    with col2:

        st.markdown("#### Age-Marital-Status Distribution")
        fig = plt.figure(figsize=(8, 5))

        # split into married and unmarried sets
        m = df[df["ever_married"] == "Yes"]
        nm = df[df["ever_married"] == "No"]

        # plot kde
        sns.kdeplot(
            data=m,
            x="age",
            fill=True,
            alpha=0.8,
            linewidth=0.7,
            ec="white",
            color=d_colours("green_3"),
        )
        sns.kdeplot(
            data=nm,
            x="age",
            fill=True,
            alpha=0.8,
            linewidth=0.7,
            ec="white",
            color=d_colours("d_green"),
        )

        plt.ylabel("Probability Density")
        plt.xlabel("Age (Years)")
        plt.legend(["Married", "Unmarried"], bbox_to_anchor=(1.1, 1.1))
        plt.xlim([0,100])

        st.pyplot(fig)

    st.markdown(
        """
    Similarly, a correlation was detected between marriage and stroke-risk. Upon further inspection, this can be explained by the age distrubtion between the married and unmarried subsets, with the median age of a non-married patient being 18, as compared to a median age of 54 for the married subset.
    """
    )


# -----------------------2.2 BMI---------------------------------------------------------------------
st.markdown("### 2.2. BMI")
st.markdown(
    """
The body mass index (BMI) is a measure that uses height and weight as a proxy health. The BMI calculation divides an adult's weight in kilograms by their height in metres squared. For example, A BMI of 25 means 25kg/m2. BMI ranges. For most adults, an ideal BMI is in the 18.5 to 24.9 range. For children and young people aged 2 to 18, the BMI calculation takes into account age and gender as well as height and weight.
The NHS defines BMI ranges as follows [[Source](https://www.nhs.uk/common-health-questions/lifestyle/what-is-the-body-mass-index-bmi/#:~:text=BMI%20ranges&text=below%2018.5%20%E2%80%93%20you're%20in,re%20in%20the%20obese%20range)]:
"""
)

bmi_df = pd.DataFrame(
    {
        "BMI": [
            "<18.5",
            "Between 18.5 & 24.9",
            "Between 25 and 29.9",
            "Between 30 and 39.9",
            ">40",
        ],
        "Classification": [
            "Underweight",
            "Healthy",
            "Overweight",
            "Obese",
            "Severely Obese",
        ],
    }
)

st.write(bmi_df)


st.markdown(
    """
The modal BMI for stroke occurence is approximately :green[30], and binning the dataset into the NHS categories, it is clear that :green[individuals who are overweight, obese or severely obese have far greater stroke incidence]. However, this does appear to taper, with overweight being the most at risk category.
"""
)

# create age buckets
def BMI_bucket(X):
    if X < 18.5:
        return "Underweight"

    if 18.5 <= X < 25:
        return "Healthy"

    if 25 <= X < 30:
        return "Overweight"

    if 30 <= X < 40:
        return "Obese"

    if 40 <= X:
        return "Severely Obese"


df["bmi_bucket"] = df["bmi"].apply(BMI_bucket)


if uploaded_file is not None:
    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.markdown("##### BMI-Stroke Distribution")

        fig = plt.figure(figsize=(8, 5))

        # plot kde graphs
        sns.kdeplot(
            data=stroke,
            x="bmi",
            fill=True,
            alpha=0.8,
            linewidth=0.7,
            ec="white",
            color=d_colours("green_3"),
        )
        sns.kdeplot(
            data=not_stroke,
            x="bmi",
            fill=True,
            alpha=0.8,
            linewidth=0.7,
            ec="white",
            color=d_colours("d_green"),
        )

        # formatting
        plt.ylabel("Probability Density")
        plt.legend(["Stroke", "Non-Stroke"])
        plt.xlim(0, 60)

        st.pyplot(fig)

    with col2:
        st.markdown("##### BMI-Stroke Risk (Binning)")

        fig = plt.figure(figsize=(8, 5))
        sns.barplot(
            data=df,
            x="bmi_bucket",
            y="stroke",
            order=["Underweight", "Healthy", "Overweight", "Obese", "Severely Obese"],
            palette={
                d_colours("d_green"),
                d_colours("green_1"),
                d_colours("green_2"),
                d_colours("green_3"),
            },
            errcolor="grey",
            ci=None,
        )

        plt.ylabel("Stroke Frequency")
        plt.xlabel("BMI Buckets")

        st.pyplot(fig)

#
##############################################2.3 Glucose Levels#########################################

st.markdown("### 2.3. Glucose Levels")

st.markdown(
    """

Glucose levels can be an indicator of stroke risk, adults with diabetes are 1.5 times more likely to have a stroke than people who don’t. Further, they are almost twice as likely to die from heart disease or stroke as people without diabetes [[Source](https://my.clevelandclinic.org/health/diseases/9812-diabetes-and-stroke)].

"""
)


if uploaded_file is not None:
    col1, col2 = st.columns(2, gap="small")

    with col1:

        st.markdown("##### Gluclose-Stroke Distribution")

        fig = plt.figure(figsize=(8, 5))

        # plot KDEs
        sns.kdeplot(
            data=stroke,
            x="avg_glucose_level",
            fill=True,
            alpha=0.8,
            linewidth=0.7,
            ec="white",
            color=d_colours("green_3"),
        )
        sns.kdeplot(
            data=not_stroke,
            x="avg_glucose_level",
            fill=True,
            alpha=0.8,
            linewidth=0.7,
            ec="white",
            color=d_colours("d_green"),
        )

        # formatting
        plt.ylabel("Probability Density")
        plt.xlabel("Average Glucose Levels (mg/dl)")
        plt.legend(["Stroke", "Non-Stroke"])

        st.pyplot(fig)

st.markdown(
    """
There are two distinct peaks in the above graph, align with the below glucose level classification chart. We do not know when the blood glucose levels were measured within the dataset, but in general, those with glucose levels below 200 mg/dl have a lower indicence of stroke than those with a blood glucose level above 200 mg/dl (diabetic). Therefore :green[diabetes could be an indicator of stroke risk].
"""
)

# image = Image.open("Images/glucose-chart.PNG")
# st.image(image, caption="Glucose Level Classifications")

glucose_classifcations_df = pd.DataFrame(
    {
        "Plasma Glucose Test": ["Random", "Fasting", "2 hour post-prandial"],
        "Normal": [
            "Below 200 mg/dl",
            "Below 100 mg/dl",
            "Below 140 mg/dl",
        ],
        "Prediabetes": [
            "NA",
            "100 to 125 mg/dl",
            "140 to 199mg/dl",
        ],
        "Diabetes": [
            "200 mg/dl or more",
            "126 mg/dl or more",
            "200 mg/dl or more",
        ],
    }
)

glucose_classifcations_df.set_index("Plasma Glucose Test", inplace=True)

st.write(glucose_classifcations_df)


##############################################2.4 Hypertension and Heart Disease#########################################

st.markdown("### 2.4. Hypertension and Heart Disease")
st.markdown(
    """

High blood pressure is a major risk factor for stroke. HBP adds to your heart’s workload and damages your arteries and organs over time. Compared to people whose blood pressure is normal, people with HBP are more likely to have a stroke.

About :green[87%] of strokes are caused by narrowed or clogged blood vessels in the brain that cut off the blood flow to brain cells. This is an ischemic stroke. High blood pressure causes damage to the inner lining of the blood vessels. This will narrow an artery. About 13% of strokes occur when a blood vessel ruptures in or near the brain.  This is a hemorrhagic stroke. Chronic HBP or aging blood vessels are the main causes of this type of stroke. HBP strains blood vessels. Over time, they no longer hold up to the pressure and rupture [[Source](https://www.stroke.org/-/media/Stroke-Files/Lets-Talk-About-Stroke/Risk-Factors/Stroke-and-High-Blood-Pressure-ucm_493407.pdf)].

"""
)


if uploaded_file is not None:

    col1, col2 = st.columns(2, gap="small")

    with col1:

        st.markdown("##### Hypertension Stroke Risk")

        fig = plt.figure(figsize=(8, 5))

        sns.barplot(
            data=df.drop(df.index[[3116]]),
            x="gender",
            y="stroke",
            hue="hypertension",
            palette={d_colours("d_green"), d_colours("green_3")},
            errcolor="grey",
            ci=None,
        )

        plt.ylabel("Stroke Frequency")
        plt.legend(
            labels=["No Hypertension", "Hypertension"], bbox_to_anchor=(1.1, 1.1)
        )
        st.pyplot(fig)

    with col2:

        st.markdown("##### Heart Disease Stroke Risk")

        fig = plt.figure(figsize=(8, 5))

        sns.barplot(
            data=df.drop(df.index[[3116]]),
            x="gender",
            y="stroke",
            hue="heart_disease",
            palette={d_colours("d_green"), d_colours("green_3")},
            errcolor="grey",
            ci=None,
        )

        plt.legend(
            labels=["No Hypertension", "Hypertension"], bbox_to_anchor=(1.1, 1.1)
        )
        plt.ylabel("Stroke Frequency")
        st.pyplot(fig)


# if uploaded_file is not None:

# st.markdown('##### Heart Disease vs Hyper Tension Confusion Matrix')

# fig = plt.figure(figsize = (8,7))

# sns.heatmap(confusion_matrix(df['hypertension'],df['heart_disease']),
#          annot=True,
#          fmt='d',
#          cmap=d_cmap,
#         xticklabels=['True','False'],
#         yticklabels=['True','False'],
#         )

# plt.ylabel('Hypertension')
# plt.xlabel('Heart Disease')

# st.pyplot(fig)


corr2 = pd.DataFrame(confusion_matrix(df["hypertension"], df["heart_disease"]))
corr2.columns = ["True", "False"]
corr2.index = ["True", "False"]

st.markdown(
    """
Evidently, :green[heart disease] and :green[high blood pressure] are :green[indicators of stroke risk], with a :green[greater prevelance of strokes] within the affected population. However, these factors are intrinsically linked, and when creating a confusion matrix of the occurence of heart disease and high blood pressure, a strong correlation is found. If an individual has heart disease, there is an :green[86%] chance that they also have high blood pressure.
"""
)

st.markdown("###### Confusion Matrix for Incidence of Heart Disease and Hypertension")
st.write(corr2)

st.markdown("### 2.4. Smoking")

##############################################2.5 Smoking#######################################################

st.markdown(
    """
Smoking tobacco increases your risk of having a stroke. Someone who smokes 20 cigarettes a day is six times more likely to have a stroke compared to a non-smoker. If you are a smoker, quitting will reduce your risk of stroke and a range of other diseases. If you live with a non-smoker, quitting will reduce their stroke risk too.

"""
)


if uploaded_file is not None:

    col1, col2 = st.columns(2, gap="small")

    with col1:

        st.markdown("##### Smoking Status Stroke Risk")

        fig = plt.figure(figsize=(8, 5))

        sns.barplot(
            data=df,
            x="smoking_status",
            y="stroke",
            order=["Unknown", "never smoked", "smokes", "formerly smoked"],
            palette={
                d_colours("d_green"),
                d_colours("green_1"),
                d_colours("green_2"),
                d_colours("green_3"),
            },
            errcolor="grey",
            ci=None,
        )

        plt.xlabel("Smoking Status")
        plt.ylabel("Stroke Frequency")

        st.pyplot(fig)

    with col2:

        st.markdown("##### Smoking Classification-Stroke Distribution ")

        fig = plt.figure(figsize=(8, 5))

        # plotting KDEs for each smoking category
        sns.kdeplot(
            data=df[df["smoking_status"] == "smokes"],
            x="age",
            fill=True,
            alpha=0.8,
            linewidth=0.7,
            ec="white",
            color=d_colours("green_1"),
        )
        sns.kdeplot(
            data=df[df["smoking_status"] == "formerly smoked"],
            x="age",
            fill=True,
            alpha=0.8,
            linewidth=0.7,
            ec="white",
            color=d_colours("green_3"),
        )
        sns.kdeplot(
            data=df[df["smoking_status"] == "never smoked"],
            x="age",
            fill=True,
            alpha=0.8,
            linewidth=0.7,
            ec="white",
            color=d_colours("d_green"),
        )
        sns.kdeplot(
            data=df[df["smoking_status"] == "Unknown"],
            x="age",
            fill=True,
            alpha=0.8,
            linewidth=0.7,
            ec="white",
            color=d_colours("green_2"),
        )

        # formatting
        plt.ylabel("Probability Density")
        plt.legend(["smokes", "formely smoked", "never smoked", "Unknown"])
        plt.xlim([0,100])
        st.pyplot(fig)

st.markdown(
    """
Former smokers appear to have the greatest risk of stroke, which is an unexpected result. However, the reason becomes clear when factoring age. Those in the never smoked category are generally older than those who smoke, as expected. Therefore, the :green[smoking categories are influenced by age].

"""
)

# calculate means and medians
df_smoke = pd.DataFrame(
    {
        "Smoking Status": ["Unknown", "never smoked", "smokes", "formely smoked"],
        "Mean Age": [
            df[df["smoking_status"] == "Unknown"]["age"].mean(),
            df[df["smoking_status"] == "never smoked"]["age"].mean(),
            df[df["smoking_status"] == "smokes"]["age"].mean(),
            df[df["smoking_status"] == "formerly smoked"]["age"].mean(),
        ],
        "Median Age": [
            df[df["smoking_status"] == "Unknown"]["age"].median(),
            df[df["smoking_status"] == "never smoked"]["age"].median(),
            df[df["smoking_status"] == "smokes"]["age"].median(),
            df[df["smoking_status"] == "formerly smoked"]["age"].median(),
        ],
    }
)

st.write(df_smoke.set_index("Smoking Status"))
