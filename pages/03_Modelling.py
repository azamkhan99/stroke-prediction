# streamlit packages
import streamlit as st
from streamlit_option_menu import option_menu
from utils.helpers import plot_metrics, load_pretrained_model, xgb_model
import joblib

# import basic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# user-interactive visulization with plotly
import plotly.express as px
import plotly.figure_factory as ff

# to split the datasets
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# feature engineering imputation
from feature_engine import imputation as mdi

# for one hot encoding with sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, plot_confusion_matrix


# for encoding with feature-engine
from feature_engine.encoding import OneHotEncoder as fe_OneHotEncoder
from feature_engine.encoding import OrdinalEncoder as fe_OrdinalEncoder

# for Q-Q plots
import scipy.stats as stats

# read images
from PIL import Image


st.title("Modelling")

relevant_columns = [
    "age",
    "hypertension",
    "heart_disease",
    "gender_Female",
    "gender_Male",
    "ever_married_Yes",
    "work_type_Private",
    "work_type_Self-employed",
    "work_type_children",
    "work_type_Govt_job",
    "Residence_type_Rural",
    "smoking_status_formerly smoked",
    "smoking_status_never smoked",
    "smoking_status_smokes",
    "avg_glucose_level_ranked",
    "bmi",
]
X_train = st.session_state.X_train
X_test = st.session_state.X_test
y_train = st.session_state.y_train
y_test = st.session_state.y_test

X_train = X_train[relevant_columns]
X_test = X_test[relevant_columns]


tab1, tab2, tab3, tab4 = st.tabs(
    ["XGBoost", "Random Forests", "SVMs", "Train a Logistic Regression"]
)

with tab1:
    # model = xgb_model(X_train, y_train)
    # plot_metrics(
    #     metrics_list=["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"],
    #     model=model,
    #     x_test=X_test,
    #     y_test=y_test,
    # )
    st.write("not implemented")

with tab2:
    model = load_pretrained_model("models/balanced_randomforest.joblib")
    plot_metrics(
        metrics_list=["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"],
        model=model,
        x_test=X_test,
        y_test=y_test,
    )

with tab3:
    model = load_pretrained_model("models/svm.pkl")
    scaler = load_pretrained_model("models/scaler.pkl")
    plot_metrics(
        metrics_list=["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"],
        model=model,
        x_test=X_test,
        y_test=y_test,
        scaler=scaler,
    )


with tab4:

    # select hyperparams to tune
    st.write("Choose hyperparameters for grid search")
    pen = st.multiselect("Penalty Hyperparameters", ["l1", "l2", "elasticnet"])
    c = st.multiselect("C Hyperparameters", [1, 10, 100, 1000])
    ratio = st.multiselect("L1 ratio (for elasticnet)", np.linspace(0, 1, 5))

    if pen and c and ratio:
        parameters = [{"penalty": pen}, {"C": c}, {"l1_ratio": ratio}]
        logreg = LogisticRegression(solver="saga", class_weight="balanced")
        grid_search = GridSearchCV(estimator=logreg, param_grid=parameters)

        gs = grid_search.fit(X_train, y_train)
        lrc = gs.best_estimator_
        joblib.dump(lrc, "models/trained_lr.joblib")
        st.write(gs.best_params_)

        y_pred = lrc.predict(X_test)

        plot_metrics(
            metrics_list=["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"],
            model=lrc,
            x_test=X_test,
            y_test=y_test,
        )
