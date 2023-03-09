# streamlit packages
import streamlit as st
from streamlit_option_menu import option_menu
from utils.helpers import plot_metrics, load_pretrained_model, xgb_model
import joblib

# import basic packages
import numpy as np

# to split the datasets
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression



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


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "XGBoost",
        "Random Forests",
        "SVMs",
        "Neural Network",
        "Train a Logistic Regression",
    ]
)

with tab1:
    # model = xgb_model(X_train, y_train)
    # plot_metrics(
    #     metrics_list=["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"],
    #     model=model,
    #     x_test=X_test,
    #     y_test=y_test,
    # )
    with st.expander("More information"):
        st.image("Images/xgb_params.png")

    st.header("Results")
    st.write("not implemented")

with tab2:
    with st.expander("More information"):
        st.image("Images/rf_features.png")
    st.header("Results")
    model = load_pretrained_model("models/balanced_randomforest.joblib")
    plot_metrics(
        metrics_list=[
            "Confusion Matrix",
            "ROC Curve",
            "Precision-Recall Curve",
            "classification_report",
        ],
        model=model,
        x_test=X_test,
        y_test=y_test,
    )

with tab3:
    with st.expander("More information"):
        st.image("Images/svms.png")
    st.header("Results")
    model = load_pretrained_model("models/svm.pkl")
    scaler = load_pretrained_model("models/scaler.pkl")
    plot_metrics(
        metrics_list=[
            "Confusion Matrix",
            "ROC Curve",
            "Precision-Recall Curve",
            "classification_report",
        ],
        model=model,
        x_test=X_test,
        y_test=y_test,
        scaler=scaler,
    )

with tab4:
    with st.expander("More information"):
        st.image("Images/svms.png")
    st.header("Results")
    model = load_pretrained_model("models/nn.h5")
    plot_metrics(
        metrics_list=[
            "Confusion Matrix",
            "ROC Curve",
            "Precision-Recall Curve",
            "classification_report",
        ],
        model=model,
        x_test=X_test,
        y_test=y_test,
        scaler=scaler,
    )

with tab5:
    with st.expander("More information"):
        st.write("more info")

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

        st.header("Results")
        plot_metrics(
            metrics_list=[
                "Confusion Matrix",
                "ROC Curve",
                "Precision-Recall Curve",
                "classification_report",
            ],
            model=lrc,
            x_test=X_test,
            y_test=y_test,
        )
