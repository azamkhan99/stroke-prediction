from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    classification_report,
)
import streamlit as st
import joblib
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
# from tensorflow import keras
# from keras.models import load_model

def pr_comparison(plot_type, x_test, y_test, y_pred):

    if plot_type == 'pr_curve':


        fig, ax = plt.subplots()
        PrecisionRecallDisplay.from_estimator(load_pretrained_model('models/trained_lr.joblib'),x_test, y_test, pos_label=1, ax=ax)
        PrecisionRecallDisplay.from_estimator(load_pretrained_model('models/balanced_randomforest.joblib'),x_test, y_test, pos_label=1, ax=ax)
        PrecisionRecallDisplay.from_estimator(load_pretrained_model('models/xgb.joblib'),x_test, y_test, pos_label=1, ax=ax)
        PrecisionRecallDisplay.from_estimator(load_pretrained_model('models/svm.pkl'),load_pretrained_model('models/scaler.pkl').transform(x_test), y_test, pos_label=1, ax=ax)
        PrecisionRecallDisplay.from_predictions(y_test, y_pred, pos_label=1, ax=ax, name ="Neural Network")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),fancybox=True, shadow=True)
        st.pyplot()

    if plot_type == 'roc_curve':
        fig, ax = plt.subplots()
        RocCurveDisplay.from_estimator(load_pretrained_model('models/trained_lr.joblib'),x_test, y_test, pos_label=1, ax=ax)
        RocCurveDisplay.from_estimator(load_pretrained_model('models/balanced_randomforest.joblib'),x_test, y_test, pos_label=1, ax=ax)
        RocCurveDisplay.from_estimator(load_pretrained_model('models/xgb.joblib'),x_test, y_test, pos_label=1, ax=ax)
        RocCurveDisplay.from_estimator(load_pretrained_model('models/svm.pkl'),load_pretrained_model('models/scaler.pkl').transform(x_test), y_test, pos_label=1, ax=ax)
        RocCurveDisplay.from_predictions(y_test, y_pred, pos_label=1, ax=ax, name ="Neural Network")
        ident = [0.0, 1.0]
        plt.plot(ident,ident, color='r', ls='--', label='Random Classification')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),fancybox=True, shadow=True)
        st.pyplot()

def plot_nn_metrics(metrics_list, model, y_test):

    y_pred = model
    col1, col2 = st.columns(2, gap="small")
    with col1:

        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            RocCurveDisplay.from_predictions(y_test, y_pred)
            st.pyplot()

        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, colorbar=False, cmap="YlGn")
            st.pyplot()

    with col2:
        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            PrecisionRecallDisplay.from_predictions(y_test, y_pred)
            st.pyplot()

        if "classification_report" in metrics_list:
            st.subheader("Classification Report")
            report = classification_report(
                y_test,
                y_pred,
                target_names=["No Stroke", "Stroke"],
                output_dict=True,
            )
            report_df = pd.DataFrame(report)
            st.dataframe(report_df.T, use_container_width=True)

def plot_metrics(metrics_list, model, x_test, y_test, scaler=None):
        if scaler:
            x_test = scaler.transform(x_test)

        col1, col2 = st.columns(2, gap="small")
        with col1:

            if "ROC Curve" in metrics_list:
                st.subheader("ROC Curve")
                RocCurveDisplay.from_estimator(model, x_test, y_test)
                st.pyplot()

            if "Confusion Matrix" in metrics_list:
                st.subheader("Confusion Matrix")
                ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, colorbar=False, cmap="YlGn")
                st.pyplot()

        with col2:
            if "Precision-Recall Curve" in metrics_list:
                st.subheader("Precision-Recall Curve")
                PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
                st.pyplot()

            if "classification_report" in metrics_list:
                st.subheader("Classification Report")
                report = classification_report(
                    y_test,
                    model.predict(x_test),
                    target_names=["No Stroke", "Stroke"],
                    output_dict=True,
                )
                report_df = pd.DataFrame(report)
                st.dataframe(report_df.T, use_container_width=True)


def load_pretrained_model(filename):
    if filename.split(".")[1] == "joblib":
        model = joblib.load(filename)
    elif filename.split(".")[1] == "pkl":
        model = pickle.load(open(filename, "rb"))
    elif filename.split(".")[1] == "h5":
        model = load_model("model.h5")
    return model


def xgb_model(X_train, y_train):
    model = xgb.XGBClassifier(
        scale_pos_weight=17,
        n_estimators=200,
        learning_rate=0.01,
        colsample_bytree=1.0,
        gamma=1.5,
        max_depth=4,
        min_child_weight=10,
        subsample=0.6,
    )
    model.fit(X_train, y_train)
    return model


def present_prediction(model, input, scaler=None):
    if scaler:
        input = scaler.transform(input)
    pred = model.predict(input)
    if pred == 0:
        st.header("Prediction: No Stroke!")
    elif pred == 1:
        st.header("Prediction: STROKE - Further consultation advised.")
