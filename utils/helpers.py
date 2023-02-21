from sklearn.metrics import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)
import streamlit as st
import joblib


def plot_metrics(metrics_list, model, x_test, y_test):
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x_test, y_test)
        st.pyplot()
    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, x_test, y_test)
        st.pyplot()
    if "Precision-Recall Curve" in metrics_list:
        st.subheader("Precision-Recall Curve")
        plot_precision_recall_curve(model, x_test, y_test)
        st.pyplot()


def load_pretrained_model(filename):
    model = joblib.load(filename)
    return model


def present_prediction(model, input):
    pred = model.predict(input)
    if pred == 0:
        st.header("Prediction: No Stroke!")
    elif pred == 1:
        st.header("Prediction: STROKE")
