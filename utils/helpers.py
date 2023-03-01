from sklearn.metrics import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)
import streamlit as st
import joblib
import pickle
from xgboost import XGBClassifier
import xgboost as xgb


def plot_metrics(metrics_list, model, x_test, y_test, scaler=None):
    if scaler:
        st.write("SCALED")
        x_test = scaler.transform(x_test)
        st.write(x_test)
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
    if filename.split(".")[1] == "joblib":
        model = joblib.load(filename)
    elif filename.split(".")[1] == "pkl":
        model = pickle.load(open(filename, "rb"))
    return model


def xgb_model(X_train, y_train):
    model = XGBClassifier(
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


def present_prediction(model, input):
    pred = model.predict(input)
    if pred == 0:
        st.header("Prediction: No Stroke!")
    elif pred == 1:
        st.header("Prediction: STROKE")
