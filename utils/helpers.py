from sklearn.metrics import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    classification_report,
)
import streamlit as st
import joblib
import pickle
import pandas as pd


def plot_metrics(metrics_list, model, x_test, y_test, scaler=None):
    if scaler:
        x_test = scaler.transform(x_test)

    col1, col2 = st.columns(2, gap="small")
    with col1:

        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test, colorbar=False, cmap="YlGn")
            st.pyplot()

    with col2:
        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
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


def present_prediction(model, input, scaler=None):
    if scaler:
        input = scaler.transform(input)
    pred = model.predict(input)
    if pred == 0:
        st.header("Prediction: No Stroke!")
    elif pred == 1:
        st.header("Prediction: STROKE")
