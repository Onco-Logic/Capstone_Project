import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import random

# Load Blood Marker Models
rf_model = joblib.load("models/models/random_forest_model.pkl")
nb_model = joblib.load("models/models/naive_bayes_model.pkl")
lr_model = joblib.load("models/models/logistic_regression_model.pkl")
dt_model = joblib.load("models/models/decision_tree_model.pkl")
scaler = joblib.load("models/models/scaler.pkl")

# Load Blood Sample Dataset
df_blood = pd.read_csv("Data/Blood_samples_dataset_balanced_2.csv")

conversion_units = {
    "Glucose": [(70, 140),   "mg/dL"],
    "Cholesterol": [(125, 200),   "mg/dL"],
    "Hemoglobin": [(13.5, 17.5),   "g/dL"],
    "Platelets": [(150000, 450000),   "per microliter of blood"],
    "White Blood Cells": [(4000, 11000),   "per cubic millimeter of blood"],
    "Red Blood Cells": [(4.2, 5.4),   "million cells per microliter of blood"],
    "Hematocrit": [(38, 52),   "percentage"],
    "Mean Corpuscular Volume": [(80, 100),   "femtoliters"],
    "Mean Corpuscular Hemoglobin": [(27, 33),   "picograms"],
    "Mean Corpuscular Hemoglobin Concentration": [(32, 36),   "grams per deciliter"],
    "Insulin": [(5, 25),   "microU/mL"],
    "BMI": [(18.5, 24.9),   "kg/m^2"],
    "Systolic Blood Pressure": [(90, 120),   "mmHg"],
    "Diastolic Blood Pressure": [(60, 80),   "mmHg"],
    "Triglycerides": [(50, 150),   "mg/dL"],
    "HbA1c": [(4, 6),   "percentage"],
    "LDL Cholesterol": [(70, 130),   "mg/dL"],
    "HDL Cholesterol":[ (40, 60),   "mg/dL"],
    "ALT": [(10, 40),   "U/L"],
    "AST": [(10, 40),   "U/L"],
    "Heart Rate": [(60, 100),   "beats per minute"],
    "Creatinine": [(0.6, 1.2),   "mg/dL"],
    "Troponin": [(0, 0.04),   "ng/mL"],
    "C-reactive Protein": [(0, 3),   "mg/L"],
}
# Function to Predict Disease from Blood Markers
def predict_disease_from_markers(user_input, model):
    try:
        user_data = np.array(user_input).reshape(1, -1)
        user_data_scaled = scaler.transform(user_data)
        prediction = model.predict(user_data_scaled)[0]
        return prediction
    except Exception as e:
        return f"Error: {str(e)}"


# Function to Recommend the Best Model
def recommend_model(user_input):
    try:
        user_data = np.array(user_input).reshape(1, -1)
        user_data_scaled = scaler.transform(user_data)

        # Check model confidence or probability
        rf_pred_prob = rf_model.predict_proba(user_data_scaled)[:, 1] if hasattr(rf_model, "predict_proba") else None
        nb_pred_prob = nb_model.predict_proba(user_data_scaled)[:, 1] if hasattr(nb_model, "predict_proba") else None
        lr_pred_prob = lr_model.predict_proba(user_data_scaled)[:, 1] if hasattr(lr_model, "predict_proba") else None
        dt_pred_prob = lr_model.predict_proba(user_data_scaled)[:, 1] if hasattr(dt_model, "predict_proba") else None

        # Compute Model Confidence
        confidence_scores = {
            "Random Forest": np.mean(rf_pred_prob) if rf_pred_prob is not None else 0,
            "Naive Bayes": np.mean(nb_pred_prob) if nb_pred_prob is not None else 0,
            "Logistic Regression": np.mean(lr_pred_prob) if lr_pred_prob is not None else 0
        }

        # Choose the best model based on confidence
        best_model = max(confidence_scores, key=confidence_scores.get)
        return best_model, confidence_scores[best_model]

    except Exception as e:
        return "Unknown", 0


# Explainability using SHAP
def explain_prediction(user_input, model):
    try:
        user_data = np.array(user_input).reshape(1, -1)
        user_data_scaled = scaler.transform(user_data)
        explainer = shap.Explainer(model, user_data_scaled)
        shap_values = explainer(user_data_scaled)

        plt.figure(figsize=(8, 6))
        shap.waterfall_plot(shap_values[0])
        st.pyplot(plt)
    except Exception as e:
        st.write(f"Explainability Error: {str(e)}")

def normalized_data():
    randomize = st.checkbox("🎲 Randomize Realistic Data")
    if "randomized_values" not in st.session_state:
        st.session_state.randomized_values = {}

    if randomize and st.button("🔄 Refresh Random Data"):
        st.session_state.randomized_values.clear()

    # Blood Marker Inputs
    marker_values = []
    generated_values = {}

    for col in df_blood.columns:
        if col != 'Disease':
            if randomize:
                if col not in st.session_state.randomized_values:
                    st.session_state.randomized_values[col] = round(random.uniform(0, 1), 2)
                value = st.session_state.randomized_values[col]
            else:
                value = st.number_input(f"{col}", min_value=0.0, max_value=1.0, format="%.2f")

            marker_values.append(value)
            generated_values[col] = value

    # Display Randomized Data
    if randomize:
        st.subheader("🔢 Randomized Blood Marker Values")
        st.dataframe(pd.DataFrame([generated_values]))

    # Recommend the Best Model
    recommended_model, confidence = recommend_model(marker_values)
    st.info(f"🔍 **Recommended Model After You've Entered All Values:** {recommended_model} (Confidence: {confidence:.2f})")

    # Model Selection
    model_choice = st.selectbox("Select Prediction Model", ["Random Forest", "Naive Bayes", "Logistic Regression", "Decision Tree"])
    model_map = {
        "Random Forest": rf_model,
        "Naive Bayes": nb_model,
        "Logistic Regression": lr_model,
        "Decision Tree": dt_model
    }

    if st.button("Analyze Markers"):
        selected_model = model_map[model_choice]
        prediction = predict_disease_from_markers(marker_values, selected_model)
        st.subheader("🧪 Predicted Disease:")
        st.write(prediction)


def raw_data():
    # a dictionary with mean for Healthy in case the patient does not have certain markers
    user_markers = df_blood[df_blood.Disease == 'Healthy'].drop('Disease', axis=1)
    healthy_mean = user_markers.mean()
    st.write("Provide your blood markers to see what they can be indicative of.")
    for k, v in conversion_units.items():
        value = st.number_input(f"{k} ({v[1]})", value=None)
        if value:
            normalized = (value-conversion_units[k][0][0])/(conversion_units[k][0][1]-conversion_units[k][0][0])
            healthy_mean[k] = normalized
    recommended_model, confidence = recommend_model(healthy_mean)
    st.info(f"🔍 **Recommended Model After You've Entered Values:** {recommended_model} (Confidence: {confidence:.2f})")
    model_choice = st.selectbox("Select Prediction Model ", ["Random Forest", "Naive Bayes", "Logistic Regression"])
    model_map = {
        "Random Forest": rf_model,
        "Naive Bayes": nb_model,
        "Logistic Regression": lr_model, 
        "Decision Tree": dt_model
    }
    if st.button("Analyze Markers "):
        selected_model = model_map[model_choice]
        prediction = predict_disease_from_markers(healthy_mean, selected_model)
        st.subheader("🧪 Predicted Disease:")
        st.write(prediction)

# Streamlit UI for Disease Prediction
def disease_prediction():
    st.title("📊 Disease Prediction from Blood Markers")
    tab1, tab2 = st.tabs(['Normalized Data', 'Raw Data'])
    # Randomization Toggle
    with tab1:
        normalized_data()
    with tab2:
        raw_data()    

# Run the Disease Prediction Page
if __name__ == "__main__":
    disease_prediction()
