import joblib
import pandas as pd
import streamlit as st
import numpy as np

# Page Title
st.title("🧬 Survival Rate Prediction")

# Load Available Models (Now Includes Boosting & Stacking)
MODEL_PATHS = {
    "Random Forest": {
        "5-Year": "models/random_forest_Survival_5Y.pkl",
        "10-Year": "models/random_forest_Survival_10Y.pkl",
        "15-Year": "models/random_forest_Survival_15Y.pkl"
    },
    "Linear Regression": {
        "5-Year": "models/linear_regression_Survival_5Y.pkl",
        "10-Year": "models/linear_regression_Survival_10Y.pkl",
        "15-Year": "models/linear_regression_Survival_15Y.pkl"
    },
    "Support Vector Machine": {
        "5-Year": "models/support_vector_machine_Survival_5Y.pkl",
        "10-Year": "models/support_vector_machine_Survival_10Y.pkl",
        "15-Year": "models/support_vector_machine_Survival_15Y.pkl"
    },
    "XGBoost": {
        "5-Year": "models/xgboost_Survival_5Y.pkl",
        "10-Year": "models/xgboost_Survival_10Y.pkl",
        "15-Year": "models/xgboost_Survival_15Y.pkl"
    },
    "LightGBM": {
        "5-Year": "models/lightgbm_Survival_5Y.pkl",
        "10-Year": "models/lightgbm_Survival_10Y.pkl",
        "15-Year": "models/lightgbm_Survival_15Y.pkl"
    },
    "CatBoost": {
        "5-Year": "models/catboost_Survival_5Y.pkl",
        "10-Year": "models/catboost_Survival_10Y.pkl",
        "15-Year": "models/catboost_Survival_15Y.pkl"
    },
    "Stacking Ensemble": {
        "5-Year": "models/stacking_ensemble_Survival_5Y.pkl",
        "10-Year": "models/stacking_ensemble_Survival_10Y.pkl",
        "15-Year": "models/stacking_ensemble_Survival_15Y.pkl"
    }
}

# User Model Selection
selected_model_name = st.sidebar.selectbox("📌 Select Prediction Model", list(MODEL_PATHS.keys()))
selected_survival_period = st.sidebar.selectbox("📆 Select Survival Period", ["5-Year", "10-Year", "15-Year"])
MODEL_PATH = MODEL_PATHS[selected_model_name][selected_survival_period]

# Ensure Model Exists Before Loading
try:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load("models/survival_encoder.pkl")
    scaler = joblib.load("models/survival_scaler.pkl")  # Load scaler for consistency

    # User Inputs (Expanded Selection Options)
    disease = st.selectbox("🔬 Select Disease", ["Diabetes", "Thalassemia", "Aplastic Anemia", "TTP"])
    gender = st.selectbox("⚥ Select Gender", ["Male", "Female"])
    age_group = st.selectbox("🎂 Select Age Group", ["0-18", "19-39", "40-59", "60+", "80+"])

    # Encode & Scale Inputs
    input_df = pd.DataFrame([[disease, gender, age_group]], columns=["Disease", "Gender", "Age_Group"])
    user_input_encoded = encoder.transform(input_df)  # Encode categorical data
    user_input_scaled = scaler.transform(user_input_encoded)  # Apply the saved scaler

    # Make Prediction
    if st.button("Predict Survival Rate"):
        survival_pred = model.predict(user_input_scaled)

        # Display Results
        st.subheader(f"📊 Predicted Survival Rate ({selected_model_name}, {selected_survival_period}):")
        st.write(f"✅ **{selected_survival_period} Survival Probability:** {survival_pred[0]:.2f}%")

        # Visualization
        st.subheader("📈 Survival Probability Chart")
        df_chart = pd.DataFrame({
            "Years": [selected_survival_period],
            "Survival Probability (%)": [survival_pred[0]]
        })
        st.bar_chart(df_chart.set_index("Years"))

except FileNotFoundError:
    st.error(f"❌ {selected_model_name} Model for {selected_survival_period} Not Found! Train the model first using train_survival_models.py.")