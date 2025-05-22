import joblib
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page Title
st.title("📊 Survival Model Performance Comparison")

# User selects survival period
selected_survival_period = st.sidebar.selectbox("📆 Select Survival Period", ["5-Year", "10-Year", "15-Year"])

# Explicit Mapping of Test Files (Prevents Wrong Filenames)
TEST_FILES = {
    "5-Year": "models/models/data/y_test_Survival_5Y.csv",
    "10-Year": "models/models/data/y_test_Survival_10Y.csv",
    "15-Year": "models/models/data/y_test_Survival_15Y.csv"
}

X_TEST_PATH = "models/models/data/X_test.csv"
Y_TEST_PATH = TEST_FILES[selected_survival_period]  # Explicit selection

# Ensure test data files exist
try:
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)
except FileNotFoundError as e:
    st.error(f"❌ Missing test file: {e}")
    st.stop()

# Explicit Mapping of Model Paths (Added New Models)
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

# Evaluate Models
results = []
sample_predictions = {}

def evaluate_model(model_path, X_test, y_test):
    try:
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Ensure we do not sample more than available test samples
        num_samples = min(len(y_test), 10)  # Take min of 10 or available samples

        if num_samples > 0:
            sample_indices = np.random.choice(len(y_test), num_samples, replace=False)
            sample_actual = y_test.iloc[sample_indices].values.flatten()
            sample_predicted = y_pred[sample_indices]

            # Compute accuracy percentage for each sample
            accuracy_percentages = 100 - (np.abs(sample_actual - sample_predicted) / sample_actual) * 100
        else:
            sample_actual, sample_predicted, accuracy_percentages = [], [], []

        return mae, mse, r2, sample_actual, sample_predicted, accuracy_percentages
    except FileNotFoundError:
        return None, None, None, None, None, None

for model_name, periods in MODEL_PATHS.items():
    model_path = periods[selected_survival_period]
    mae, mse, r2, sample_actual, sample_predicted, accuracy_percentages = evaluate_model(model_path, X_test, y_test)

    if mae is not None:
        results.append({
            "Model": model_name,
            "Period": selected_survival_period,
            "MAE": mae,
            "MSE": mse,
            "R2 Score": r2
        })
        sample_predictions[model_name] = (sample_actual, sample_predicted, accuracy_percentages)
    else:
        st.warning(f"⚠️ Model {model_name} for {selected_survival_period} not found! Please train the model.")

# Convert results to DataFrame
results_df = pd.DataFrame(results)

if results_df.empty:
    st.error("❌ No valid models found. Please train the models first.")
    st.stop()

# Display Table
st.subheader("📋 Model Performance Metrics")
st.dataframe(results_df)

# Visualization
st.subheader("📊 Performance Comparison Charts")

# Bar Chart for R² Score
st.write("### R² Score Comparison (Higher is Better)")
fig, ax = plt.subplots()
results_df.pivot(index="Period", columns="Model", values="R2 Score").plot(kind='bar', ax=ax)
st.pyplot(fig)

# Line Chart for MAE & MSE
st.write("### Mean Absolute Error (Lower is Better)")
fig, ax = plt.subplots()
results_df.pivot(index="Period", columns="Model", values="MAE").plot(kind='line', marker='o', ax=ax)
st.pyplot(fig)

st.write("### Mean Squared Error (Lower is Better)")
fig, ax = plt.subplots()
results_df.pivot(index="Period", columns="Model", values="MSE").plot(kind='line', marker='o', ax=ax)
st.pyplot(fig)

# Best Model Selection
best_model = results_df.sort_values(by="R2 Score", ascending=False).iloc[0]
st.success(
    f"🎯 The best performing model is **{best_model['Model']}** for **{best_model['Period']}** period with an R² score of **{best_model['R2 Score']:.2f}**"
)

# **Additional Implementation: Show 10 Actual vs. Predicted Survival Rates with Accuracy Percentage**
st.subheader("🔍 Sample Predictions (Actual vs. Predicted)")

for model_name, (sample_actual, sample_predicted, accuracy_percentages) in sample_predictions.items():
    if len(sample_actual) > 0:
        st.write(f"📌 **{model_name} - {selected_survival_period}**")

        # 🔹 Create DataFrame for Display
        comparison_df = pd.DataFrame({
            "Actual Survival Rate": sample_actual,
            "Predicted Survival Rate": sample_predicted,
            "Accuracy (%)": accuracy_percentages
        })

        st.table(comparison_df)
    else:
        st.warning(f"⚠️ Not enough test samples available to display for {model_name}.")