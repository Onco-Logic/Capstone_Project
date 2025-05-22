import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import shap
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.figure_factory as ff

# Load dataset
DATASET_PATH1 = pd.read_csv("Data/Blood_samples_dataset_balanced_2.csv")
DATASET_PATH2 = pd.read_csv("Data/blood_samples_dataset_test.csv")

df = pd.concat([DATASET_PATH1, DATASET_PATH2], ignore_index=True)
df = df.replace('Thalasse', 'Thalassemia').replace('Heart Di', 'Heart Disease')
# Sidebar: Feature Selection (Removed Target Selection)
st.sidebar.header("🔍 Feature Selection")
available_features = df.columns[df.columns != "Disease"]
default_features = [feature for feature in ['Glucose', 'Cholesterol', 'Hemoglobin', 'Platelets', 'White Blood Cells'] if feature in available_features]
selected_features = st.sidebar.multiselect(
    "Select features for visualization:",
    available_features,
    default=default_features
)

# Sidebar: Correlation Thresholding
st.sidebar.header("🔗 Correlation Filtering")
corr_threshold = st.sidebar.slider(
    "Minimum correlation with Disease:", 0.0, 1.0, 0.0, 0.05
)

# Encode "Disease" into numeric values before correlation computation
df_encoded = df.copy()
df_encoded["Disease"] = df_encoded["Disease"].astype("category").cat.codes  # Convert to numeric labels

# Compute correlation
correlation_matrix = df_encoded.corr()["Disease"].abs()

# Apply correlation filtering
if corr_threshold > 0:
    filtered_features = correlation_matrix[correlation_matrix > corr_threshold].index.tolist()
    selected_features = [f for f in selected_features if f in filtered_features]

# Sidebar: Feature Scaling
scaling_option = st.sidebar.radio("📏 Feature Scaling", ["None", "Standard Scaling", "Min-Max Scaling"])

# Sidebar: Train on Full Dataset or Subset
train_size = st.sidebar.slider("Training Set Size (%)", 10, 100, 80, 10) / 100

# Sidebar: Model Selection
model_type = st.sidebar.selectbox("Choose Model", ["Random Forest", "Naive Bayes", "Logistic Regression", "Decision Tree"])

# Sidebar: Hyperparameter Customization (Random Forest)
if model_type == "Random Forest":
    n_estimators = st.sidebar.slider("🌳 Number of Trees (Random Forest)", 50, 500, 100, 50)
else:
    n_estimators = 100  # Default for non-RF models

# Data Preparation
X = df[selected_features]
y = df["Disease"]
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# Apply selected scaling
if scaling_option == "Standard Scaling":
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
elif scaling_option == "Min-Max Scaling":
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - train_size), random_state=42)

# Model Initialization
if model_type == "Random Forest":
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
elif model_type == "Naive Bayes":
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
elif model_type == "Logistic Regression":
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
else:
    import joblib
    model =joblib.load("models/models/decision_tree_model.pkl")

# Train the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Load and Display Saved Graphs from train_models.py
st.subheader("📊 Model Insights from Disease Prediction Training")

# Pair Plot of Selected Blood Markers
if selected_features:
    st.subheader("🔍 Pair Plot of Blood Markers")
    df_subset = df[selected_features + ["Disease"]]
    fig = sns.pairplot(df_subset, hue="Disease", diag_kind="kde")
    st.pyplot(fig)

# Feature Importance (Only for Random Forest)
if model_type == "Random Forest":
    st.subheader("📊 Feature Importance of Blood Markers")
    feature_importances = model.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]
    sorted_features = X.columns[sorted_idx]
    sorted_importances = feature_importances[sorted_idx]

    fig = px.bar(x=sorted_features, y=sorted_importances,
                 title="Feature Importance for Classification",
                 labels={"x": "Feature", "y": "Importance"})
    st.plotly_chart(fig)

# SHAP Explainability (For Random Forest & Logistic Regression)
if model_type in ["Random Forest", "Logistic Regression"]:
    st.subheader("📊 SHAP Explainability - Why was this predicted?")

    # Initialize SHAP Explainer
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    # Convert to NumPy array if necessary
    if isinstance(shap_values, list):
        shap_values_array = np.array(shap_values[0])
    else:
        shap_values_array = shap_values.values if hasattr(shap_values, "values") else np.array(shap_values)

    feature_names = list(X.columns)
    X_test_array = np.array(X_test)

    # Fix Dimension Mismatch
    if shap_values_array.ndim == 3:
        shap_values_array = shap_values_array[:, :, 0]

    if shap_values_array.shape[1] != X_test_array.shape[1]:
        shap_values_array = shap_values_array[:, :X_test_array.shape[1]]

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values_array, X_test_array, feature_names=feature_names, plot_type="bar", show=False)
    st.pyplot(fig)


if model_type=='Decision Tree':
    from tempfile import NamedTemporaryFile
    from streamlit.components.v1 import html
    from supertree import SuperTree
    temp_dir = os.getcwd()

    super_tree = SuperTree(model, X_train, y_train, feature_names=["Glucose","Cholesterol","Hemoglobin","Platelets","White Blood Cells"], target_names=label_encoder.classes_)
    st.subheader("🌲 Decision Tree")
    with NamedTemporaryFile(suffix=".html", dir=temp_dir, delete=True) as f:
        super_tree.save_html(f.name)
        f.seek(0) 
        html(f.read(), height=800)
        

# Correlation Matrix
st.subheader("📊 Correlation Matrix")
df_numeric = df.select_dtypes(include=['number'])
if not df_numeric.empty:
    corr_matrix = df_numeric.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Disease Distribution Chart
st.subheader("📊 Disease Distribution")
fig = px.bar(df["Disease"].value_counts(), x=df["Disease"].unique(), y=df["Disease"].value_counts(),
             labels={"x": "Disease Type", "y": "Count"}, title="Distribution of Diseases")
st.plotly_chart(fig)

# Model Performance Metrics
st.subheader("📊 Model Performance")
st.write(f"**Model Selected:** {model_type}")
st.write(f"**Training Set Size:** {train_size * 100:.0f}%")
st.write(f"**Accuracy:** {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig = ff.create_annotated_heatmap(
    z=cm,
    x=[f"Pred {c}" for c in np.unique(y)],
    y=[f"Actual {c}" for c in np.unique(y)],
    colorscale="Viridis"
)
st.plotly_chart(fig)

st.write("Confusion Matrix Heatmap (Higher diagonal values mean better accuracy)")
