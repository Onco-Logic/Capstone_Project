import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

# Load dataset
DATASET_PATH1 = pd.read_csv("Blood_samples_dataset_balanced_2.csv")
DATASET_PATH2 = pd.read_csv("blood_samples_dataset_test.csv")

df = pd.concat([DATASET_PATH1, DATASET_PATH2], ignore_index=True)
df = df.replace('Thalasse', 'Thalassemia').replace('Heart Di', 'Heart Disease')
# Prepare data
X = df.drop(columns=["Disease"])
y = df["Disease"]

# Sidebar: Model Selection
st.sidebar.header("📥 Train and Compare Models")
model_type = st.sidebar.selectbox("Choose Model", ["Random Forest", "Naive Bayes", "Logistic Regression", "Decision Tree"])

# Normalize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier()
}

# Train the Selected Model
st.subheader(f"🔄 Training **{model_type}** on Dataset...")
model = models[model_type]
model.fit(X_train, y_train)
y_pred_new = model.predict(X_test)

# Compute Classification Metrics
new_accuracy = accuracy_score(y_test, y_pred_new)
accuracy_percentage = new_accuracy * 100
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred_new, average="weighted")

# Show Enhanced Accuracy Comparison
st.subheader("📊 **Disease Prediction Accuracy Comparison**")

st.write(f"""
✅ **{model_type} Model Trained Successfully!**
- **Accuracy:** {new_accuracy:.4f} (**{accuracy_percentage:.2f}%**)
- **Precision:** {precision:.4f}
- **Recall:** {recall:.4f}
- **F1-Score:** {f1_score:.4f}
""")

# Bar Chart for Model Accuracy Comparison
accuracy_data = pd.DataFrame({
    "Model": [model_type],
    "Accuracy (%)": [accuracy_percentage]
})

fig = px.bar(accuracy_data, x="Model", y="Accuracy (%)", text="Accuracy (%)",
             title=f"📊 Accuracy of {model_type}", color="Accuracy (%)", color_continuous_scale="Blues")
st.plotly_chart(fig)

# Confusion Matrix
st.subheader("🧩 **Confusion Matrix & Class-Level Accuracy**")

cm = confusion_matrix(y_test, y_pred_new)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {model_type}")
st.pyplot(plt)

# **NEW FEATURE 1: Show 10 Actual vs. Predicted Values**
st.subheader("🔍 Sample Predictions (Actual vs. Predicted)")

# Select 10 random values for actual vs. predicted comparison
num_samples = min(len(y_test), 10)
sample_indices = np.random.choice(len(y_test), num_samples, replace=False)
sample_actual = y_test.iloc[sample_indices].values
sample_predicted = y_pred_new[sample_indices]

# Create DataFrame for Display
comparison_df = pd.DataFrame({
    "Actual Disease": sample_actual,
    "Predicted Disease": sample_predicted
})

st.table(comparison_df)

# **NEW FEATURE 2: Randomly Generated Realistic Samples vs Predicted Disease & Confidence**
st.subheader("🎲 Random Generated Samples vs. Predicted Disease & Confidence")

# **Add a Refresh Button for New Random Samples**
if st.button("🔄 Refresh Random Data"):
    st.session_state.random_samples = np.random.uniform(0, 1, (10, X.shape[1]))  # Generate new values

# Ensure random samples exist in session state
if "random_samples" not in st.session_state:
    st.session_state.random_samples = np.random.uniform(0, 1, (10, X.shape[1]))  # Initial generation

# Scale new random samples
random_samples_scaled = scaler.transform(st.session_state.random_samples)

# Make predictions
random_pred = model.predict(random_samples_scaled)

# If model supports probability scores, get confidence
if hasattr(model, "predict_proba"):
    confidence_scores = model.predict_proba(random_samples_scaled).max(axis=1)
    mean_confidence_score = np.mean(confidence_scores)  # Compute average confidence score
else:
    confidence_scores = ["N/A"] * 10
    mean_confidence_score = "N/A"

# Create DataFrame for Display
random_comparison_df = pd.DataFrame({
    "Predicted Disease": random_pred,
    "Confidence Score": confidence_scores
})

st.table(random_comparison_df)

# **Final Messages Based on Performance**
## **Actual vs. Predicted Accuracy Message**
if accuracy_percentage >= 95:
    st.success(f"🚀 **Excellent Accuracy!** {model_type} performed exceptionally well on test data with {accuracy_percentage:.2f}% accuracy.")
elif accuracy_percentage >= 85:
    st.info(f"👍 **Good Performance!** {model_type} achieved a strong accuracy of {accuracy_percentage:.2f}% on test data.")
else:
    st.warning(f"⚠️ **Needs Improvement**: {model_type} had an accuracy of {accuracy_percentage:.2f}% on test data. Consider improvements.")

## **Randomly Generated vs. Predicted Confidence Message**
if mean_confidence_score != "N/A":
    if mean_confidence_score >= 0.9:
        st.success(f"🎯 **High Confidence!** Based on the 10 samples, the model has strong confidence in the random data predictions with an average confidence score of {mean_confidence_score:.2f}.")
    elif mean_confidence_score >= 0.7:
        st.info(f"🔎 **Moderate Confidence**: Based on the 10 samples, the model's predictions on random data have an average confidence score of {mean_confidence_score:.2f}.")
    else:
        st.warning(f"⚠️ **Low Confidence**: Based on the 10 samples, the model shows lower confidence in the random data predictions with an average score of {mean_confidence_score:.2f}.")
else:
    st.warning("⚠️ **Confidence Scores Not Available**: Based on the 10 samples, the selected model does not support probability-based confidence scores.")

