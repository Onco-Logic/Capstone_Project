import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report

# Ensure 'models/' directory exists (Fixes nested models/models issue)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
# BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# MODELS_DIR = os.path.join(BASE_DIR, "models/models")
# os.makedirs(MODELS_DIR, exist_ok=True)

# Load dataset safely
df1 = pd.read_csv("Data/Blood_samples_dataset_balanced_2.csv")
df2 = pd.read_csv("Data/blood_samples_dataset_test.csv")

df = pd.concat([df1, df2], ignore_index=True)
df = df.replace('Thalasse', 'Thalassemia').replace('Heart Di', 'Heart Disease')
# Prepare data
X = df.drop(columns=["Disease"])
y = df["Disease"]

# Normalize features (Apply same scaling for all splits)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified train-validation-test split (Ensures balance)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Define Models with Hyperparameter Tuning (Random Forest & Logistic Regression)
models = {
    "Random Forest": GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid={"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]},
        cv=3, scoring="accuracy", n_jobs=-1
    ),
    "Naive Bayes": GaussianNB(),  # No hyperparameters needed
    "Logistic Regression": GridSearchCV(
        LogisticRegression(max_iter=2000, solver="liblinear"),
        param_grid={"C": [0.1, 1, 10]},
        cv=3, scoring="accuracy", n_jobs=-1
    ),
    "Decision Tree": GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid={
            "max_depth": [None, 5, 10, 20], 
            "min_samples_split": [2, 5, 10], 
            "criterion": ["gini"] 
        },
        cv=3, scoring="accuracy", n_jobs=-1
    ),
}

trained_models = {}

# Train & Save Models
for name, model in models.items():
    print(f"Training {name}...")

    # Apply GridSearchCV only when needed
    if isinstance(model, GridSearchCV):
        model.fit(X_train, y_train)
        best_model = model.best_estimator_
    else:
        best_model = model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    # Save trained models (Ensuring correct paths)
    model_filename = f"{name.lower().replace(' ', '_')}_model.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(best_model, model_path)
    trained_models[name] = best_model

# Save the scaler (Ensuring future predictions are normalized the same way)
scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)

print("✅ All models trained and saved successfully!")

# Feature Importance (Random Forest & Logistic Regression)
plt.figure(figsize=(10, 5))
if "Random Forest" in trained_models:
    feature_importance = trained_models["Random Forest"].feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    sorted_features = X.columns[sorted_idx]
    sorted_importances = feature_importance[sorted_idx]

    sns.barplot(x=sorted_importances, y=sorted_features, label="Random Forest")

if "Logistic Regression" in trained_models:
    importance = np.abs(trained_models["Logistic Regression"].coef_).mean(axis=0)
    sorted_idx = np.argsort(importance)[::-1]
    sorted_features = X.columns[sorted_idx]
    sorted_importances = importance[sorted_idx]

    sns.barplot(x=sorted_importances, y=sorted_features, label="Logistic Regression", color="orange")

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance")
plt.legend()
plt.show()

# ROC Curve Comparison (Skipping Naive Bayes since it lacks `predict_proba`)
plt.figure(figsize=(12, 5))

for name, model in trained_models.items():
    if hasattr(model, "predict_proba"):  # Ensure probability prediction exists
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=y_test.unique()[1])
        auc_score = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Disease Prediction Models")
plt.legend()
plt.show()
