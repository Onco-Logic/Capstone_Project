
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def categorize_survival(months):
    if months < 12:
        return 0
    elif 12 <= months < 24:
        return 1
    elif 24 <= months < 36:
        return 2
    elif 36 <= months < 48:
        return 3
    elif 48 <= months < 60:
        return 4
    elif 60 <= months < 72:
        return 5
    elif 72 <= months < 84:
        return 6
    elif 84 <= months < 96:
        return 7
    else:
        return 8

# ENN (parameter = 2) is used to remove noise from the dataset
# Use random over-sampling to balance the dataset
# Use Random Forest Survival Prediction Model


# Load the raw and balanced dataset
filepathRaw = 'Data/Breast_Cancer.csv'
filepathBalanced = 'Data/Breast_Cancer_Balanced.csv'
dfRaw = pd.read_csv(filepathRaw)
dfBalanced = pd.read_csv(filepathBalanced)

# Encode categorical variables
dfEncoded = dfBalanced.copy()
labelEnc = LabelEncoder()
for col in dfEncoded.select_dtypes(include='object').columns:
    dfEncoded[col] = labelEnc.fit_transform(dfEncoded[col])

# Define features and new categorized target
dfEncoded['survival_category'] = dfEncoded['Survival Months'].apply(categorize_survival)
X = dfEncoded.drop(['Status', 'Survival Months', 'survival_category'], axis=1)
y = dfEncoded['survival_category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
rf_preds = rf.predict(X_test_scaled)

# Model 2: XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb.fit(X_train_scaled, y_train)
xgb_preds = xgb.predict(X_test_scaled)

# Model 3: Support Vector Machine
svm = SVC(probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
svm_preds = svm.predict(X_test_scaled)

# Evaluation
models = {'Random Forest': rf_preds, 'XGBoost': xgb_preds, 'SVM': svm_preds}
for name, preds in models.items():
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))
