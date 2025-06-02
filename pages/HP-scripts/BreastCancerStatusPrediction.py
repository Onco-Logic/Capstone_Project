import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import streamlit as st
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from xgboost import XGBClassifier

# Load the raw and balanced dataset
filepathRaw = 'Data/Breast_Cancer.csv'
filepathBalanced = 'Data/Breast_Cancer_Balanced.csv'
dfRaw = pd.read_csv(filepathRaw)
dfBalanced = pd.read_csv(filepathBalanced)

# Displaying the descriptive statistics for the 'Survival Months' column to understand its distribution
print(dfRaw['Survival Months'].describe())
print(dfBalanced['Survival Months'].describe())

# Encode categorical variables for raw dataset
dfEncoded = dfBalanced.copy()
labelEncoders = {}
for column in dfEncoded.columns:
    if dfEncoded[column].dtype == object:
        le = LabelEncoder()
        dfEncoded[column] = le.fit_transform(dfEncoded[column])
        labelEncoders[column] = le

# Standardize the features
features = dfEncoded.columns.drop(['Status'])
x = dfEncoded[features].values
x = StandardScaler().fit_transform(x)

# Separate features and target
X = dfEncoded.drop("Status", axis=1)
y = dfEncoded["Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model 1: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_preds = rf.predict(X_test_scaled)

# Model 2: XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train_scaled, y_train)
xgb_preds = xgb.predict(X_test_scaled)

# Model 3: Support Vector Machine
svm = SVC(probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
svm_preds = svm.predict(X_test_scaled)

# Evaluation
alive_indices = (y_test == 0)
dead_indices = (y_test == 1)

models = {'Random Forest': rf_preds, 'XGBoost': xgb_preds, 'SVM': svm_preds}
for name, preds in models.items():
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy_score(y_test, preds):.2f}")
    print(f"Alive Accuracy: {accuracy_score(y_test[alive_indices], preds[alive_indices]):.2f}")
    print(f"Dead Accuracy: {accuracy_score(y_test[dead_indices], preds[dead_indices]):.2f}\n")
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))
    

# # Class-specific accuracy
# alive_indices = (y_test == 0)
# dead_indices = (y_test == 1)

# for name, preds in models.items():
#     alive_accuracy = accuracy_score(y_test[alive_indices], preds[alive_indices])
#     dead_accuracy = accuracy_score(y_test[dead_indices], preds[dead_indices])
#     print(f"\n{name} Class-specific Accuracy:")
#     print(f"Alive Accuracy: {alive_accuracy:.2f}")
#     print(f"Dead Accuracy: {dead_accuracy:.2f}")


# # Evaluate
# print("\nAccuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # Evaluation
# models = {'Random Forest': rf_preds, 'XGBoost': xgb_preds, 'SVM': svm_preds}
# for name, preds in models.items():
#     print(f"\n{name} Results:")
#     print("Accuracy:", accuracy_score(y_test, preds))
#     print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
#     print("Classification Report:\n", classification_report(y_test, preds))

# # Confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labelEncoders['Status'].classes_, yticklabels=labelEncoders['Status'].classes_)
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()

# # Overall accuracy
# accuracyPercent = accuracy_score(y_test, y_pred) * 100

# # Alive and Dead class-specific accuracy
# aliveLabel = labelEncoders['Status'].transform(['Alive'])[0]
# deadLabel = labelEncoders['Status'].transform(['Dead'])[0]

# aliveIndices = (y_test == aliveLabel)
# deadIndices = (y_test == deadLabel)

# aliveAccuracy = accuracy_score(y_test[aliveIndices], y_pred[aliveIndices]) * 100
# deadAccuracy = accuracy_score(y_test[deadIndices], y_pred[deadIndices]) * 100

# # Print results
# print(f"Overall Accuracy: {accuracyPercent:.2f}%")
# print(f"Alive Accuracy:   {aliveAccuracy:.2f}%")
# print(f"Dead Accuracy:    {deadAccuracy:.2f}%")