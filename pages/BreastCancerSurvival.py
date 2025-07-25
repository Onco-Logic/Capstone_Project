import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import streamlit as st
import seaborn as sns
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE, KMeansSMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Breast Cancer Prognosis",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items=None,
)

st.title("Breast Cancer Prognosis")
st.markdown("---")

# Load the dataset
file_path = 'Data/Breast_Cancer.csv'
data = pd.read_csv(file_path)

############################################# Data Preprocessing #############################################

st.markdown("---")
st.subheader("Data Preprocessing")
st.markdown("---")

# Label encode classification columns
le = LabelEncoder()
pdata = data.copy()
for i in pdata.columns:
    if pdata[i].dtype == 'object':
        pdata[i] = le.fit_transform(pdata[i])

st.subheader("Encoded Dataset")
st.dataframe(pdata.head())

# Copy of encoded dataset to use for survival prediction
pdataS = pdata.copy()

################################################## Survival Model #############################################

st.markdown("---")
st.subheader("Survival Model (Multi-Class Classification)")
st.markdown("---")

# 1. Create the new target column 'Survival Class' with 6 categories
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

pdataS['Survival Class'] = pdataS['Survival Months'].apply(categorize_survival)

# Define features (X1) and target (y1) for the classification task
X1 = pdataS.drop(['Survival Months', 'Survival Class'], axis=1)
y1 = pdataS['Survival Class']

# Get unique class names for display purposes
class_names = ['<1 Year', '1-2 Years', '2-3 Years', '3-4 Years', '4-5 Years', '5-6 Years', '7-8 Years', '8-9 Years', '9+ Years']

st.subheader("Splitting data into X1 and y1 (Survival Class)")
st.dataframe(X1.head())
st.dataframe(y1.head())

##################################################################################

'''RandomOverSampler with modified parameters - This approach shows promise'''
st.write(f"Original dataset shape: {Counter(y1)}")
original_counts = Counter(y1) 
desired_sampling_strategy = {
    0: 600,
    1: 600,
    2: 600,
    3: 600,
    4: 1200,
    5: 1300,
    6: 1300,
    7: 1300,
    8: 1300
}
RandomSample_survival = RandomOverSampler(random_state=42, sampling_strategy=desired_sampling_strategy)
X1_resampled, y1_resampled = RandomSample_survival.fit_resample(X1, y1)
st.write(f"Resampled dataset shape after RandomOverSampler: {Counter(y1_resampled)}")

##################################################################################

# Splitting data into training and testing sets

X1_train, X1_test, y1_train, y1_test = train_test_split(X1_resampled, y1_resampled, test_size=0.2, random_state=42) #resampled

####################################### Train Random Forest #######################################

st.subheader("Random Forest Classifier (Tuned for Survival Class)")

modelRFC_sm = RandomForestClassifier(random_state=42)
modelRFC_sm.fit(X1_train, y1_train)

y1_pred_sm = modelRFC_sm.predict(X1_test)

# Overall metrics
accuracy_rfc = accuracy_score(y1_test, y1_pred_sm)
balanced_accuracy_rfc = balanced_accuracy_score(y1_test, y1_pred_sm)
precision_rfc = precision_score(y1_test, y1_pred_sm, average='weighted', zero_division=0)
recall_rfc = recall_score(y1_test, y1_pred_sm, average='weighted', zero_division=0)
f1_rfc = f1_score(y1_test, y1_pred_sm, average='weighted', zero_division=0)

st.write(f"Accuracy: {accuracy_rfc:.3f}")
st.write(f"Balanced Accuracy: {balanced_accuracy_rfc:.3f}")
st.write(f"Precision (Weighted): {precision_rfc:.3f}")
st.write(f"Recall (Weighted): {recall_rfc:.3f}")
st.write(f"F1 Score (Weighted): {f1_rfc:.3f}")

# Confusion matrix
conf_mat_sm = confusion_matrix(y1_test, y1_pred_sm)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pd.DataFrame(conf_mat_sm,
    columns=[f"Pred {name}" for name in class_names],
    index=[f"Actual {name}" for name in class_names]),
    annot=True, cmap="Blues", fmt="d", ax=ax
)
ax.set_title("Random Forest Survival Class Confusion Matrix")
ax.set_xlabel("Predicted labels")
ax.set_ylabel("True labels")
st.pyplot(fig)

# Classification report
st.write("Classification Report: Random Forest (Survival Class): ")
report_rfc_dict = classification_report(y1_test, y1_pred_sm, target_names=class_names, output_dict=True, zero_division=0)
report_rfc_df = pd.DataFrame(report_rfc_dict).transpose().round(2)
st.table(report_rfc_df)

####################################### Train XGBoost #######################################

st.subheader("XGBoost Classifier (Survival Class)")

modelXGB_sm = XGBClassifier(random_state=42, eval_metric='logloss')
modelXGB_sm.fit(X1_train, y1_train)

y1_pred_xgb = modelXGB_sm.predict(X1_test)

# Overall metrics
accuracy_xgb = accuracy_score(y1_test, y1_pred_xgb)
balanced_accuracy_xgb = balanced_accuracy_score(y1_test, y1_pred_xgb)
precision_xgb = precision_score(y1_test, y1_pred_xgb, average='weighted', zero_division=0)
recall_xgb = recall_score(y1_test, y1_pred_xgb, average='weighted', zero_division=0)
f1_xgb = f1_score(y1_test, y1_pred_xgb, average='weighted', zero_division=0)

st.write(f"Accuracy: {accuracy_xgb:.3f}")
st.write(f"Balanced Accuracy: {balanced_accuracy_xgb:.3f}")
st.write(f"Precision (Weighted): {precision_xgb:.3f}")
st.write(f"Recall (Weighted): {recall_xgb:.3f}")
st.write(f"F1 Score (Weighted): {f1_xgb:.3f}")

# Confusion matrix
conf_mat_xgb = confusion_matrix(y1_test, y1_pred_xgb)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pd.DataFrame(conf_mat_xgb,
    columns=[f"Pred {name}" for name in class_names],
    index=[f"Actual {name}" for name in class_names]),
    annot=True, cmap="Blues", fmt="d", ax=ax
)
ax.set_title("XGBoost Survival Class Confusion Matrix")
ax.set_xlabel("Predicted labels")
ax.set_ylabel("True labels")
st.pyplot(fig)

# Classification report
st.write("Classification Report: XGBoost (Survival Class): ")
report_xgb_dict = classification_report(y1_test, y1_pred_xgb, target_names=class_names, output_dict=True, zero_division=0)
report_xgb_df = pd.DataFrame(report_xgb_dict).transpose().round(2)
st.table(report_xgb_df)

##################### Clean noisy points with ENN) ############################

# Clean noisy points with ENN (KNN‐based)

# Use ENN to clean noisy points from resampled data
enn = EditedNearestNeighbours(
    sampling_strategy='all',
    n_neighbors=2,
    kind_sel='all'
)

X1_cleaned, y1_cleaned = enn.fit_resample(X1_resampled, y1_resampled)

# Print shape of cleaned data
st.write(f"Shape AFTER ENN cleaning: {Counter(y1_cleaned)}")

# Split cleaned data into training and testing sets
X1_train_cln, X1_test_cln, y1_train_cln, y1_test_cln = train_test_split(
    X1_cleaned, 
    y1_cleaned, 
    test_size=0.2, 
    stratify=y1_cleaned, 
    random_state=42
)

# Train a Random Forest model on cleaned data
modelRFC_cln = RandomForestClassifier(random_state=42)
modelRFC_cln.fit(X1_train_cln, y1_train_cln)
y1_pred_cln = modelRFC_cln.predict(X1_test_cln)

# Print metrics for cleaned data
acc_cln = accuracy_score(y1_test_cln, y1_pred_cln)
bal_acc_cln = balanced_accuracy_score(y1_test_cln, y1_pred_cln)
prec_cln = precision_score(y1_test_cln, y1_pred_cln, average='weighted', zero_division=0)
recall_cln = recall_score(y1_test_cln, y1_pred_cln, average='weighted', zero_division=0)
f1_cln = f1_score(y1_test_cln, y1_pred_cln, average='weighted', zero_division=0)

st.write("### After ENN Cleaning → Random Forest Metrics")
st.write(f"Accuracy: {acc_cln:.3f}")
st.write(f"Balanced Accuracy: {bal_acc_cln:.3f}")
st.write(f"Precision (Weighted): {prec_cln:.3f}")
st.write(f"Recall (Weighted): {recall_cln:.3f}")
st.write(f"F1 Score (Weighted): {f1_cln:.3f}")

# Plot confusion matrix for cleaned data
conf_mat_cln = confusion_matrix(y1_test_cln, y1_pred_cln)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    pd.DataFrame(
        conf_mat_cln,
        columns=[f"Pred {name}" for name in class_names],
        index=[f"Actual {name}" for name in class_names]
    ),
    annot=True,
    cmap="Blues",
    fmt="d",
    ax=ax
)
ax.set_title("After ENN Cleaning: RF Survival Class Confusion Matrix")
ax.set_xlabel("Predicted labels")
ax.set_ylabel("True labels")
st.pyplot(fig)

# ------------------- XGBoost on ENN-cleaned -------------------
modelXGB_cln = XGBClassifier(random_state=42, eval_metric='logloss')
modelXGB_cln.fit(X1_train_cln, y1_train_cln)
y1_pred_xgb_cln = modelXGB_cln.predict(X1_test_cln)

acc_xgb_cln = accuracy_score(y1_test_cln, y1_pred_xgb_cln)
bal_acc_xgb_cln = balanced_accuracy_score(y1_test_cln, y1_pred_xgb_cln)
prec_xgb_cln = precision_score(y1_test_cln, y1_pred_xgb_cln, average='weighted', zero_division=0)
recall_xgb_cln = recall_score(y1_test_cln, y1_pred_xgb_cln, average='weighted', zero_division=0)
f1_xgb_cln = f1_score(y1_test_cln, y1_pred_xgb_cln, average='weighted', zero_division=0)

st.write("### After ENN Cleaning → XGBoost Metrics")
st.write(f"Accuracy: {acc_xgb_cln:.3f}")
st.write(f"Balanced Accuracy: {bal_acc_xgb_cln:.3f}")
st.write(f"Precision (Weighted): {prec_xgb_cln:.3f}")
st.write(f"Recall (Weighted): {recall_xgb_cln:.3f}")
st.write(f"F1 Score (Weighted): {f1_xgb_cln:.3f}")

# Confusion matrix XGBoost
conf_mat_xgb_cln = confusion_matrix(y1_test_cln, y1_pred_xgb_cln)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    pd.DataFrame(conf_mat_xgb_cln,
                 columns=[f"Pred {name}" for name in class_names],
                 index=[f"Actual {name}" for name in class_names]),
    annot=True, cmap="Blues", fmt="d", ax=ax)
ax.set_title("After ENN Cleaning: XGBoost Survival Class Confusion Matrix")
ax.set_xlabel("Predicted labels")
ax.set_ylabel("True labels")
st.pyplot(fig)

# Optional: Full classification report
st.write("Classification Report: XGBoost after ENN Cleaning")
report_xgb_cln = classification_report(
    y1_test_cln, y1_pred_xgb_cln, target_names=class_names, output_dict=True, zero_division=0
)
st.table(pd.DataFrame(report_xgb_cln).transpose().round(2))