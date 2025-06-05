import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

# ─── IMBALANCE‐HANDLING IMPORTS ───────────────────────────────────────────────
from imblearn.over_sampling import RandomOverSampler, SMOTE, KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter

# ─── SKLEARN / XGBOOST IMPORTS ────────────────────────────────────────────────
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
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

# Load the raw and balanced dataset
filepathRaw = 'Data/Breast_Cancer.csv'
dfRaw = pd.read_csv(filepathRaw)
filepathBalanced = 'Data/Breast_Cancer_Balanced.csv'
dfBalanced = pd.read_csv(filepathBalanced)

#############################################
# 1) ENCODE CATEGORICAL VARIABLES
#############################################

# Encode categorical variables
le = LabelEncoder()
dfEncoded = dfBalanced.copy()
for i in dfEncoded.columns:
    if dfEncoded[i].dtype == 'object':
        dfEncoded[i] = le.fit_transform(dfEncoded[i])

st.subheader("Encoded Dataset")
st.dataframe(dfEncoded.head())

#######################################################
# 2) CREATE SIX‐BIN “Survival Class” TARGET VARIABLE
#######################################################

# Function to categorize survival months into bins
def categorize_survival(months):
    if months < 12:
        return 0
    elif 12 <= months < 24:
        return 1
    elif 24 <= months < 48:
        return 2
    elif 48 <= months < 72:
        return 3
    elif 72 <= months < 96:
        return 4
    else:
        return 5

# Define features and new categorized target
dfEncoded['Survival Class'] = dfEncoded['Survival Months'].apply(categorize_survival)

# Drop original “Status” and “Survival Months” from features
X = dfEncoded.drop(['Status', 'Survival Months', 'Survival Class'], axis=1)
y = dfEncoded['Survival Class']

# Get unique class names for display purposes
class_names = [
    '<1 Year',
    '1-2 Years',
    '2-4 Years',
    '4-6 Years',
    '6-8 Years',
    '9+ Years'
    ]

st.subheader("Splitting data into X and y (Survival Class)")
st.dataframe(X.head())
st.dataframe(y.head())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#############################################
# 3) BALANCE THE TRAINING SET WITH SMOTE
#############################################

# Check original training‐set class distribution
original_counts = Counter(y_train)
st.write("Original training‐set counts:", dict(original_counts))

# (a) Use SMOTE + RandomOverSampler to upsample rare bins
smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_train_scaled, y_train)

st.write("After SMOTE training‐set counts:", dict(Counter(y_sm)))


#############################################
# 4) RANDOM FOREST WITH GRIDSEARCH TUNING
#############################################

st.subheader("4. Random Forest Hyperparameter Tuning")

# Use class_weight='balanced' so RF penalizes majority classes less
rfc = RandomForestClassifier(random_state=42, class_weight='balanced')

# Choose a small hyperparameter grid to keep runtime reasonable
# ▶️ CHANGE: Slightly expanded RF grid (added max_features, bootstrap)
param_grid_rf = {
    "n_estimators": [100, 200],       # try 100 and 200 trees
    "max_depth": [None, 10, 20],      # no max, or depth 10/20
    "min_samples_split": [2, 5],      # minimum split threshold
    "min_samples_leaf": [1, 2]        # leaf node size
}

grid_rf = GridSearchCV(
    estimator=rfc,
    param_grid=param_grid_rf,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit on the SMOTE→ENN→PCA training set
grid_rf.fit(X_sm, y_sm)
best_rfc = grid_rf.best_estimator_
st.write("Best RF params:", grid_rf.best_params_)

# Predict on the original (un‐oversampled) test set
rf_preds = best_rfc.predict(X_test_scaled)

# Overall metrics
accuracy_rfc = accuracy_score(y_test, rf_preds)
balanced_accuracy_rfc = balanced_accuracy_score(y_test, rf_preds)
precision_rfc = precision_score(y_test, rf_preds, average='weighted', zero_division=0)
recall_rfc = recall_score(y_test, rf_preds, average='weighted', zero_division=0)
f1_rfc = f1_score(y_test, rf_preds, average='weighted', zero_division=0)

st.write(f"Accuracy: {accuracy_rfc:.3f}")
st.write(f"Balanced Accuracy: {balanced_accuracy_rfc:.3f}")
st.write(f"Precision (Weighted): {precision_rfc:.3f}")
st.write(f"Recall (Weighted): {recall_rfc:.3f}")
st.write(f"F1 Score (Weighted): {f1_rfc:.3f}")

# Confusion matrix
conf_mat_rf = confusion_matrix(y_test, rf_preds)
fig_rf, ax_rf = plt.subplots(figsize=(8, 6))
sns.heatmap(pd.DataFrame(conf_mat_rf,
    columns=[f"Pred {name}" for name in class_names],
    index=[f"Actual {name}" for name in class_names]),
    annot=True, cmap="Blues", fmt="d", ax=ax_rf
)
ax_rf.set_title("Random Forest Survival Class Confusion Matrix")
ax_rf.set_xlabel("Predicted labels")
ax_rf.set_ylabel("True labels")
st.pyplot(fig_rf)

# Classification report
st.write("Classification Report: Random Forest (Survival Class): ")
report_rfc_dict = classification_report(y_test, rf_preds, target_names=class_names, output_dict=True, zero_division=0)
report_rfc_df = pd.DataFrame(report_rfc_dict).transpose().round(2)
st.table(report_rfc_df)

#-------------ENN Results------------------------------#

# ▶️ CHANGE: Apply EditedNearestNeighbours to remove noisy neighbors that SMOTE may have introduced
enn = EditedNearestNeighbours(sampling_strategy='all', n_neighbors=2, kind_sel='all')
X_res, y_res = enn.fit_resample(X_sm, y_sm)

st.write("Counts after SMOTE → ENN:", dict(Counter(y_res)))
X_enn_train, X_enn_test, y_enn_train, y_enn_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

grid_rf.fit(X_enn_train, y_enn_train)
y_enn_pred = grid_rf.predict(X_enn_test)
st.write("Best RF params:", grid_rf.best_params_)

# Overall metrics
accuracy_rfc = accuracy_score(y_enn_test, y_enn_pred)
balanced_accuracy_rfc = balanced_accuracy_score(y_enn_test, y_enn_pred)
precision_rfc = precision_score(y_enn_test, y_enn_pred, average='weighted', zero_division=0)
recall_rfc = recall_score(y_enn_test, y_enn_pred, average='weighted', zero_division=0)
f1_rfc = f1_score(y_enn_test, y_enn_pred, average='weighted', zero_division=0)

st.write(f"Accuracy: {accuracy_rfc:.3f}")
st.write(f"Balanced Accuracy: {balanced_accuracy_rfc:.3f}")
st.write(f"Precision (Weighted): {precision_rfc:.3f}")
st.write(f"Recall (Weighted): {recall_rfc:.3f}")
st.write(f"F1 Score (Weighted): {f1_rfc:.3f}")

# Confusion matrix
conf_mat_rf = confusion_matrix(y_enn_test, y_enn_pred)
fig_rf, ax_rf = plt.subplots(figsize=(8, 6))
sns.heatmap(pd.DataFrame(conf_mat_rf,
    columns=[f"Pred {name}" for name in class_names],
    index=[f"Actual {name}" for name in class_names]),
    annot=True, cmap="Blues", fmt="d", ax=ax_rf
)
ax_rf.set_title("Random Forest Survival Class Confusion Matrix")
ax_rf.set_xlabel("Predicted labels")
ax_rf.set_ylabel("True labels")
st.pyplot(fig_rf)

# Classification report
st.write("Classification Report: Random Forest (Survival Class): ")
report_rfc_dict = classification_report(y_enn_test, y_enn_pred, target_names=class_names, output_dict=True, zero_division=0)
report_rfc_df = pd.DataFrame(report_rfc_dict).transpose().round(2)
st.table(report_rfc_df)

#############################################
# 5) XGBOOST WITH GRIDSEARCH TUNING
#############################################

st.subheader("5. XGBoost Hyperparameter Tuning")

# When using XGBoost, we pass scale_pos_weight, but since it's multiclass, we'll rely on SMOTE balancing + objective='multi:softmax'
xgb = XGBClassifier(
    objective='multi:softmax',
    num_class=6,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

param_grid_xgb = {
    "n_estimators": [100, 200],
    "max_depth": [3, 6],
    "learning_rate": [0.1, 0.01],
    "subsample": [0.8, 1.0]
}

grid_xgb = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid_xgb,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit on SMOTE‐balanced training data
grid_xgb.fit(X_sm, y_sm)
best_xgb = grid_xgb.best_estimator_

st.write("Best XGB params:", grid_xgb.best_params_)

# Predict on the original test set
xgb_preds = best_xgb.predict(X_test_scaled)

accuracy_xgb = accuracy_score(y_test, xgb_preds)
balanced_accuracy_xgb = balanced_accuracy_score(y_test, xgb_preds)
report_xgb = classification_report(
    y_test, xgb_preds, target_names=class_names, zero_division=0
)

st.write(f"XGB Overall Accuracy: {accuracy_xgb:.3f}")
st.write(f"XGB Balanced Accuracy: {balanced_accuracy_xgb:.3f}")
st.write("XGBoost Classification Report:")
st.text(report_xgb)

# ─── CONFUSION MATRIX VISUALIZATION ────────────────────────────────────────
conf_mat_xgb = confusion_matrix(y_test, xgb_preds)
fig_xgb, ax_xgb = plt.subplots(figsize=(8, 6))
sns.heatmap(
    pd.DataFrame(
        conf_mat_xgb,
        index=[f"Actual {cn}" for cn in class_names],
        columns=[f"Pred {cn}" for cn in class_names]
    ),
    annot=True, fmt="d", cmap="Greens", ax=ax_xgb
)
ax_xgb.set_title("Tuned XGB: Survival Class Confusion Matrix")
ax_xgb.set_xlabel("Predicted Labels")
ax_xgb.set_ylabel("True Labels")
st.pyplot(fig_xgb)

####################################### Train XGBoost #######################################

# # Model 2: XGBoost
# xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
# xgb.fit(X_train_scaled, y_train)
# xgb_preds = xgb.predict(X_test_scaled)

# # Model 3: Support Vector Machine
# svm = SVC(probability=True, random_state=42)
# svm.fit(X_train_scaled, y_train)
# svm_preds = svm.predict(X_test_scaled)

# # Evaluation
# models = {'Random Forest': rf_preds, 'XGBoost': xgb_preds, 'SVM': svm_preds}
# for name, preds in models.items():
#     print(f"\n{name} Results:")
#     print("Accuracy:", accuracy_score(y_test, preds))
#     print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
#     print("Classification Report:\n", classification_report(y_test, preds))