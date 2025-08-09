import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import streamlit as st
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
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

############################################# Data Exploration #############################################

st.subheader("Data Exploration")
st.markdown("---")

# Display the first few rows of the dataset
st.subheader("Preview of Dataset")
st.dataframe(data.head())

# Display the shape of the dataset
st.subheader("Shape of Dataset")
st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

# Display information about the dataset
summary_df = pd.DataFrame({
    "Null Count": data.isna().sum(),
    "Unique": data.nunique(),
    "Dtype": data.dtypes.astype(str)
})
st.subheader("Summary of Dataset")
st.dataframe(summary_df, use_container_width=True)

# Display dataframe stats
st.subheader("Dataset Statistical Information")
st.dataframe(data.describe(), use_container_width=True)

# Plotting the distribution each column
st.subheader("Data Distribution by Column")
def plot_distribution(data, column):
    fig, ax = plt.subplots()
    sns.histplot(data=data, x=column, ax=ax)
    ax.set_title(f"Distribution of {column}")
    st.pyplot(fig)
selected_column = st.selectbox("Select a column to plot", data.columns)
plot_distribution(data, selected_column)

# Plotting the survival months by each column
st.subheader("Survival Months by Column")
def plot_survival_by_category(data, category):
    fig, ax = plt.subplots(figsize=(20, 15))
    sns.countplot(data=data, x=category, hue=data['Survival Months'] // 12, palette=sns.color_palette("muted", 9), ax=ax)
    ax.set_title(f"{category.capitalize()} by Survival Months")
    st.pyplot(fig)
categories = data.columns.drop('Survival Months')
selected_category = st.selectbox("Select a column to plot:", categories)
plot_survival_by_category(data, selected_category)

# Plotting Status by each column
st.subheader("Status by Column")
def plot_status_by_category(data, category):
    fig, ax = plt.subplots(figsize=(20, 15))
    sns.countplot(data=data, x=category, hue='Status', palette=sns.color_palette("muted", 2), ax=ax)
    ax.set_title(f"{category.capitalize()} by Status")
    st.pyplot(fig)
categories = data.columns.drop('Status')
selected_category = st.selectbox("Select a column to plot:", categories)
plot_status_by_category(data, selected_category)

# Plotting the survival months distribution
st.subheader("Survival Months Distribution")
fig, ax = plt.subplots()
sns.histplot(data=data, x='Survival Months', hue=data['Survival Months'] // 12, palette=sns.color_palette("muted", 9), ax=ax)
ax.set_title("Distribution of Survival Months")
ax.set_xticks(range(0, int(data['Survival Months'].max()) + 1, 12))
ax.set_xticklabels([str(i) for i in range(0, int(data['Survival Months'].max()) + 1, 12)])
st.pyplot(fig)

# Plotting Status distribution
st.subheader("Status Distribution")
fig, ax = plt.subplots()
sns.countplot(data=data, x='Status', hue='Status', palette=sns.color_palette("muted", 2), ax=ax)
ax.set_title("Distribution of Status")
st.pyplot(fig)

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

# Display the shape of the dataset
st.subheader("Shape of Dataset")
st.write(f"Rows: {pdata.shape[0]}, Columns: {pdata.shape[1]}")

# Display information about the dataset
summary_df = pd.DataFrame({
    "Null Count": pdata.isna().sum(),
    "Unique": pdata.nunique(),
    "Dtype": pdata.dtypes.astype(str)
})

st.subheader("Summary of Dataset")
st.dataframe(summary_df, use_container_width=True)

# Correlation Heatmap on Encoded Data
correlation_matrix = pdata.corr()
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, ax=ax)
ax.set_title('Correlation Heatmap')
st.pyplot(fig)

################################### PCA #######################################

st.markdown("---")
st.subheader("Principal Component Analysis")
st.markdown("---")

# Apply PCA to data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pdata)
# Apply PCA to scaled data
pca_model = PCA()
pca_data = pca_model.fit_transform(scaled_data)

# Plot explained variance
explained_variance = pca_model.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)
fig, ax = plt.subplots()
ax.plot(cumulative_explained_variance)
ax.set_title("Explained Variance by PCA Components")
ax.set_xlabel("Number of Components")
ax.set_ylabel("Cumulative Explained Variance")
st.pyplot(fig)

################################################## Model Building #############################################

st.markdown("---")
st.subheader("Status Models")
st.markdown("---")

# Splitting data into X and Y target status
X = pdata.drop("Status", axis=1)
y = pdata["Status"]

# Splitting data into X and Y
st.subheader("Splitting data into X")
X
st.subheader("Splitting data into Y")
y

RandomSample = RandomOverSampler(random_state=42)
X, y = RandomSample.fit_resample(X,y)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

############################ Random Forest ############################

# Initialize the Random Forest Classifier
st.subheader("Random Forest Classifier")
modelRFC = RandomForestClassifier(random_state=42)
modelRFC.fit(X_train, y_train)

# Make predictions and prediction probabilities
y_pred = modelRFC.predict(X_test)
y_pred_DOA = modelRFC.predict_proba(X_test)[:, 1]

# Overall metrics
accuracy_RFC = accuracy_score(y_test, y_pred)
balanced_accuracy_RFC = balanced_accuracy_score(y_test, y_pred)
precision_RFC = precision_score(y_test, y_pred)
recall_RFC = recall_score(y_test, y_pred)
f1_RFC = f1_score(y_test, y_pred)
roc_auc_RFC = roc_auc_score(y_test, y_pred_DOA)

st.write(f"Accuracy: {accuracy_RFC:.3f}")
st.write(f"Balanced Accuracy: {balanced_accuracy_RFC:.3f}")
st.write(f"Precision: {precision_RFC:.3f}")
st.write(f"Recall: {recall_RFC:.3f}")
st.write(f"F1 Score: {f1_RFC:.3f}")
st.write(f"ROC AUC Score: {roc_auc_RFC:.3f}")

# Calculate and display the confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_mat.ravel()
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(pd.DataFrame(conf_mat, columns=["Predicted Dead", "Predicted Alive"], index=["Actual Dead", "Actual Alive"]), 
    annot=True, cmap="Blues", fmt="d", ax=ax)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted labels")
ax.set_ylabel("True labels")
st.pyplot(fig)

alive_accuracy = tp / (tp + fn)
st.write(f"Alive Accuracy: {alive_accuracy:.3f}")
dead_accuracy = tn / (tn + fp)
st.write(f"Dead Accuracy: {dead_accuracy:.3f}")

# Classification report for random forest
st.write("Classification Report: Random Forest: ")
report_rfc_dict = classification_report(y_test, y_pred, target_names=['Dead', 'Alive'], output_dict=True)
report_rfc_df = pd.DataFrame(report_rfc_dict).transpose().round(2)
st.table(report_rfc_df)

####################### XGBoost Classifier ############################

# Initialize the XGBoost Classifier
st.subheader("XGBoost Classifier")
modelXGB = XGBClassifier(random_state=42, eval_metric='logloss')
modelXGB.fit(X_train, y_train)

# Make predictions and prediction probabilities
y_pred_xgb = modelXGB.predict(X_test)
y_pred_DOA_xgb = modelXGB.predict_proba(X_test)[:, 1]

# Overall metrics
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
balanced_accuracy_xgb = balanced_accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
roc_auc_xgb = roc_auc_score(y_test, y_pred_DOA_xgb)

st.write(f"Accuracy: {accuracy_xgb:.3f}")
st.write(f"Balanced Accuracy: {balanced_accuracy_xgb:.3f}")
st.write(f"Precision: {precision_xgb:.3f}")
st.write(f"Recall: {recall_xgb:.3f}")
st.write(f"F1 Score: {f1_xgb:.3f}")
st.write(f"ROC AUC Score: {roc_auc_xgb:.3f}")

# Confusion matrix
conf_mat_xgb = confusion_matrix(y_test, y_pred_xgb)
tn, fp, fn, tp = conf_mat_xgb.ravel()

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(pd.DataFrame(conf_mat_xgb,
    columns=["Predicted Dead", "Predicted Alive"],
    index=["Actual Dead", "Actual Alive"]),
    annot=True, cmap="Blues", fmt="d", ax=ax
)
ax.set_title("XGBoost Confusion Matrix")
ax.set_xlabel("Predicted labels")
ax.set_ylabel("True labels")
st.pyplot(fig)

# Class accuracy
alive_accuracy_xgb = tp / (tp + fn)
dead_accuracy_xgb  = tn / (tn + fp)
st.write(f"Alive Accuracy: {alive_accuracy_xgb:.3f}")
st.write(f"Dead Accuracy:  {dead_accuracy_xgb:.3f}")

# Classification report for XGBoost
st.write("Classification Report: XGBoost: ")
report_xgb_dict = classification_report(y_test, y_pred_xgb, target_names=['Dead', 'Alive'], output_dict=True)
report_xgb_df = pd.DataFrame(report_xgb_dict).transpose().round(2)
st.table(report_xgb_df)

############################ Support Vector Machine ############################

# Initialize the Support Vector Classifier
st.subheader("Support Vector Machine (SVM) Classifier")
modelSVM = SVC(probability=True, random_state=42)
modelSVM.fit(X_train, y_train)

# Make predictions and prediction probabilities
y_pred_svm = modelSVM.predict(X_test)
y_pred_DOA_svm = modelSVM.predict_proba(X_test)[:, 1]

# Overall metrics
accuracy_svm = accuracy_score(y_test, y_pred_svm)
balanced_accuracy_svm = balanced_accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
roc_auc_svm = roc_auc_score(y_test, y_pred_DOA_svm)

st.write(f"Accuracy: {accuracy_svm:.3f}")
st.write(f"Balanced Accuracy: {balanced_accuracy_svm:.3f}")
st.write(f"Precision: {precision_svm:.3f}")
st.write(f"Recall: {recall_svm:.3f}")
st.write(f"F1 Score: {f1_svm:.3f}")
st.write(f"ROC AUC Score: {roc_auc_svm:.3f}")

# Confusion matrix
conf_mat_svm = confusion_matrix(y_test, y_pred_svm)
tn, fp, fn, tp = conf_mat_svm.ravel()

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(pd.DataFrame(conf_mat_svm,
    columns=["Predicted Dead", "Predicted Alive"],
    index=["Actual Dead", "Actual Alive"]),
    annot=True, cmap="Blues", fmt="d", ax=ax
)
ax.set_title("SVM Confusion Matrix")
ax.set_xlabel("Predicted labels")
ax.set_ylabel("True labels")
st.pyplot(fig)

# Class accuracy
alive_accuracy_svm = tp / (tp + fn)
dead_accuracy_svm  = tn / (tn + fp)
st.write(f"Alive Accuracy: {alive_accuracy_svm:.3f}")
st.write(f"Dead Accuracy:  {dead_accuracy_svm:.3f}")

# Classification report for SVM
st.write("Classification Report: Support Vector Machine (SVM): ")
report_svm_dict = classification_report(y_test, y_pred_svm, target_names=['Dead', 'Alive'], output_dict=True)
report_svm_df = pd.DataFrame(report_svm_dict).transpose().round(2)
st.table(report_svm_df)