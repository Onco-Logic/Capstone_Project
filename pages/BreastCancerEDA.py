import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

st.set_page_config(
    page_title="Breast Cancer Prognosis",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("Breast Cancer Prognosis")
st.markdown("---")

# Load the dataset
file_path = 'Data/Breast_Cancer.csv'
data = pd.read_csv(file_path)

############################################# Data Analysis #############################################

# Display the first few rows of the dataset
st.subheader("Preview of Dataset")
st.dataframe(data.head())
st.markdown("---")

# Display the shape of the dataset
st.subheader("Shape of Dataset")
st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
st.markdown("---")

# Display information about the dataset
summary_df = pd.DataFrame({
    "Null Count": data.isna().sum(),
    "Unique": data.nunique(),
    "Dtype": data.dtypes.astype(str)
})

st.subheader("Summary of Dataset")
st.dataframe(summary_df, use_container_width=True)
st.markdown("---")

# Display dataframe stats
st.subheader("Dataset Statistical Information")
st.dataframe(data.describe(), use_container_width=True)
st.markdown("---")

# Plotting the distribution each column
st.subheader("Data Distribution by Column")
def plot_distribution(data, column):
    fig, ax = plt.subplots()
    sns.histplot(data=data, x=column, ax=ax)
    ax.set_title(f"Distribution of {column}")
    st.pyplot(fig)

selected_column = st.selectbox("Select a column to plot", data.columns)
plot_distribution(data, selected_column)
st.markdown("---")

# Plotting the survival months by each column
st.subheader("Survival Months by Column")
def plot_survival_by_category(data, category):
    fig, ax = plt.subplots(figsize=(20, 15))
    sns.countplot(data=data, x=category, hue=data['Survival Months'] // 12, palette=sns.color_palette("muted", 8), ax=ax)
    ax.set_title(f"{category.capitalize()} by Survival Months")
    st.pyplot(fig)

categories = data.columns.drop('Survival Months')
selected_category = st.selectbox("Select a column to plot:", categories)
plot_survival_by_category(data, selected_category)
st.markdown("---")

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
st.markdown("---")

# Plotting the survival months distribution
st.subheader("Survival Months Distribution")
fig, ax = plt.subplots()
sns.histplot(data=data, x='Survival Months', hue=data['Survival Months'] // 12, palette=sns.color_palette("muted", 9), ax=ax)
ax.set_title("Distribution of Survival Months")
ax.set_xticks(range(0, int(data['Survival Months'].max()) + 1, 12))
ax.set_xticklabels([str(i) for i in range(0, int(data['Survival Months'].max()) + 1, 12)])
st.pyplot(fig)
st.markdown("---")

# Plotting Status distribution
st.subheader("Status Distribution")
fig, ax = plt.subplots()
sns.countplot(data=data, x='Status', palette=sns.color_palette("muted", 2), ax=ax)
ax.set_title("Distribution of Status")
st.pyplot(fig)
st.markdown("---")

############################################# Data Preprocessing #############################################

st.title("Data Preprocessing")
st.markdown("---")

# Label encode classification columns
le = LabelEncoder()
pdata = data.copy()
for i in pdata.columns:
    if pdata[i].dtype == 'object':
        pdata[i] = le.fit_transform(pdata[i])

st.subheader("Encoded Dataset")
st.dataframe(pdata.head())
st.markdown("---")

# Copy of encoded dataset to use for survival prediction
pdataS = pdata.copy()

# Display the shape of the dataset
st.subheader("Shape of Dataset")
st.write(f"Rows: {pdata.shape[0]}, Columns: {pdata.shape[1]}")
st.markdown("---")

# Display information about the dataset
summary_df = pd.DataFrame({
    "Null Count": pdata.isna().sum(),
    "Unique": pdata.nunique(),
    "Dtype": pdata.dtypes.astype(str)
})

st.subheader("Summary of Dataset")
st.dataframe(summary_df, use_container_width=True)
st.markdown("---")

# Correlation Heatmap on Encoded Data
correlation_matrix = pdata.corr()
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, ax=ax)
ax.set_title('Correlation Heatmap')
st.pyplot(fig)
st.markdown("---")

################################### PCA #######################################

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
st.markdown("---")

###################################################################################

# Splitting data into X and Y target status
X = pdata.drop("Status", axis=1)
Y = pdata["Status"]

# Splitting data into X and Y
st.subheader("Splitting data into X")
X
st.markdown("---")
st.subheader("Splitting data into Y")
Y
st.markdown("---")

################################################## Model Building Status #############################################

st.title("Status Model")
st.markdown("---")
# Splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

# Initialize the Random Forest Classifier
st.subheader("Random Forest Classifier")
modelRFC = RandomForestClassifier(random_state=42)

modelRFC.fit(X_train, Y_train)

st.subheader("Model Evaluation")
st.write(f"Model: {modelRFC}")

Y_pred = modelRFC.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
st.write(f"Model Accuracy: {accuracy:.3f}")

# Initialize the XGBoost Classifier
st.subheader("XGBoost Classifier")

modelXGB = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Train the XGBoost model on the training data
modelXGB.fit(X_train, Y_train)

st.write(f"Trained Model: {modelXGB}")

# Make predictions on the test data
Y_pred_xgb = modelXGB.predict(X_test)

# Calculate the accuracy
accuracy_xgb = accuracy_score(Y_test, Y_pred_xgb)
st.write(f"XGBoost Accuracy on Test Set: {accuracy_xgb:.3f}")
st.markdown("---")

################################################## Model Building Survival #############################################

st.title("Survival Model")
st.markdown("---")

# Create the new target column on pdataS
pdataS['Survival Years'] = pdataS['Survival Months'] // 12

# Define features (X1) and target (Y1) for pdataS
# Drop both the original months and the years
X1 = pdataS.drop(['Survival Months', 'Survival Years'], axis=1)
Y1 = pdataS['Survival Years']

# Splitting data into training and testing sets
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.1, random_state=42)

# Initialize the Random Forest Classifier
st.subheader("Random Forest Classifier")
modelRFC = RandomForestClassifier(random_state=42)

modelRFC.fit(X1_train, Y1_train)

st.subheader("Model Evaluation")
st.write(f"Model: {modelRFC}")

Y1_pred = modelRFC.predict(X1_test)
accuracy = accuracy_score(Y1_test, Y1_pred)
st.write(f"Model Accuracy: {accuracy:.3f}")

# Initialize the XGBoost Classifier
st.subheader("XGBoost Classifier")

modelXGB = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Train the XGBoost model on the training data
modelXGB.fit(X1_train, Y1_train)

st.write(f"Trained Model: {modelXGB}")

# Make predictions on the test data
Y1_pred_xgb = modelXGB.predict(X1_test)

# Calculate the accuracy
accuracy_xgb = accuracy_score(Y_test, Y1_pred_xgb)
st.write(f"XGBoost Accuracy on Test Set: {accuracy_xgb:.3f}")
st.markdown("---")