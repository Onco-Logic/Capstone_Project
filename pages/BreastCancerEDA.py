import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Breast Cancer Prognosis",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("Breast Cancer Prognosis")

# Load the dataset
file_path = 'Data/Breast_Cancer.csv'
data = pd.read_csv(file_path)

############################################# Data Analysis #############################################

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
    sns.countplot(data=data, x=category, hue=data['Survival Months'] // 12, palette=sns.color_palette("muted", 8), ax=ax)
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
sns.countplot(data=data, x='Status', palette=sns.color_palette("muted", 2), ax=ax)
ax.set_title("Distribution of Status")
st.pyplot(fig)

############################################# Data Preprocessing #############################################

st.title("Data Preprocessing")
# Label encode classification columns
le = LabelEncoder()
pdata = data.copy()
for i in pdata.columns:
    if pdata[i].dtype == 'object':
        pdata[i] = le.fit_transform(pdata[i])

st.subheader("Encoded Dataset")
st.dataframe(pdata.head())

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

# Splitting data into X and Y target status
X = pdata.drop("Status", axis=1)
Y = pdata["Status"]

# Splitting data into X and Y
st.subheader("Splitting data into X")
X
st.subheader("Splitting data into Y")
Y

# Convert 'Survival Months' to 'Year'
#pdataSM['Year'] = pdata['Survival Months'] // 12










'''
# Displaying the descriptive statistics for the 'Survival Months' column
print(data['Survival Months'].describe())

# Filtering the data to include only rows where 'Survival Months' is less than or equal to 100
data = data[data['Survival Months'] <= 100]

# Displaying the descriptive statistics for the filtered 'Survival Months' column to verify the changes
print(data['Survival Months'].describe())

# Data Preparation
# Removing the target columns as instructed
# data.drop(columns=['Status', 'Survival Months'], inplace=True)

# Data Cleaning
# Checking for missing values
missing_values = data.isnull().sum()

# Categorical Data Analysis
# Identifying categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# One-Hot Encoding
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
print(data_encoded)

# Step 4: Numerical Data Analysis
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns

# Distribution analysis
for column in numerical_columns:
    plt.figure()
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

# Step 5: Correlation Analysis
correlation_matrix = data_encoded.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap')
plt.show()

# Step 6: PCA Preparation
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_encoded)

# Applying PCA
pca = PCA()
principal_components = pca.fit_transform(data_scaled)
explained_variance = pca.explained_variance_ratio_

# Visualizing PCA Explained Variance
plt.figure()
plt.plot(np.cumsum(explained_variance))
plt.title('Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# Displaying Results
print("\nMissing Values:")
print(missing_values)
print("\nPCA Explained Variance:")
print(explained_variance)
'''