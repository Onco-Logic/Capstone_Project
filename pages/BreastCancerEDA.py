import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

# Load the dataset
file_path = '../Data/Breast_Cancer.csv'
data = pd.read_csv(file_path)

# Displaying the descriptive statistics for the 'Survival Months' column to understand its distribution
print(data['Survival Months'].describe())

# Filtering the data to include only rows where 'Survival Months' is greater than or equal to 54
# data = data[data['Survival Months'] >= 54]

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
