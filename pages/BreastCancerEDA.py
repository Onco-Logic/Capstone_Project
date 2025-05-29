import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
file_path = 'Data/Breast_Cancer.csv'
df = pd.read_csv(file_path)

# Displaying the descriptive statistics for the 'Survival Months' column to understand its distribution
print(df['Survival Months'].describe())

# Encode categorical variables
df_clean = df.copy()
label_encoders = {}
for column in df_clean.columns:
    if df_clean[column].dtype == object:
        le = LabelEncoder()
        df_clean[column] = le.fit_transform(df_clean[column])
        label_encoders[column] = le

# Standardize the features
features = df_clean.columns.drop(['Status'])
x = df_clean[features].values
x = StandardScaler().fit_transform(x)

# PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(x)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
pca_df['Status'] = df_clean['Status']

# Explained variance plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, 4), pca.explained_variance_ratio_, marker='o', linestyle='--')
plt.title('Explained Variance by PCA Components')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.tight_layout()
plt.show()

# 2D PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Status', alpha=0.7)
plt.title('2D PCA of Breast Cancer Dataset')
plt.tight_layout()
plt.show()

# 3D PCA
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
statuses = pca_df['Status'].unique()

for status in statuses:
    idx = pca_df['Status'] == status
    ax.scatter(pca_df.loc[idx, 'PC1'],
               pca_df.loc[idx, 'PC2'],
               pca_df.loc[idx, 'PC3'],
               label=status, alpha=0.6)

ax.set_title('3D PCA of Breast Cancer Dataset')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.legend()
plt.tight_layout()
plt.show()


# Filtering the data to include only rows where 'Survival Months' is greater than or equal to 54
df = df[df['Survival Months'] >= 54]

# Filtering the data to include only rows where 'Survival Months' is less than or equal to 100
df = df[df['Survival Months'] <= 100]

# Displaying the descriptive statistics for the filtered 'Survival Months' column to verify the changes
print(df['Survival Months'].describe())

# Data Preparation
# Removing the target columns as instructed
df.drop(columns=['Status', 'Survival Months'], inplace=True)

# Data Cleaning
# Checking for missing values
missing_values = df.isnull().sum()

# Categorical Data Analysis
# Identifying categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns

# One-Hot Encoding
data_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
print(data_encoded)

# Step 4: Numerical Data Analysis
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Distribution analysis
for column in numerical_columns:
    plt.figure()
    sns.histplot(df[column], kde=True)
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
