import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
file_path = 'Data/Breast_Cancer_Balanced.csv'
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
correlation_matrix = data_encoded.corr().round(2)
plt.figure(figsize=(20, 12))
sns.heatmap(correlation_matrix, cmap='copper_r', annot=True)
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

# Separate features and target
X = df_clean.drop("Status", axis=1)
y = df_clean["Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
# model = LogisticRegression(max_iter=1000)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluate
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoders['Status'].classes_, yticklabels=label_encoders['Status'].classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Overall accuracy
accuracy_percent = accuracy_score(y_test, y_pred) * 100

# Alive and Dead class-specific accuracy
alive_label = label_encoders['Status'].transform(['Alive'])[0]
dead_label = label_encoders['Status'].transform(['Dead'])[0]

alive_indices = (y_test == alive_label)
dead_indices = (y_test == dead_label)

alive_accuracy = accuracy_score(y_test[alive_indices], y_pred[alive_indices]) * 100
dead_accuracy = accuracy_score(y_test[dead_indices], y_pred[dead_indices]) * 100

# Print results
print(f"Overall Accuracy: {accuracy_percent:.2f}%")
print(f"Alive Accuracy:   {alive_accuracy:.2f}%")
print(f"Dead Accuracy:    {dead_accuracy:.2f}%")