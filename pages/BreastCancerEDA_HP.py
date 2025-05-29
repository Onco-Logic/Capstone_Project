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
file_path = 'Data/Breast_Cancer.csv'
df = pd.read_csv(file_path)

# Basic EDA
print("First 5 rows:")
print(df.head())
print("\nData info:")
print(df.info())
print("\nStatistical summary:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())

# Correlation analysis
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.show()

# PCA
# Prepare data for PCA (drop non-numeric and status)
features = df.columns.drop(['Status'])
X = df[features]
# One-hot encode categorical variables if any
X = pd.get_dummies(X, drop_first=True)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
pc_df = pd.DataFrame(data=principal_components, columns=['PC1','PC2'])
pc_df = pd.concat([pc_df, df['status'].reset_index(drop=True)], axis=1)

plt.figure(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', hue='status', data=pc_df)
plt.title('PCA - First two principal components')
plt.savefig('pca_scatter.png')
plt.show()

# Status prediction
# Balance dataset
df_alive = df[df['status']=='alive']
df_dead = df[df['status']=='dead']
min_len = min(len(df_alive), len(df_dead))
df_balanced = pd.concat([
    df_alive.sample(min_len, random_state=42),
    df_dead.sample(min_len, random_state=42)
]).reset_index(drop=True)

# Features and target
X_bal = df_balanced.drop('status', axis=1)
X_bal = pd.get_dummies(X_bal, drop_first=True)
y_bal = df_balanced['status']
le = LabelEncoder()
y_bal_enc = le.fit_transform(y_bal)  # alive=0/dead=1 or vice versa

X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal_enc, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Compute class-specific accuracies
alive_label = le.transform(['alive'])[0]
dead_label = le.transform(['dead'])[0]
alive_mask = (y_test == alive_label)
dead_mask = (y_test == dead_label)
alive_accuracy = np.sum(y_pred[alive_mask] == y_test[alive_mask]) / np.sum(alive_mask)
dead_accuracy = np.sum(y_pred[dead_mask] == y_test[dead_mask]) / np.sum(dead_mask)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(f"Overall Accuracy: {accuracy:.2f}")
print(f"Alive Accuracy: {alive_accuracy:.2f}")
print(f"Dead Accuracy: {dead_accuracy:.2f}")