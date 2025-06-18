import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

# Load tab-delimited text file into a DataFrame
df_x = pd.read_csv('Data/cancer_subtype_data.csv')
st.write("First few rows of the DataFrame:")
st.write(df_x.head())

# Load tab-delimited text file into a DataFrame
df_y = pd.read_csv('Data/cancer_subtype_labels.csv')
st.write("First few rows of the DataFrame:")
st.write(df_y.head())

df = pd.merge(df_x, df_y)
st.write(df.head())
st.write(df.shape)

# Drop the instance label column
df = df.drop(df.columns[0], axis=1)
st.write(df.head())

### Basic Exploratory Data Analysis ###
column_name = 'Class'

# Get the value counts for the specified column
value_counts = df[column_name].value_counts()
st.write(value_counts)

# Plot the value counts using seaborn
plt.figure(figsize=(6, 4))
sns.barplot(x=value_counts.index, y=value_counts.values)
plt.title(f'Counts of {column_name}')
plt.xlabel(column_name)
plt.ylabel('Count')
plt.xticks(rotation=0)
st.pyplot(plt)

# Missing Value Assessment
grouped_df = df.groupby('Class')
missing_values_per_class = grouped_df.apply(lambda x: x.isnull().sum().sum())

st.write("Sum of missing values for each class:")
st.write(missing_values_per_class)

# Count missing values in dataset
total_missing_values = df.isna().sum().sum()
st.write("Total number of missing values:", total_missing_values)

### PCA ###
df_x = df.drop(columns=['Class'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_x)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', marker='o', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('KMeans Clustering')
plt.colorbar(label='Cluster')
st.pyplot(plt)

### Train and Evaluate Decision Tree Models (with 3-fold cross-validation) ###
def train_model(model):
    X = df.drop(columns=['Class'])
    y = df['Class']

    accuracies = []
    balanced_accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='weighted'))
        recalls.append(recall_score(y_test, y_pred, average='weighted'))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_balanced_accuracy = sum(balanced_accuracies) / len(balanced_accuracies)
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1 = sum(f1_scores) / len(f1_scores)

    st.write("Average Accuracy:", avg_accuracy)
    st.write("Average Balanced Accuracy:", avg_balanced_accuracy)
    st.write("Average Precision:", avg_precision)
    st.write("Average Recall:", avg_recall)
    st.write("Average F1 Score:", avg_f1)

# Train a decision tree model
model = DecisionTreeClassifier()
train_model(model)

# Train a random forest model
model = RandomForestClassifier()
train_model(model)
