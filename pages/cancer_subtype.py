import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
import shap

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from matplotlib.lines import Line2D
from sklearn.metrics import (
    ConfusionMatrixDisplay, 
    accuracy_score, 
    balanced_accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)

# Load tab-delimited text file into a DataFrame
df_x = pd.read_csv('Data/cancer_subtype_data.csv')
df_y = pd.read_csv('Data/cancer_subtype_labels.csv')

### Basic Exploratory Data Analysis ####################################################################################
st.subheader("Basic Exploratory Data Analysis")

st.write("First few rows of the DataFrame:")
st.write(df_x.head())

st.write("First few rows of the DataFrame:")
st.write(df_y.head())

# Merge the two DataFrames
df = pd.merge(df_x, df_y)
st.write("First few rows of the DataFrame after merge:")
st.write(df.head())
st.write("Shape of the DataFrame after merge:")
st.write(df.shape)

# Drop the instance label column
df = df.drop(df.columns[0], axis=1)
st.write("DataFrame after dropping instance label column:")
st.write(df.head())

column_name = 'Class'

st.write("Number of duplicate rows:")
st.write(df.duplicated().sum())

# Missing Value Assessment
grouped_df = df.groupby('Class')
missing_values_per_class = grouped_df.apply(lambda x: x.isnull().sum().sum())

# Get the value counts for the specified column
value_counts = df[column_name].value_counts()
st.write("Class value counts:")
st.write(value_counts)

st.write("Sum of missing values for each class:")
st.write(missing_values_per_class)

# Count missing values in dataset
total_missing_values = df.isna().sum().sum()
st.write("Total number of missing values:", total_missing_values)

# Plot the value counts using seaborn
plt.figure(figsize=(6, 4))
sns.barplot(x=value_counts.index, y=value_counts.values)
plt.title(f'Counts of {column_name}')
plt.xlabel(column_name)
plt.ylabel('Count')
plt.xticks(rotation=0)
st.pyplot(plt)

### PCA ################################################################################################################
st.subheader("Principal Component Analysis")

df_x = df.drop(columns=['Class'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_x)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', marker='o', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('KMeans Clustering')
plt.colorbar(label='Cluster')
st.pyplot(plt)

# Variance Thresholding
features = df.drop(columns='Class')
variances = features.var()

st.write("Summary of Feature Variances:")
st.write(variances.describe())

st.subheader("Distribution of Gene Variances")
plt.figure(figsize=(8, 4))
sns.histplot(variances, bins=100, kde=True)
plt.xlabel("Variance")
plt.ylabel("Number of Genes")
st.pyplot(plt)

threshold = 0.5
selector = VarianceThreshold(threshold=threshold)
selector.fit(features)

selected_genes = features.columns[selector.get_support()]
st.write(f"Retained {len(selected_genes)} of {features.shape[1]} genes with variance > {threshold}")
df_filtered = df[selected_genes.tolist() + ['Class']]

### PCA After Variance Filtering ###
st.subheader("PCA After Variance Filtering")

df_x_filtered = df_filtered.drop(columns=['Class'])
scaler_filt = StandardScaler()
X_scaled_filt = scaler_filt.fit_transform(df_x_filtered)

pca_filt = PCA(n_components=2)
X_pca_filt = pca_filt.fit_transform(X_scaled_filt)

kmeans_filt = KMeans(n_clusters=5, random_state=42)
clusters_filt = kmeans_filt.fit_predict(X_scaled_filt)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca_filt[:, 0], X_pca_filt[:, 1], c=clusters_filt, cmap='viridis', marker='o', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('KMeans Clustering (After Variance Filtering)')
plt.colorbar(label='Cluster')
st.pyplot(plt)

st.write(df_filtered.shape)

### UMAP ################################################################################################################
st.subheader("UMAP Visualization (on Original Data)")

df_x_umap = df.drop(columns=['Class'])
scaler_umap = StandardScaler()
X_scaled_umap = scaler_umap.fit_transform(df_x_umap)

reducer = umap.UMAP(n_components=100, n_neighbors=5, min_dist=1, random_state=42)
df_umap_full = reducer.fit_transform(X_scaled_umap)

# Convert UMAP output to DataFrame and add Class
dmap_df = pd.DataFrame(df_umap_full, columns=[f"UMAP_{i+1}" for i in range(df_umap_full.shape[1])])
dmap_df["Class"] = df["Class"].values

label_encoder = LabelEncoder()
label_ids = label_encoder.fit_transform(df['Class'])
class_names = label_encoder.classes_

plt.figure(figsize=(8, 6))
scatter = plt.scatter(df_umap_full[:, 0], df_umap_full[:, 1], c=label_ids, cmap='tab10', alpha=0.6)
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.title("UMAP Projection (Original Data, Colored by Class)")

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=class_names[i],
           markerfacecolor=plt.cm.tab10(i / len(class_names)), markersize=8)
    for i in range(len(class_names))
]
plt.legend(handles=legend_elements, title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(plt)
st.write(df_umap_full.shape)

### Train and Evaluate Model ################################################################################################
def train_model(model, data):
    X = data.drop(columns=['Class'])
    y = data['Class']

    st.subheader(f"Model: {model.__class__.__name__}")

    accuracies, balanced_accuracies, precisions, recalls, f1_scores = [], [], [], [], []
    all_y_true, all_y_pred = [], []

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        accuracies.append(accuracy_score(y_test, y_pred))
        balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='weighted'))
        recalls.append(recall_score(y_test, y_pred, average='weighted'))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

    st.write("Average Accuracy:", np.mean(accuracies))
    st.write("Average Balanced Accuracy:", np.mean(balanced_accuracies))
    st.write("Average Precision:", np.mean(precisions))
    st.write("Average Recall:", np.mean(recalls))
    st.write("Average F1 Score:", np.mean(f1_scores))

    st.write("Confusion Matrix:")
    cm = confusion_matrix(all_y_true, all_y_pred, labels=np.unique(y))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    st.pyplot(fig)

    # Classification Report
    st.write("Classification Report:")
    report = classification_report(all_y_true, all_y_pred)
    st.text(report)

    # SHAP Feature Importance
    st.subheader("SHAP Feature Importance")
    model.fit(X, y)

    try:
        explainer = shap.Explainer(model, X)
    except Exception:
        background = shap.sample(X, 100, random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)

    shap_values = explainer(X)

    shap.summary_plot(
        shap_values,
        features=X,
        feature_names=X.columns,
        plot_type="bar",
        max_display=20,
        show=False,
    )
    st.pyplot(plt.gcf(), clear_figure=True)

# Train models on original, filtered, and UMAP data
for dataset, label in [(df, "Original"), (df_filtered, "Filtered"), (dmap_df, "UMAP")]:
    st.markdown(f"### Training on {label} Data")
    train_model(DecisionTreeClassifier(), dataset)
    train_model(RandomForestClassifier(), dataset)
    train_model(ExtraTreesClassifier(), dataset)