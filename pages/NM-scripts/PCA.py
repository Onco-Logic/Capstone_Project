import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

n_pca_components = 29 # Number of principal components
n_top_features = 10 # Number of features per component

# Load dataset
file_name = '../../Data/NM-datasets/Breast_Cancer.csv'
df = pd.read_csv(file_name)

# Preprocess
X = df.drop('Status', axis=1)

# Identify numerical and categorical features
numerical_features = ['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months']
categorical_features = [col for col in X.columns if col not in numerical_features]

# Create preprocessing pipelines
numerical_pipeline = StandardScaler()
categorical_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

# Create ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ],
    remainder='passthrough'
)

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)
if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()

# Get feature names after preprocessing
feature_names_processed = preprocessor.get_feature_names_out()

# Do Principal Component Analysis
pca = PCA(n_components=n_pca_components)
X_pca_scores = pca.fit_transform(X_processed)

# Captured Variance
explained_variances = pca.explained_variance_ratio_

print(f"\n--- Captured Variance by each of the {n_pca_components} Principal Components ---")
for i, ratio in enumerate(explained_variances):
    print(f"  PC{i+1}: {ratio:.4f} {ratio*100:.2f}% of total variance")
print(f"Total variance explained by these {n_pca_components} components: {np.sum(explained_variances):.4f} ({np.sum(explained_variances)*100:.2f}%)")

# Loadings: Positive and Negative
loadings_matrix = pca.components_

# Features are rows and PCs are columns
loadings_df = pd.DataFrame(
    loadings_matrix.T,
    columns=[f'PC{i+1}' for i in range(n_pca_components)],
    index=feature_names_processed
)

print("\n--- Loadings: Top Contributing Features for Each Principal Component ---")
for i in range(n_pca_components):
    pc_name = f'PC{i+1}'
    print(f"\n{pc_name} {explained_variances[i]*100:.2f}% Variance ---")
    
    # Get the series of loadings for the current PC
    pc_loadings_series = loadings_df[pc_name]
    
    # Sort by absolute value to find top contributors, then get their loading values
    top_features_indices = pc_loadings_series.abs().sort_values(ascending=False).head(n_top_features).index
    top_features_with_loadings = loadings_df.loc[top_features_indices, pc_name]
    
    print("Features:")
    print("----------")
    for feature_name, loading_value in top_features_with_loadings.items():
        print(f"{feature_name:<40} {loading_value:+.3f}")