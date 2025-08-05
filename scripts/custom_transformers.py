import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import umap.umap_ as umap

class TopKVarianceSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=350):
        self.k = k

    def fit(self, X, y=None):
        self.variances_ = np.var(X, axis=0)
        self.topk_idx_ = np.argsort(self.variances_)[-self.k:]
        return self

    def transform(self, X):
        return X[:, self.topk_idx_]

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features)[self.topk_idx_]

class UMAPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=100, n_neighbors=5, min_dist=1, random_state=42):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self.umap_ = None

    def fit(self, X, y=None):
        self.umap_ = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state
        )
        self.umap_.fit(X)
        return self

    def transform(self, X):
        return self.umap_.transform(X)
