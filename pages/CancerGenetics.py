import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
import shap

from matplotlib.lines import Line2D
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score,
    silhouette_score, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

# ──────────────────────────────────────────────
# Custom Transformer: Top-K Variance Selector
# ──────────────────────────────────────────────
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
        if input_features is None:
            return [f"f{i}" for i in self.topk_idx_]
        return np.array(input_features)[self.topk_idx_]

# ──────────────────────────────────────────────
# Helper 1: cluster-quality metrics
# ──────────────────────────────────────────────
def evaluate_clusters(name, X, cluster_labels, true_labels):
    y_true = true_labels.astype(str).values
    y_pred = pd.Series(cluster_labels).astype(str)

    scores = {
        "Adjusted Rand Index":    adjusted_rand_score(y_true, y_pred),
        "Normalized Mutual Info": normalized_mutual_info_score(y_true, y_pred),
        "Homogeneity":            homogeneity_score(y_true, y_pred),
        "Completeness":           completeness_score(y_true, y_pred),
        "V-measure":              v_measure_score(y_true, y_pred),
    }
    st.markdown(f"##### {name}: cluster quality vs. true classes")
    st.write({k: f"{v:.3f}" for k, v in scores.items()})

    if len(set(cluster_labels)) > 1 and X.shape[1] >= 2:
        st.write(f"Silhouette score: {silhouette_score(X, cluster_labels):.3f}")

# ──────────────────────────────────────────────
# Helper 2: two-column display
# ──────────────────────────────────────────────
def plot_step(name, X_2d, cluster_labels, X_full, true_labels):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"##### {name}: K-Means clusters")
        fig_c, ax_c = plt.subplots(figsize=(5, 4))
        pts = ax_c.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, cmap="viridis", alpha=0.6)
        ax_c.set_xlabel("Component 1"); ax_c.set_ylabel("Component 2")
        fig_c.colorbar(pts, ax=ax_c, label="Cluster")
        st.pyplot(fig_c)

    with col2:
        st.markdown(f"##### {name}: true class labels")
        label_codes, class_names = pd.factorize(true_labels)
        fig_t, ax_t = plt.subplots(figsize=(5, 4))
        ax_t.scatter(X_2d[:, 0], X_2d[:, 1], c=label_codes, cmap="tab10", alpha=0.6)
        ax_t.set_xlabel("Component 1"); ax_t.set_ylabel("Component 2")
        handles = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=plt.cm.tab10(i / max(1, len(class_names))),
                   markersize=8, label=cls)
            for i, cls in enumerate(class_names)
        ]
        ax_t.legend(handles=handles, title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(fig_t)

    evaluate_clusters(name, X_full, cluster_labels, true_labels)

# ──────────────────────────────────────────────
# Data Loading and Caching
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df_x = pd.read_csv("Data/cancer_subtype_data.csv")
    df_y = pd.read_csv("Data/cancer_subtype_labels.csv")
    return pd.merge(df_x, df_y).drop(df_x.columns[0], axis=1)

# ──────────────────────────────────────────────
# Embedding for Visualization (global, for EDA only)
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_embedding(df, method, **kwargs):
    X_raw = df.drop(columns="Class")
    X_scaled = StandardScaler().fit_transform(X_raw)
    if method == "PCA":
        X2d     = PCA(**kwargs).fit_transform(X_scaled)
        X_clust = X_scaled
    elif method == "UMAP":
        X_emb   = umap.UMAP(**kwargs).fit_transform(X_scaled)
        X2d     = X_emb[:, :2]
        X_clust = X_emb
    else:
        raise ValueError(f"Unknown method {method}")
    return X2d, X_clust

@st.cache_data(show_spinner=False)
def compute_clusters(X_clust, n_clusters=5, random_state=42):
    return KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(X_clust)

# ──────────────────────────────────────────────
# Main Modeling CV Pipeline (no leakage)
# ──────────────────────────────────────────────
def get_pipeline(tag):
    """ Returns sklearn Pipeline (no leakage) for each preprocessing variant. """
    # Note: All pipelines operate on np.array input (X), not DataFrame
    # All transformers fitted inside each CV fold only on training data
    if tag == "Original":
        steps = [("scaler", StandardScaler())]
    elif tag == "Threshold-Filtered":
        steps = [
            ("scaler", StandardScaler()),
            ("variance", VarianceThreshold(0.5)),
        ]
    elif tag == "Threshold-Filtered (log)":
        steps = [
            ("log", FunctionTransformer(np.log1p)),
            ("scaler", StandardScaler()),
            ("variance", VarianceThreshold(0.5)),
        ]
    elif tag == "Selection-Top-350":
        steps = [
            ("scaler", StandardScaler()),
            ("topk", TopKVarianceSelector(k=350)),
        ]
    elif tag == "Selection-Top-350 (log)":
        steps = [
            ("log", FunctionTransformer(np.log1p)),
            ("scaler", StandardScaler()),
            ("topk", TopKVarianceSelector(k=350)),
        ]
    elif tag == "Filtered-UMAP":
        # For modeling with UMAP features as input (note: fit UMAP per fold)
        def umap_transform(X):
            return umap.UMAP(
                n_components=100, n_neighbors=5, min_dist=1, random_state=42
            ).fit_transform(X)
        steps = [
            ("scaler", StandardScaler()),
            ("umap", FunctionTransformer(umap_transform, validate=False)),
        ]
    else:
        raise ValueError(tag)
    return steps

# ──────────────────────────────────────────────
# Model training and evaluation (safe, per-fold pipeline)
# ──────────────────────────────────────────────
def train_and_explain(df, pipeline_steps, model_cls, model_kwargs, seed=42):
    X = df.drop(columns="Class").values
    y = df["Class"].values
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    accs = []; bal_accs = []; precs = []; recs = []; f1s = []
    all_true = []; all_pred = []

    for tr, te in kf.split(X, y):
        pipe = Pipeline(pipeline_steps + [("clf", model_cls(**model_kwargs))])
        pipe.fit(X[tr], y[tr])
        preds = pipe.predict(X[te])
        all_true.extend(y[te]); all_pred.extend(preds)
        accs.append(accuracy_score(y[te], preds))
        bal_accs.append(balanced_accuracy_score(y[te], preds))
        precs.append(precision_score(y[te], preds, average="weighted"))
        recs.append(recall_score(y[te], preds, average="weighted"))
        f1s.append(f1_score(y[te], preds, average="weighted"))

    # SHAP: fit full pipeline on all data (no leakage for feature importance)
    full_pipe = Pipeline(pipeline_steps + [("clf", model_cls(**model_kwargs))])
    full_pipe.fit(X, y)
    try:
        explainer = shap.Explainer(full_pipe.named_steps["clf"], full_pipe[:-1].transform(X))
        sv = explainer(full_pipe[:-1].transform(X))
        fig_shap = plt.figure()
        shap.summary_plot(
            sv, features=full_pipe[:-1].transform(X),
            feature_names=[f"f{i}" for i in range(full_pipe[:-1].transform(X).shape[1])],
            class_names=full_pipe.named_steps["clf"].classes_,
            plot_type="bar", max_display=20, show=False)
        plt.close(fig_shap)
    except Exception as e:
        fig_shap = plt.figure()
        plt.text(0.5, 0.5, f"SHAP unavailable\n{e}", ha="center")
        plt.close(fig_shap)

    # confusion matrix (global, for visualization only)
    cm = confusion_matrix(all_true, all_pred, labels=np.unique(y))
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=np.unique(y)).plot(
        ax=ax_cm, cmap="Blues", xticks_rotation=45)
    plt.close(fig_cm)

    # label-shuffle baseline
    y_shuf = y.copy()
    rng = np.random.RandomState(seed)
    rng.shuffle(y_shuf)
    shuf_accs = []
    for tr, te in kf.split(X, y_shuf):
        pipe = Pipeline(pipeline_steps + [("clf", model_cls(**model_kwargs))])
        pipe.fit(X[tr], y_shuf[tr])
        shuf_accs.append(accuracy_score(y_shuf[te], pipe.predict(X[te])))

    class_props = pd.Series(y).value_counts(normalize=True)
    baseline = float((class_props**2).sum())

    return {
        "metrics": {
            "Accuracy": np.mean(accs),
            "Balanced Accuracy": np.mean(bal_accs),
            "Precision": np.mean(precs),
            "Recall": np.mean(recs),
            "F1 Score": np.mean(f1s),
        },
        "confusion_fig": fig_cm,
        "classification_report": classification_report(all_true, all_pred),
        "shap_fig": fig_shap,
        "shuffle": {
            "baseline": baseline,
            "shuffled_acc": np.mean(shuf_accs),
            "shuffled_std": np.std(shuf_accs)
        }
    }

# ──────────────────────────────────────────────
# Precompute everything (pipelines for modeling, global for EDA)
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def precompute_all():
    df = load_data()
    cluster_cfg = {
        "Original": (
            df, "PCA",
            {"n_components": 2, "random_state": 42},
            "PCA"
        ),
        "Threshold-Filtered": (
            df, "PCA",
            {"n_components": 2, "random_state": 42},
            "PCA (filtered)"
        ),
        "Threshold-Filtered (log)": (
            df, "PCA",
            {"n_components": 2, "random_state": 42},
            "PCA (filtered, log-scaled)"
        ),
        "Filtered-UMAP": (
            df, "UMAP",
            {"n_components": 100, "n_neighbors": 5, "min_dist": 1, "random_state": 42},
            "UMAP"
        ),
        "Selection-Top-350": (
            df, "PCA",
            {"n_components": 2, "random_state": 42},
            "PCA (top-350)"
        ),
        "Selection-Top-350 (log)": (
            df, "PCA",
            {"n_components": 2, "random_state": 42},
            "PCA (top-350, log-scaled)"
        ),
    }

    model_specs = [
        (DecisionTreeClassifier, {}, "Decision Tree"),
        (RandomForestClassifier, {}, "Random Forest"),
        (SGDClassifier, {"loss":"log_loss","penalty":"l2","max_iter":1000}, "SGD Classifier")
    ]

    results = {}
    for tag in cluster_cfg:
        results[tag] = {"clusters": [], "models": {}}
        # For clustering/EDA
        df_cl, method, m_kwargs, step_name = cluster_cfg[tag]
        X2d, Xcl = compute_embedding(df_cl, method, **m_kwargs)
        cl_labels = compute_clusters(Xcl)
        results[tag]["clusters"].append((step_name, X2d, cl_labels, Xcl, df_cl["Class"]))

        # Modeling
        pipeline_steps = get_pipeline(tag)
        for cls, kw, name in model_specs:
            results[tag]["models"][name] = train_and_explain(df_cl, pipeline_steps, cls, kw)
    return results

# ──────────────────────────────────────────────
# Streamlit page
# ──────────────────────────────────────────────
st.title("Cancer Subtype Analysis: CV-Safe Pipelines")

df = load_data()

# Basic EDA (unchanged)
st.subheader("Basic Exploratory Data Analysis")
st.write("Features:", df.head())
st.write("Labels:", df[["Class"]].head())
st.write("Shape:", df.shape)

class_counts = df["Class"].value_counts()
st.write("Class counts:", class_counts)
st.write("Total missing values:", df.isna().sum().sum())

fig_cls, ax_cls = plt.subplots(figsize=(6, 4))
sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax_cls)
ax_cls.set_title("Class distribution")
st.pyplot(fig_cls)

variances = df.drop(columns="Class").var()
st.write("Feature variance summary:", variances.describe())

fig_var, ax_var = plt.subplots(figsize=(8, 4))
sns.histplot(variances, bins=100, kde=True, ax=ax_var)
ax_var.set_xlabel("Variance"); ax_var.set_ylabel("Count")
st.pyplot(fig_var)

# Precompute
with st.spinner("🚀 Precomputing embeddings, clustering, training & SHAP…"):
    all_results = precompute_all()

# Display
for tag, data in all_results.items():
    with st.expander(f"📦 {tag}"):
        # clusters
        for step_name, X2d, cl_labels, Xfull, lbls in data["clusters"]:
            plot_step(step_name, X2d, cl_labels, Xfull, lbls)
        # models
        for model_name, mr in data["models"].items():
            st.markdown("---")
            st.subheader(f"Model: {model_name}")
            st.write(mr["metrics"])
            st.pyplot(mr["confusion_fig"])
            st.text(mr["classification_report"])
            st.pyplot(mr["shap_fig"])
            sh = mr["shuffle"]
            st.write(f"Baseline ∑pᵢ²: {sh['baseline']:.3f}")
            st.write(f"Shuffled‑label acc: {sh['shuffled_acc']:.3f} ± {sh['shuffled_std']:.3f}")
            if sh["shuffled_acc"] < sh["baseline"]:
                st.warning("⚠️ Shuffled‑label accuracy below theoretical baseline — check CV or leakage.")