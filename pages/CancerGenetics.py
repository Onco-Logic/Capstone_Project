import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap

from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay, silhouette_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import shap

# ────────────────────────────────────────────
# 1) Custom Transformer: Top‑K Variance Selector
# ────────────────────────────────────────────
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

# ────────────────────────────────────────────
# 1b) Custom Transformer: UMAP (fits per fold)
# ────────────────────────────────────────────
class UMAPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=100, n_neighbors=5, min_dist=1, random_state=42):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
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

# ────────────────────────────────────────────
# 2) Plot helpers (silhouette removed from evaluate_clusters)
# ────────────────────────────────────────────
def evaluate_clusters(name, X, cluster_labels, true_labels):
    from sklearn.metrics import (
        adjusted_rand_score, normalized_mutual_info_score,
        homogeneity_score, completeness_score, v_measure_score
    )
    y_true = true_labels.astype(str).values
    y_pred = pd.Series(cluster_labels).astype(str)
    scores = {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "Homogeneity": homogeneity_score(y_true, y_pred),
        "Completeness": completeness_score(y_true, y_pred),
        "V-measure": v_measure_score(y_true, y_pred),
    }
    st.write({k: f"{v:.3f}" for k, v in scores.items()})
    # silhouette handled in plot_step

def plot_step(name, X2d, clust, Xfull, labels):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### {name}: K‑Means clusters")
        fig, ax = plt.subplots(figsize=(5, 4))
        pts = ax.scatter(X2d[:, 0], X2d[:, 1], c=clust, cmap="viridis", alpha=0.6)
        fig.colorbar(pts, ax=ax, label="Cluster")
        st.pyplot(fig)
    with col2:
        st.markdown(f"#### {name}: True labels")
        codes, classes = pd.factorize(labels)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(X2d[:, 0], X2d[:, 1], c=codes, cmap="tab10", alpha=0.6)
        handles = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=plt.cm.tab10(i / max(1, len(classes))),
                   label=cls)
            for i, cls in enumerate(classes)
        ]
        ax.legend(handles=handles, title="Class",
                  bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(fig)

    # cluster metrics
    evaluate_clusters(name, Xfull, clust, labels)
    # correct silhouette
    if name == "Filtered-UMAP":
        st.write(f"Silhouette (2D): {silhouette_score(X2d, clust):.3f}")
    elif Xfull.shape[1] >= 2:
        st.write(f"Silhouette: {silhouette_score(Xfull, clust):.3f}")

# ────────────────────────────────────────────
# 3) Load & merge data (cached)
# ────────────────────────────────────────────
@st.cache_data
def load_data():
    df_x = pd.read_csv("Data/cancer_subtype_data.csv")
    df_y = pd.read_csv("Data/cancer_subtype_labels.csv")
    df = pd.merge(df_x, df_y).drop(df_x.columns[0], axis=1)
    return df

# Common list of tags
tags = [
    "Original",
    "Threshold-Filtered",
    "Threshold-Filtered (log)",
    "Filtered-UMAP",
    "Selection-Top-350",
    "Selection-Top-350 (log)"
]

# ────────────────────────────────────────────
# 4) Compute EDA transforms per‑tag (cached)
# ────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_eda(tag: str):
    df = load_data()
    X = df.drop(columns="Class")
    y = df["Class"]

    if tag == "Original":
        Xs = StandardScaler().fit_transform(X)
        X2d = PCA(n_components=2, random_state=42).fit_transform(Xs)
        cl = KMeans(n_clusters=5, random_state=42).fit_predict(Xs)
        return X2d, cl, Xs, y

    if tag == "Threshold-Filtered":
        sel = VarianceThreshold(0.5).fit(X)
        Xf = X.loc[:, sel.get_support()]
        Xs = StandardScaler().fit_transform(Xf)
        X2d = PCA(n_components=2, random_state=42).fit_transform(Xs)
        cl = KMeans(n_clusters=5, random_state=42).fit_predict(Xs)
        return X2d, cl, Xs, y

    if tag == "Threshold-Filtered (log)":
        Xlog = np.log1p(X)
        sel = VarianceThreshold(0.5).fit(Xlog)
        Xf = Xlog.loc[:, sel.get_support()]
        # Removed StandardScaler here:
        Xs = Xf.values
        X2d = PCA(n_components=2, random_state=42).fit_transform(Xs)
        cl = KMeans(n_clusters=5, random_state=42).fit_predict(Xs)
        return X2d, cl, Xs, y

    if tag == "Filtered-UMAP":
        Xs = StandardScaler().fit_transform(X)
        Emb = umap.UMAP(
            n_components=100, n_neighbors=5, min_dist=1, random_state=42
        ).fit_transform(Xs)
        X2d = Emb[:, :2]
        cl = KMeans(n_clusters=5, random_state=42).fit_predict(Emb)
        return X2d, cl, Emb, y

    if tag == "Selection-Top-350":
        top = X.var().sort_values(ascending=False).head(350).index
        Xf = X[top]
        Xs = StandardScaler().fit_transform(Xf)
        X2d = PCA(n_components=2, random_state=42).fit_transform(Xs)
        cl = KMeans(n_clusters=5, random_state=42).fit_predict(Xs)
        return X2d, cl, Xs, y

    if tag == "Selection-Top-350 (log)":
        Xlog = np.log1p(X)
        top = Xlog.var().sort_values(ascending=False).head(350).index
        Xf = Xlog[top]
        # Removed StandardScaler here:
        Xs = Xf.values
        X2d = PCA(n_components=2, random_state=42).fit_transform(Xs)
        cl = KMeans(n_clusters=5, random_state=42).fit_predict(Xs)
        return X2d, cl, Xs, y

    raise ValueError(f"Unknown tag {tag}")

# ────────────────────────────────────────────
# 5) Main page layout
# ────────────────────────────────────────────
st.title("Cancer Subtype Analysis: EDA + Modeling")

df = load_data()
st.subheader("Basic Exploratory Data Analysis")
st.write("Merged shape:", df.shape)
st.write("Class counts:", df["Class"].value_counts())
st.write("Missing values:", int(df.isna().sum().sum()))

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x="Class", data=df, ax=ax)
st.pyplot(fig)

variances = df.drop(columns="Class").var()
st.write("Variance summary:", variances.describe())
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(variances, bins=100, kde=True, ax=ax)
st.pyplot(fig)

for tag in tags:
    st.subheader(f"Pipeline: {tag}")
    X2d, clusters, Xfull, labels = compute_eda(tag)
    plot_step(tag, X2d, clusters, Xfull, labels)

# ────────────────────────────────────────────
# 6) Modeling pipelines & SHAP
# ────────────────────────────────────────────
def get_pipeline(tag):
    if tag == "Original":
        return [("scaler", StandardScaler())]
    if tag == "Threshold-Filtered":
        return [("scaler", StandardScaler()), ("variance", VarianceThreshold(0.5))]
    if tag == "Threshold-Filtered (log)":
        # REMOVED StandardScaler here:
        return [
            ("log", FunctionTransformer(np.log1p, validate=True)),
            ("variance", VarianceThreshold(0.5)),
        ]
    if tag == "Selection-Top-350":
        return [("scaler", StandardScaler()), ("topk", TopKVarianceSelector(k=350))]
    if tag == "Selection-Top-350 (log)":
        # REMOVED StandardScaler here:
        return [
            ("log", FunctionTransformer(np.log1p, validate=True)),
            ("topk", TopKVarianceSelector(k=350)),
        ]
    if tag == "Filtered-UMAP":
        return [
            ("scaler", StandardScaler()),
            ("umap", UMAPTransformer(
                n_components=100, n_neighbors=5, min_dist=1, random_state=42
            ))
        ]
    raise ValueError(f"Unknown tag {tag}")

def train_and_explain(df, pipeline_steps, model_cls, model_kwargs, seed=42):
    # Keep X as a DataFrame to preserve column (gene) names
    X = df.drop(columns="Class")
    y = df["Class"].values
    orig_feats = X.columns.to_list()
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    accs=[]; bal_accs=[]; precs=[]; recs=[]; f1s=[]
    all_true=[]; all_pred=[]

    for tr, te in kf.split(X, y):
        pipe = Pipeline(pipeline_steps + [("clf", model_cls(**model_kwargs))])
        # Use .iloc for DataFrame indexing
        pipe.fit(X.iloc[tr], y[tr])
        preds = pipe.predict(X.iloc[te])
        all_true.extend(y[te]); all_pred.extend(preds)
        accs.append(accuracy_score(y[te], preds))
        bal_accs.append(balanced_accuracy_score(y[te], preds))
        precs.append(precision_score(y[te], preds, average="weighted"))
        recs.append(recall_score(y[te], preds, average="weighted"))
        f1s.append(f1_score(y[te], preds, average="weighted"))

    full_pipe = Pipeline(pipeline_steps + [("clf", model_cls(**model_kwargs))])
    full_pipe.fit(X, y)

    preproc = full_pipe[:-1]
    try:
        feat_names = preproc.get_feature_names_out(orig_feats)
    except:
        feat_names = orig_feats

    try:
        explainer = shap.Explainer(full_pipe.named_steps["clf"], preproc.transform(X))
        sv = explainer(preproc.transform(X))
        fig_shap = plt.figure()
        shap.summary_plot(
            sv,
            features=preproc.transform(X),
            feature_names=feat_names,
            class_names=full_pipe.named_steps["clf"].classes_,
            plot_type="bar", max_display=20, show=False
        )
        plt.close(fig_shap)
    except Exception as e:
        fig_shap = plt.figure()
        plt.text(0.5,0.5,f"SHAP unavailable\n{e}",ha="center")
        plt.close(fig_shap)

    cm = confusion_matrix(all_true, all_pred, labels=np.unique(y))
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=np.unique(y)).plot(
        ax=ax_cm, cmap="Blues", xticks_rotation=45
    )
    plt.close(fig_cm)

    y_shuf = y.copy()
    rng = np.random.RandomState(seed)
    rng.shuffle(y_shuf)
    shuf_accs=[]
    for tr, te in kf.split(X, y_shuf):
        pipe = Pipeline(pipeline_steps + [("clf", model_cls(**model_kwargs))])
        pipe.fit(X.iloc[tr], y_shuf[tr])
        shuf_accs.append(accuracy_score(y_shuf[te], pipe.predict(X.iloc[te])))

    baseline = float((pd.Series(y).value_counts(normalize=True)**2).sum())
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

# ────────────────────────────────────────────
# 7) Precompute models with live progress bar
# ────────────────────────────────────────────
def precompute_models():
    df = load_data()
    model_specs = [
        (DecisionTreeClassifier, {}, "Decision Tree"),
        (RandomForestClassifier, {}, "Random Forest"),
        (SGDClassifier, {"loss":"log_loss","penalty":"l2","max_iter":1000}, "SGD Classifier")
    ]

    total_tasks = len(tags) * len(model_specs)
    progress = st.progress(0)
    count = 0
    results = {}

    for tag in tags:
        steps = get_pipeline(tag)
        results[tag] = {}
        for cls, kw, name in model_specs:
            results[tag][name] = train_and_explain(df, steps, cls, kw)
            count += 1
            progress.progress(count / total_tasks)

    return results

# 8) Run training spinner + display
with st.spinner("🚀 Training all models…"):
    all_model_results = precompute_models()

for tag in tags:
    with st.expander(f"Models: {tag}", expanded=False):
        for model_name, mr in all_model_results[tag].items():
            st.subheader(model_name)
            st.write(mr["metrics"])
            st.pyplot(mr["confusion_fig"])
            st.text(mr["classification_report"])
            st.pyplot(mr["shap_fig"])
            sh = mr["shuffle"]
            st.write(f"Baseline ∑pᵢ²: {sh['baseline']:.3f}")
            st.write(f"Shuffled‑label acc: {sh['shuffled_acc']:.3f} ± {sh['shuffled_std']:.3f}")
            if sh["shuffled_acc"] < sh["baseline"]:
                st.warning("⚠️ Shuffled‑label accuracy below theoretical baseline — check CV or leakage.")
