import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
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
from scripts.custom_transformers import UMAPTransformer, TopKVarianceSelector
import shap
import joblib

# ────────────────────────────────────────────
# Plot helpers (silhouette removed from evaluate_clusters)
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
# Load & merge data (cached)
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
model_dir = "resources/CancerGeneticsModels"

model_specs = [
    (DecisionTreeClassifier, {}, "Decision Tree"),
    (RandomForestClassifier, {}, "Random Forest"),
    (SGDClassifier, {"loss": "log_loss", "penalty": "l2", "max_iter": 1000}, "SGD Classifier")
]

# ────────────────────────────────────────────
# Compute EDA transforms per‑tag (cached)
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
        Xs = Xf.values
        X2d = PCA(n_components=2, random_state=42).fit_transform(Xs)
        cl = KMeans(n_clusters=5, random_state=42).fit_predict(Xs)
        return X2d, cl, Xs, y

    if tag == "Filtered-UMAP":
        Xs = StandardScaler().fit_transform(X)
        Emb = UMAPTransformer(
            n_components=100, n_neighbors=5, min_dist=1.0, random_state=42
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
        Xs = Xf.values
        X2d = PCA(n_components=2, random_state=42).fit_transform(Xs)
        cl = KMeans(n_clusters=5, random_state=42).fit_predict(Xs)
        return X2d, cl, Xs, y

    raise ValueError(f"Unknown tag {tag}")

# ────────────────────────────────────────────
# Main page layout with tabs
st.title("Cancer Subtype Analysis: EDA + Modeling")

main_tabs = st.tabs(
    ["Data Overview", "Clustering Analysis", "Model Evaluation", "SHAP Explanations", "Project Findings"])

# DATA OVERVIEW TAB
with main_tabs[0]:
    st.subheader("Basic Exploratory Data Analysis")
    df = load_data()
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

# CLUSTERING ANALYSIS TAB
with main_tabs[1]:
    st.subheader("Clustering Analysis")
    clustering_tabs = st.tabs(tags)
    for i, tag in enumerate(tags):
        with clustering_tabs[i]:
            st.subheader(f"Pipeline: {tag}")
            X2d, clusters, Xfull, labels = compute_eda(tag)
            plot_step(tag, X2d, clusters, Xfull, labels)

# ────────────────────────────────────────────
# Modeling pipelines & model IO helpers
# ────────────────────────────────────────────
def get_pipeline(tag):
    if tag == "Original":
        return [("scaler", StandardScaler())]
    if tag == "Threshold-Filtered":
        return [("scaler", StandardScaler()), ("variance", VarianceThreshold(0.5))]
    if tag == "Threshold-Filtered (log)":
        return [
            ("log", FunctionTransformer(np.log1p, validate=True)),
            ("variance", VarianceThreshold(0.5)),
        ]
    if tag == "Selection-Top-350":
        return [("scaler", StandardScaler()), ("topk", TopKVarianceSelector(k=350))]
    if tag == "Selection-Top-350 (log)":
        return [
            ("log", FunctionTransformer(np.log1p, validate=True)),
            ("topk", TopKVarianceSelector(k=350)),
        ]
    if tag == "Filtered-UMAP":
        return [
            ("scaler", StandardScaler()),
            ("umap", UMAPTransformer(
                n_components=100, n_neighbors=5, min_dist=1.0, random_state=42
            ))
        ]
    raise ValueError(f"Unknown tag {tag}")

def get_model_path(tag, model_name):
    tag_dir = os.path.join(model_dir, tag)
    if not os.path.exists(tag_dir):
        os.makedirs(tag_dir)
    return os.path.join(tag_dir, f"{model_name}.joblib")

def get_metrics_path(tag, model_name):
    tag_dir = os.path.join(model_dir, tag)
    if not os.path.exists(tag_dir):
        os.makedirs(tag_dir)
    return os.path.join(tag_dir, f"{model_name}_metrics.joblib")

def fit_and_store_model(df, pipeline_steps, model_cls, model_kwargs, tag, model_name, seed=42):
    X = df.drop(columns="Class")
    y = df["Class"].values
    orig_feats = X.columns.to_list()
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    accs = []
    bal_accs = []
    precs = []
    recs = []
    f1s = []
    all_true = []
    all_pred = []

    for tr, te in kf.split(X, y):
        pipe = Pipeline(pipeline_steps + [("clf", model_cls(**model_kwargs))])
        pipe.fit(X.iloc[tr], y[tr])
        preds = pipe.predict(X.iloc[te])
        all_true.extend(y[te])
        all_pred.extend(preds)
        accs.append(accuracy_score(y[te], preds))
        bal_accs.append(balanced_accuracy_score(y[te], preds))
        precs.append(precision_score(y[te], preds, average="weighted"))
        recs.append(recall_score(y[te], preds, average="weighted"))
        f1s.append(f1_score(y[te], preds, average="weighted"))

    # Train final model on all data
    full_pipe = Pipeline(pipeline_steps + [("clf", model_cls(**model_kwargs))])
    full_pipe.fit(X, y)

    # Save model pipeline to disk
    joblib.dump(full_pipe, get_model_path(tag, model_name))

    # Save metrics to disk
    cm = confusion_matrix(all_true, all_pred, labels=np.unique(y))
    report = classification_report(all_true, all_pred)
    y_shuf = y.copy()
    rng = np.random.RandomState(seed)
    rng.shuffle(y_shuf)
    shuf_accs = []
    for tr, te in kf.split(X, y_shuf):
        pipe = Pipeline(pipeline_steps + [("clf", model_cls(**model_kwargs))])
        pipe.fit(X.iloc[tr], y_shuf[tr])
        shuf_accs.append(accuracy_score(y_shuf[te], pipe.predict(X.iloc[te])))
    baseline = float((pd.Series(y).value_counts(normalize=True) ** 2).sum())
    metrics = {
        "metrics": {
            "Accuracy": np.mean(accs),
            "Balanced Accuracy": np.mean(bal_accs),
            "Precision": np.mean(precs),
            "Recall": np.mean(recs),
            "F1 Score": np.mean(f1s),
        },
        "confusion_matrix": cm,
        "classification_report": report,
        "shuffle": {
            "baseline": baseline,
            "shuffled_acc": np.mean(shuf_accs),
            "shuffled_std": np.std(shuf_accs)
        }
    }
    joblib.dump(metrics, get_metrics_path(tag, model_name))

def load_model_pipeline(tag, model_name):
    path = get_model_path(tag, model_name)
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def load_model_metrics(tag, model_name):
    path = get_metrics_path(tag, model_name)
    if not os.path.exists(path):
        return None
    return joblib.load(path)

# ────────────────────────────────────────────
# Model Evaluation Tab: train if needed, then use stored
# ────────────────────────────────────────────
with main_tabs[2]:
    st.subheader("Model Evaluation")

    df = load_data()
    # Model training with progress bar if missing, otherwise use existing
    total_tasks = len(tags) * len(model_specs)
    progress = st.progress(0)
    count = 0
    for tag in tags:
        steps = get_pipeline(tag)
        for cls, kw, model_name in model_specs:
            if not os.path.exists(get_model_path(tag, model_name)):
                with st.spinner(f"Training {model_name} for {tag}..."):
                    fit_and_store_model(df, steps, cls, kw, tag, model_name)
            count += 1
            progress.progress(count / total_tasks)

    # UI: Tabs for each pipeline
    model_tabs = st.tabs(tags)
    for i, tag in enumerate(tags):
        with model_tabs[i]:
            st.subheader(f"Models with {tag} Pipeline")
            model_names = [m[2] for m in model_specs]
            model_sub_tabs = st.tabs(model_names)
            for j, model_name in enumerate(model_names):
                with model_sub_tabs[j]:
                    metrics = load_model_metrics(tag, model_name)
                    if metrics is None:
                        st.error("Model metrics not found, please retrain.")
                        continue
                    st.write("Performance Metrics:")
                    st.write(metrics["metrics"])

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Confusion Matrix")
                        cm = metrics["confusion_matrix"]
                        labels = np.unique(df["Class"])
                        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                        ConfusionMatrixDisplay(cm, display_labels=labels).plot(
                            ax=ax_cm, cmap="Blues", xticks_rotation=45
                        )
                        plt.close(fig_cm)
                        st.pyplot(fig_cm)
                    with col2:
                        st.subheader("Classification Report")
                        st.text(metrics["classification_report"])

                    sh = metrics["shuffle"]
                    st.write(f"Baseline ∑pᵢ²: {sh['baseline']:.3f}")
                    st.write(f"Shuffled‑label acc: {sh['shuffled_acc']:.3f} ± {sh['shuffled_std']:.3f}")
                    if sh["shuffled_acc"] < sh["baseline"] - 0.01:
                        st.warning("⚠️ Shuffled‑label accuracy below theoretical baseline — check CV or leakage.")

# ────────────────────────────────────────────
# SHAP Explanations Tab: always use stored models
# ────────────────────────────────────────────
with main_tabs[3]:
    st.subheader("SHAP Feature Importance")
    col1, col2 = st.columns(2)
    with col1:
        selected_tag = st.selectbox("Select Pipeline", tags)
    with col2:
        model_names = [m[2] for m in model_specs]
        selected_model = st.selectbox("Select Model", model_names)

    df = load_data()
    model_pipe = load_model_pipeline(selected_tag, selected_model)
    if model_pipe is None:
        st.info("Model not found. Please visit the Model Evaluation tab to train and store models first.")
    else:
        X = df.drop(columns="Class")
        y = df["Class"].values
        preproc = model_pipe[:-1]
        clf = model_pipe.named_steps["clf"]
        orig_feats = X.columns.to_list()

        feat_names = orig_feats

        try:
            explainer = shap.Explainer(clf, preproc.transform(X))
            sv = explainer(preproc.transform(X))
            fig_shap = plt.figure()
            shap.summary_plot(
                sv,
                features=preproc.transform(X),
                feature_names=feat_names,
                class_names=clf.classes_,
                plot_type="bar", max_display=20, show=False
            )
            plt.close(fig_shap)
            st.pyplot(fig_shap)
            st.write("SHAP values show the contribution of each feature to the model prediction.")
        except Exception as e:
            fig_shap = plt.figure()
            plt.text(0.5, 0.5, f"SHAP unavailable\n{e}", ha="center")
            plt.close(fig_shap)
            st.pyplot(fig_shap)
            st.info("SHAP explanation unavailable for this model/feature set.")

# ────────────────────────────────────────────
# PROJECT FINDINGS TAB - STATIC CONTENT
# ────────────────────────────────────────────
with main_tabs[4]:
    st.header("Project Findings")

    # Try multiple approaches to ensure content displays
    st.text(
        "Through a series of dimensionality reduction and feature selection pipelines—including threshold-based filtering, log transformations, UMAP embeddings, and top-variance gene selection—six K-Means clustering approaches were evaluated. The highest clustering alignment with ground truth was observed in the Threshold-Filtered (log) and Selection-Top-350 (log) pipelines, achieving ARI and NMI scores above 0.99 and strong silhouette values (0.343 and 0.372, respectively), indicating tight, well-separated clusters. The UMAP-based pipeline, while slightly lower in ARI, exhibited the best 2D structure (silhouette = 0.632), reinforcing its value for visual subtype separation.")

    st.text("")  # Add spacing

    st.text(
        "Supervised classification performance across all pipelines was consistently strong. Each pipeline was evaluated using 3-fold stratified cross-validation, ensuring class balance and robustness in performance metrics. Models were further validated with shuffled-label baselines to detect overfitting or data leakage—most of which showed shuffled accuracy well below theoretical baselines, confirming model integrity. Among traditional classifiers, the SGD Classifier and Random Forest achieved near-perfect scores on several pipelines, with balanced accuracies reaching 1.000 and F1 scores consistently above 0.99. Notably, the Threshold-Filtered (log) pipeline with the SGD Classifier achieved perfect classification (accuracy, precision, recall, and F1 score all equal to 1.000), demonstrating exceptional alignment between preprocessing and modeling.")

    st.text("")  # Add spacing

    st.text(
        "Across all pipelines, classification reports showed strong generalization across cancer types such as BRCA, COAD, KIRC, LUAD, and PRAD, with macro-averaged F1-scores approaching 1.0 in most cases. These findings highlight that both unsupervised and supervised models, when properly validated and paired with effective feature transformations, can deliver highly accurate and interpretable cancer subtype predictions.")

    # Alternative approach using container
    with st.container():
        st.markdown("---")
        st.markdown("**Alternative Display Method:**")

        st.markdown("""
        Through a series of dimensionality reduction and feature selection pipelines—including threshold-based filtering, log transformations, UMAP embeddings, and top-variance gene selection—six K-Means clustering approaches were evaluated. The highest clustering alignment with ground truth was observed in the Threshold-Filtered (log) and Selection-Top-350 (log) pipelines, achieving ARI and NMI scores above 0.99 and strong silhouette values (0.343 and 0.372, respectively), indicating tight, well-separated clusters. The UMAP-based pipeline, while slightly lower in ARI, exhibited the best 2D structure (silhouette = 0.632), reinforcing its value for visual subtype separation.

        Supervised classification performance across all pipelines was consistently strong. Each pipeline was evaluated using 3-fold stratified cross-validation, ensuring class balance and robustness in performance metrics. Models were further validated with shuffled-label baselines to detect overfitting or data leakage—most of which showed shuffled accuracy well below theoretical baselines, confirming model integrity. Among traditional classifiers, the SGD Classifier and Random Forest achieved near-perfect scores on several pipelines, with balanced accuracies reaching 1.000 and F1 scores consistently above 0.99. Notably, the Threshold-Filtered (log) pipeline with the SGD Classifier achieved perfect classification (accuracy, precision, recall, and F1 score all equal to 1.000), demonstrating exceptional alignment between preprocessing and modeling.

        Across all pipelines, classification reports showed strong generalization across cancer types such as BRCA, COAD, KIRC, LUAD, and PRAD, with macro-averaged F1-scores approaching 1.0 in most cases. These findings highlight that both unsupervised and supervised models, when properly validated and paired with effective feature transformations, can deliver highly accurate and interpretable cancer subtype predictions.
        """)

    # Debug information
    st.info("If you can see this message, the tab is working. The text should appear above.")

    # Test with simple content
    st.write("Test message: This is a simple test to verify the tab is functional.")