import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
import shap

from matplotlib.lines import Line2D
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score,
    silhouette_score
)

# ───────────────────────────────────────────────────────────
# helper 1 ── cluster-quality metrics (no heat-map)
# ───────────────────────────────────────────────────────────
def evaluate_clusters(name, X, cluster_labels, true_labels):
    """Print external indices + silhouette (dtype mismatch handled)."""
    y_true = true_labels.astype(str).values
    y_pred = pd.Series(cluster_labels).astype(str)

    scores = {
        "Adjusted Rand Index":        adjusted_rand_score(y_true, y_pred),
        "Normalized Mutual Info":     normalized_mutual_info_score(y_true, y_pred),
        "Homogeneity":                homogeneity_score(y_true, y_pred),
        "Completeness":               completeness_score(y_true, y_pred),
        "V-measure":                  v_measure_score(y_true, y_pred),
    }
    st.markdown(f"##### {name}: cluster quality vs. true classes")
    st.write({k: f"{v:.3f}" for k, v in scores.items()})

    if len(set(cluster_labels)) > 1 and X.shape[1] >= 2:
        st.write(f"Silhouette score: {silhouette_score(X, cluster_labels):.3f}")

# ───────────────────────────────────────────────────────────
# helper 2 ── two-column display: clusters | true labels
# ───────────────────────────────────────────────────────────
def plot_step(name, X_2d, cluster_labels, X_full, true_labels):
    """Column-1: clusters; Column-2: true labels; metrics underneath."""
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
        sc = ax_t.scatter(
            X_2d[:, 0], X_2d[:, 1],
            c=label_codes, cmap="tab10", alpha=0.6
        )
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

# ───────────────────────────────────────────────────────────
# data loading & EDA
# ───────────────────────────────────────────────────────────
df_x = pd.read_csv("Data/cancer_subtype_data.csv")
df_y = pd.read_csv("Data/cancer_subtype_labels.csv")

st.subheader("Basic Exploratory Data Analysis")
st.write("Features:", df_x.head())
st.write("Labels:", df_y.head())

df = pd.merge(df_x, df_y).drop(df_x.columns[0], axis=1)
st.write("Merged shape:", df.shape)

class_counts = df["Class"].value_counts()
st.write("Class counts:", class_counts)
st.write("Total missing values:", df.isna().sum().sum())

fig_cls, ax_cls = plt.subplots(figsize=(6, 4))
sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax_cls)
ax_cls.set_title("Class distribution")
st.pyplot(fig_cls)

features = df.drop(columns='Class')
variances = features.var()

st.write("Summary of Feature Variances:")
st.write(variances.describe())

st.write("Distribution of Gene Variances")
fig_var, ax_var = plt.subplots(figsize=(8, 4))
sns.histplot(variances, bins=100, kde=True, ax=ax_var)
ax_var.set_xlabel("Variance")
ax_var.set_ylabel("Number of Genes")
st.pyplot(fig_var)

# ───────────────────────────────────────────────────────────
# 1) PCA on full feature space
# ───────────────────────────────────────────────────────────
st.subheader("PCA on full feature space")
X_full = StandardScaler().fit_transform(df.drop(columns="Class"))
X_pca_full = PCA(n_components=2).fit_transform(X_full)
clusters_full = KMeans(n_clusters=5, random_state=42).fit_predict(X_full)
plot_step("PCA", X_pca_full, clusters_full, X_full, df["Class"])

# ───────────────────────────────────────────────────────────
# 2) variance-threshold → PCA
# ───────────────────────────────────────────────────────────
st.subheader("Variance threshold → PCA")
selector = VarianceThreshold(0.5).fit(df.drop(columns="Class"))
df_filt = df.iloc[:, list(selector.get_support(indices=True)) + [-1]]
X_filt = StandardScaler().fit_transform(df_filt.drop(columns="Class"))
X_pca_filt = PCA(n_components=2).fit_transform(X_filt)
clusters_filt = KMeans(n_clusters=5, random_state=42).fit_predict(X_filt)
plot_step("PCA (filtered)", X_pca_filt, clusters_filt, X_filt, df_filt["Class"])

st.subheader("Variance threshold → PCA (with Log Scaling)")
df_log_scaled = pd.DataFrame(np.log1p(df.drop(columns="Class")), columns=df.drop(columns="Class").columns)
selector_log = VarianceThreshold(0.5).fit(df_log_scaled)
df_filt_log = df.iloc[:, list(selector_log.get_support(indices=True)) + [-1]]
X_filt_log = np.log1p(df_filt_log.drop(columns="Class"))
X_filt_log_scaled = StandardScaler().fit_transform(X_filt_log)
X_pca_filt_log = PCA(n_components=2).fit_transform(X_filt_log_scaled)
clusters_filt_log = KMeans(n_clusters=5, random_state=42).fit_predict(X_filt_log_scaled)
plot_step("PCA (filtered, log-scaled)", X_pca_filt_log, clusters_filt_log, X_filt_log_scaled, df_filt_log["Class"])

# ───────────────────────────────────────────────────────────
# 3) UMAP (100-D) + K-Means
# ───────────────────────────────────────────────────────────
st.subheader("UMAP + K-Means")
X_umap_in = StandardScaler().fit_transform(df.drop(columns="Class"))
umap_emb = umap.UMAP(n_components=100, n_neighbors=5, min_dist=1, random_state=42).fit_transform(X_umap_in)
clusters_umap = KMeans(n_clusters=5, random_state=42).fit_predict(umap_emb)
plot_step("UMAP", umap_emb[:, :2], clusters_umap, X_umap_in, df["Class"])

dmap_df = pd.DataFrame(umap_emb, columns=[f"UMAP_{i+1}" for i in range(umap_emb.shape[1])])
dmap_df["Class"] = df["Class"]

# ───────────────────────────────────────────────────────────
# 4) top-350 variance genes → PCA
# ───────────────────────────────────────────────────────────
st.subheader("Top-350 variance genes → PCA")
K_GENES = 350
top_genes = df.drop(columns="Class").var().sort_values(ascending=False).head(K_GENES).index
df_topk = df[top_genes.tolist() + ["Class"]]
X_top = StandardScaler().fit_transform(df_topk.drop(columns="Class"))
X_pca_top = PCA(n_components=2, random_state=42).fit_transform(X_top)
clusters_top = KMeans(n_clusters=5, random_state=42).fit_predict(X_top)
plot_step("PCA (top-350)", X_pca_top, clusters_top, X_top, df_topk["Class"])

st.subheader("Top-350 variance genes → PCA (with Log Scaling)")
df_log_scaled_for_variance = pd.DataFrame(np.log1p(df.drop(columns="Class")), columns=df.drop(columns="Class").columns)
top_genes_log = df_log_scaled_for_variance.var().sort_values(ascending=False).head(K_GENES).index
df_topk_log = df[top_genes_log.tolist() + ["Class"]]
X_top_log = np.log1p(df_topk_log.drop(columns="Class"))
X_top_log_scaled = StandardScaler().fit_transform(X_top_log)
X_pca_top_log = PCA(n_components=2, random_state=42).fit_transform(X_top_log_scaled)
clusters_top_log = KMeans(n_clusters=5, random_state=42).fit_predict(X_top_log_scaled)
plot_step("PCA (top-350, log-scaled)", X_pca_top_log, clusters_top_log, X_top_log_scaled, df_topk_log["Class"])

# ───────────────────────────────────────────────────────────
# model-training utility
# ───────────────────────────────────────────────────────────
def train_model(model, data, model_name, dataset_name):
    st.markdown("---")
    st.markdown(f"## Results for Model: {model_name} — Dataset: {dataset_name}")
    st.markdown("---")

    X = data.drop(columns=["Class"])
    y = data["Class"]

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    accs, bal_accs, precs, recs, f1s, all_true, all_pred = ([] for _ in range(7))

    for tr_idx, te_idx in kf.split(X):
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        preds = model.predict(X.iloc[te_idx])
        all_true.extend(y.iloc[te_idx])
        all_pred.extend(preds)
        accs.append(accuracy_score(y.iloc[te_idx], preds))
        bal_accs.append(balanced_accuracy_score(y.iloc[te_idx], preds))
        precs.append(precision_score(y.iloc[te_idx], preds, average="weighted"))
        recs.append(recall_score(y.iloc[te_idx], preds, average="weighted"))
        f1s.append(f1_score(y.iloc[te_idx], preds, average="weighted"))

    st.subheader(f"Performance Metrics — {model_name} ({dataset_name})")
    st.write({
        "Accuracy": f"{np.mean(accs):.3f}",
        "Balanced Accuracy": f"{np.mean(bal_accs):.3f}",
        "Precision": f"{np.mean(precs):.3f}",
        "Recall": f"{np.mean(recs):.3f}",
        "F1 Score": f"{np.mean(f1s):.3f}"
    })

    st.subheader(f"Confusion Matrix — {model_name} ({dataset_name})")
    cm = confusion_matrix(all_true, all_pred, labels=np.unique(y))
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y)).plot(
        ax=ax_cm, cmap="Blues", xticks_rotation=45
    )
    st.pyplot(fig_cm)

    st.subheader(f"Classification Report — {model_name} ({dataset_name})")
    st.text(classification_report(all_true, all_pred))

    # ───────────────────────────────────────────────────────────
    # FIX: isolate SHAP into its own figure
    # ───────────────────────────────────────────────────────────
    st.subheader(f"SHAP Feature Importance — {model_name} ({dataset_name})")
    model.fit(X, y)
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        fig_shap = plt.figure()
        shap.summary_plot(
            shap_values,
            features=X,
            feature_names=X.columns,
            class_names=np.unique(y),
            plot_type="bar",
            max_display=20,
            show=False
        )
        st.pyplot(fig_shap)
        plt.clf()
    except Exception as e:
        st.warning(f"SHAP skipped: {e}")

    st.markdown(f"#### Label-Shuffling Sanity Test — {model_name} ({dataset_name})")
    y_shuf = y.sample(frac=1.0, random_state=42).reset_index(drop=True)
    shuf_accs = []
    for tr_idx, te_idx in kf.split(X):
        model_shuf = clone(model)
        model_shuf.fit(X.iloc[tr_idx], y_shuf.iloc[tr_idx])
        preds_shuf = model_shuf.predict(X.iloc[te_idx])
        shuf_accs.append(accuracy_score(y_shuf.iloc[te_idx], preds_shuf))

    st.write(
        f"Chance ≈ {1 / y.nunique():.3f} — "
        f"shuffled-label accuracy: {np.mean(shuf_accs):.3f} ± {np.std(shuf_accs):.3f}"
    )

# ───────────────────────────────────────────────────────────
# train models on each processed dataset
# ───────────────────────────────────────────────────────────
for dataset, tag in [
    (df, "Original"),
    (df_filt, "Threshold-Filtered"),
    (df_filt_log, "Threshold-Filtered (log-scaled)"),
    (dmap_df, "Filtered-UMAP"),
    (df_topk, "Selection-Top-350"),
    (df_topk_log, "Selection-Top-350 (log-scaled)")
]:
    train_model(DecisionTreeClassifier(), dataset, "Decision Tree", tag)
    train_model(RandomForestClassifier(), dataset, "Random Forest", tag)
    train_model(SGDClassifier(loss="log_loss", penalty="l2", max_iter=1000), dataset, "SGD Classifier", tag)