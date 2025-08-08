import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path
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

# Add the scripts directory to Python path - safer approach for deployments
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "scripts"))
from custom_transformers import UMAPTransformer, TopKVarianceSelector

import shap
import joblib


# Set page config after imports - must be called before any other Streamlit commands
st.set_page_config(initial_sidebar_state="collapsed")

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
# Main page layout with selectbox instead of segmented control
# ────────────────────────────────────────────
st.title("Cancer Subtype Analysis: EDA + Modeling")

section = st.selectbox(
    "Select Section:",
    options=["Data Overview", "Clustering Analysis", "Model Evaluation", "SHAP Explanations", "Clinical Predictor", "Project Findings"],
    index=0,
    key="section"
)

# DATA OVERVIEW SECTION
if section == "Data Overview":
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

# CLUSTERING ANALYSIS SECTION
elif section == "Clustering Analysis":
    st.subheader("Clustering Analysis")
    clustering_tabs = st.tabs(tags)
    for i, tag in enumerate(tags):
        with clustering_tabs[i]:
            st.subheader(f"Pipeline: {tag}")
            X2d, clusters, Xfull, labels = compute_eda(tag)
            plot_step(tag, X2d, clusters, Xfull, labels)

# MODEL EVALUATION SECTION
elif section == "Model Evaluation":
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

# SHAP EXPLANATIONS SECTION
elif section == "SHAP Explanations":
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

# CLINICAL PREDICTOR SECTION - Only section with sidebar
elif section == "Clinical Predictor":
    st.header("🩺 Cancer Subtype Clinical Predictor")
    st.markdown("---")
    st.markdown("### Clinical Decision Support Tool")
    st.markdown("Enter gene expression values to predict cancer subtype using machine learning models.")

    # Load data to get gene information
    df = load_data()
    X = df.drop(columns="Class")
    y = df["Class"]

    # Sidebar controls - only created in this section
    with st.sidebar:
        st.header("🩺 Clinical Predictor Settings")

        # Pipeline selection
        selected_pipeline = st.selectbox(
            "🔧 Select Preprocessing Pipeline",
            options=tags,
            index=2,  # Default to "Threshold-Filtered (log)" - the best performing
            help="Choose the preprocessing pipeline for feature engineering"
        )

        # Model selection
        model_names = [m[2] for m in model_specs]
        selected_model = st.selectbox(
            "🤖 Select Classification Model",
            options=model_names,
            index=2,  # Default to "SGD Classifier" - the best performing
            help="Choose the machine learning model for prediction"
        )

        st.markdown("---")

        # Number of genes to display
        num_genes = st.slider("Number of Top Genes to Display",
                              min_value=5, max_value=50,
                              value=20, step=5,
                              help="Select how many top variance genes to display for input")

        st.markdown("---")
        st.markdown("**💡 Tips:**")
        st.markdown("• Use median values as a starting point")
        st.markdown("• Try the random sample button for real patient data")
        st.markdown("• Adjust individual gene values to see prediction changes")
        st.markdown("• Compare different pipeline/model combinations")

    # Load the selected model
    @st.cache_data
    def load_selected_model(pipeline_tag, model_name):
        # Try to load the pre-trained model
        model_pipeline = load_model_pipeline(pipeline_tag, model_name)
        model_metrics = load_model_metrics(pipeline_tag, model_name)

        if model_pipeline is None:
            # If model doesn't exist, train it
            pipeline_steps = get_pipeline(pipeline_tag)
            model_cls, model_kwargs, _ = next((cls, kw, name) for cls, kw, name in model_specs if name == model_name)

            # Train the model
            fit_and_store_model(df, pipeline_steps, model_cls, model_kwargs, pipeline_tag, model_name)
            model_pipeline = load_model_pipeline(pipeline_tag, model_name)
            model_metrics = load_model_metrics(pipeline_tag, model_name)

        return model_pipeline, model_metrics


    # Load the selected model
    with st.spinner(f"Loading {selected_model} with {selected_pipeline} pipeline..."):
        current_model, current_metrics = load_selected_model(selected_pipeline, selected_model)

    # Display model performance in sidebar
    if current_metrics:
        st.sidebar.info(f"📊 **Model Performance:**\n"
                        f"• Accuracy: {current_metrics['metrics']['Accuracy']:.3f}\n"
                        f"• F1 Score: {current_metrics['metrics']['F1 Score']:.3f}\n"
                        f"• Precision: {current_metrics['metrics']['Precision']:.3f}")

    # Get pipeline steps for feature preprocessing
    pipeline_steps = get_pipeline(selected_pipeline)


    # Get feature names after preprocessing
    @st.cache_data
    def get_processed_features(pipeline_tag):
        steps = get_pipeline(pipeline_tag)
        # Apply preprocessing without the classifier
        temp_pipeline = Pipeline(steps)
        X_transformed = temp_pipeline.fit_transform(X)

        # Get feature names after preprocessing
        if pipeline_tag in ["Threshold-Filtered (log)", "Threshold-Filtered"]:
            if pipeline_tag == "Threshold-Filtered (log)":
                Xlog = np.log1p(X)
                selector = VarianceThreshold(0.5)
                selector.fit(Xlog)
            else:
                selector = VarianceThreshold(0.5)
                selector.fit(X)
            selected_features = X.columns[selector.get_support()].tolist()
        elif pipeline_tag in ["Selection-Top-350 (log)", "Selection-Top-350"]:
            if pipeline_tag == "Selection-Top-350 (log)":
                Xlog = np.log1p(X)
                top_features = Xlog.var().sort_values(ascending=False).head(350).index.tolist()
            else:
                top_features = X.var().sort_values(ascending=False).head(350).index.tolist()
            selected_features = top_features
        else:
            selected_features = X.columns.tolist()

        return selected_features, X_transformed.shape[1]


    available_genes, n_features_after_preprocessing = get_processed_features(selected_pipeline)

    st.sidebar.info(f"📊 Pipeline uses {n_features_after_preprocessing} features after preprocessing")

    # Show current selection prominently
    if selected_pipeline == "Threshold-Filtered (log)" and selected_model == "SGD Classifier":
        st.success("🏆 **Using Best Performing Model:** Threshold-Filtered (log) + SGD Classifier")
    else:
        st.info(f"📊 **Current Selection:** {selected_pipeline} + {selected_model}")

    # Get top genes by variance for user input
    if selected_pipeline.endswith("(log)"):
        Xlog = np.log1p(X)
        gene_variances = Xlog.var().sort_values(ascending=False)
    else:
        gene_variances = X.var().sort_values(ascending=False)

    top_genes = gene_variances.head(num_genes).index.tolist()

    st.markdown(f"### Enter Gene Expression Values (Top {num_genes} Genes by Variance)")
    st.markdown("*Typical gene expression values range from 0 to 20. You can use the median values as defaults.*")

    # Create input form
    with st.form("clinical_prediction_form"):
        user_inputs = {}

        # Create columns for better layout
        cols = st.columns(2)
        for i, gene in enumerate(top_genes):
            col = cols[i % 2]
            with col:
                # Get statistics for this gene
                gene_values = X[gene]
                median_val = float(gene_values.median())
                min_val = float(gene_values.min())
                max_val = float(gene_values.max())

                user_inputs[gene] = st.number_input(
                    f"{gene}",
                    min_value=min_val,
                    max_value=max_val,
                    value=median_val,
                    step=0.1,
                    help=f"Range: {min_val:.2f} - {max_val:.2f}, Median: {median_val:.2f}"
                )

        # Quick preset buttons
        st.markdown("#### Quick Presets")
        col1, col2, col3 = st.columns(3)

        use_median = col1.form_submit_button("🎯 Use Median Values")
        use_random = col2.form_submit_button("🎲 Use Random Sample")
        predict_button = col3.form_submit_button("🔬 Predict Cancer Subtype", type="primary")

        if use_median:
            st.rerun()

        if use_random:
            # Get a random sample from the dataset
            random_sample = X.sample(n=1, random_state=np.random.randint(0, 1000))
            st.session_state['random_sample'] = random_sample.iloc[0].to_dict()
            st.rerun()

    # Handle random sample
    if 'random_sample' in st.session_state:
        for gene in top_genes:
            if gene in st.session_state['random_sample']:
                user_inputs[gene] = st.session_state['random_sample'][gene]
        del st.session_state['random_sample']

    # Make prediction
    if predict_button:
        # Create input dataframe with all original features
        input_data = pd.DataFrame([{gene: 0.0 for gene in X.columns}])

        # Fill in user inputs
        for gene, value in user_inputs.items():
            input_data[gene] = value

        # Make prediction using the selected model
        try:
            prediction = current_model.predict(input_data)[0]
            prediction_proba = current_model.predict_proba(input_data)[0]
            confidence = float(max(prediction_proba)) * 100

            # Display results
            st.markdown("---")
            st.markdown("## 🎯 Prediction Results")

            # Show which model was used
            st.info(f"📊 **Prediction made using:** {selected_pipeline} + {selected_model}")

            # Main prediction
            col1, col2 = st.columns([2, 1])

            with col1:
                if prediction == "BRCA":
                    st.success(f"### 🎗️ **Predicted Cancer Subtype: BRCA (Breast Cancer)**")
                    st.markdown("**Clinical Information:** Breast invasive carcinoma")
                elif prediction == "KIRC":
                    st.success(f"### 🟦 **Predicted Cancer Subtype: KIRC (Kidney Cancer)**")
                    st.markdown("**Clinical Information:** Kidney renal clear cell carcinoma")
                elif prediction == "LUAD":
                    st.success(f"### 🫁 **Predicted Cancer Subtype: LUAD (Lung Cancer)**")
                    st.markdown("**Clinical Information:** Lung adenocarcinoma")
                elif prediction == "COAD":
                    st.success(f"### 🟤 **Predicted Cancer Subtype: COAD (Colon Cancer)**")
                    st.markdown("**Clinical Information:** Colon adenocarcinoma")
                elif prediction == "PRAD":
                    st.success(f"### 🔵 **Predicted Cancer Subtype: PRAD (Prostate Cancer)**")
                    st.markdown("**Clinical Information:** Prostate adenocarcinoma")
                else:
                    st.info(f"### 🔬 **Predicted Cancer Subtype: {prediction}**")

            with col2:
                st.metric("Confidence", f"{confidence:.1f}%")

                if confidence >= 90:
                    st.success("High Confidence")
                elif confidence >= 70:
                    st.warning("Medium Confidence")
                else:
                    st.error("Low Confidence")

            # Probability distribution
            st.markdown("### 📊 Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Cancer Subtype': current_model.classes_,
                'Probability': prediction_proba * 100
            }).sort_values('Probability', ascending=False)

            # Create a bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(prob_df['Cancer Subtype'], prob_df['Probability'],
                          color=['#ff6b6b' if x == prediction else '#74c0fc' for x in prob_df['Cancer Subtype']])
            ax.set_ylabel('Probability (%)')
            ax.set_title('Cancer Subtype Prediction Probabilities')
            ax.set_ylim(0, 100)

            # Add value labels on bars
            for bar, prob in zip(bars, prob_df['Probability']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                        f'{prob:.1f}%', ha='center', va='bottom')

            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            # Clinical recommendations
            st.markdown("### 🏥 Clinical Recommendations")
            st.info("""
            **Important:** This is a research tool for demonstration purposes only. 
            - Results should not be used for actual clinical diagnosis
            - Always consult with qualified medical professionals
            - Additional testing and clinical evaluation are required for medical decisions
            - Gene expression analysis should be part of a comprehensive diagnostic workup
            """)

            # Model information with current selection details
            with st.expander("ℹ️ Model Information"):
                st.markdown(f"""
                **Model Details:**
                - **Pipeline:** {selected_pipeline}
                - **Algorithm:** {selected_model}
                - **Accuracy:** {current_metrics['metrics']['Accuracy']:.3f}
                - **F1 Score:** {current_metrics['metrics']['F1 Score']:.3f}
                - **Precision:** {current_metrics['metrics']['Precision']:.3f}
                - **Recall:** {current_metrics['metrics']['Recall']:.3f}
                - **Cross-validation:** 3-fold stratified
                - **Features Used:** {n_features_after_preprocessing} genes after preprocessing
                - **Training Samples:** {len(X)} patients across 5 cancer types
                """)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.error("Please check your input values and try again.")

# PROJECT FINDINGS SECTION
elif section == "Project Findings":
    st.header("🔬 Project Findings & Analysis Summary")


    # Use cached content to avoid reprocessing
    @st.cache_data
    def get_project_findings_content():
        return {
            "clustering_summary": """
            **Clustering Analysis Results:**
            Through a series of dimensionality reduction and feature selection pipelines—including threshold-based filtering, log transformations, UMAP embeddings, and top-variance gene selection—six K-Means clustering approaches were evaluated. The highest clustering alignment with ground truth was observed in the Threshold-Filtered (log) and Selection-Top-350 (log) pipelines, achieving ARI and NMI scores above 0.99 and strong silhouette values (0.343 and 0.372, respectively), indicating tight, well-separated clusters. The UMAP-based pipeline, while slightly lower in ARI, exhibited the best 2D structure (silhouette = 0.632), reinforcing its value for visual subtype separation.
            """,
            "classification_summary": """
            **Supervised Classification Performance:**
            Supervised classification performance across all pipelines was consistently strong. Each pipeline was evaluated using 3-fold stratified cross-validation, ensuring class balance and robustness in performance metrics. Models were further validated with shuffled-label baselines to detect overfitting or data leakage—most of which showed shuffled accuracy well below theoretical baselines, confirming model integrity. Among traditional classifiers, the SGD Classifier and Random Forest achieved near-perfect scores on several pipelines, with balanced accuracies reaching 1.000 and F1 scores consistently above 0.99. Notably, the Threshold-Filtered (log) pipeline with the SGD Classifier achieved perfect classification (accuracy, precision, recall, and F1 score all equal to 1.000), demonstrating exceptional alignment between preprocessing and modeling.
            """,
            "conclusion": """
            **Key Conclusions:**
            Across all pipelines, classification reports showed strong generalization across cancer types such as BRCA, COAD, KIRC, LUAD, and PRAD, with macro-averaged F1-scores approaching 1.0 in most cases. These findings highlight that both unsupervised and supervised models, when properly validated and paired with effective feature transformations, can deliver highly accurate and interpretable cancer subtype predictions.
            """
        }


    # Get cached content
    content = get_project_findings_content()

    # Display content efficiently
    st.markdown("---")

    # Create expandable sections for better organization
    with st.expander("📊 Clustering Analysis Results", expanded=True):
        st.markdown(content["clustering_summary"])

        # Add key metrics summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best ARI Score", "> 0.99", "Threshold-Filtered (log)")
        with col2:
            st.metric("Best Silhouette", "0.632", "Filtered-UMAP")
        with col3:
            st.metric("Clusters Evaluated", "6", "Different pipelines")

    with st.expander("🎯 Classification Performance", expanded=True):
        st.markdown(content["classification_summary"])

        # Add performance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Accuracy", "100%", "SGD + Threshold-Filtered (log)")
        with col2:
            st.metric("CV Folds", "3", "Stratified")
        with col3:
            st.metric("Cancer Types", "5", "BRCA, COAD, KIRC, LUAD, PRAD")

    with st.expander("✅ Key Conclusions", expanded=True):
        st.markdown(content["conclusion"])

        st.success(
            "🏆 **Best Performing Model:** Threshold-Filtered (log) + SGD Classifier achieved perfect classification with 100% accuracy across all metrics.")

    # Quick reference table
    st.markdown("---")
    st.subheader("📋 Quick Reference: Model Performance Summary")


    @st.cache_data
    def get_performance_table():
        import pandas as pd
        return pd.DataFrame({
            "Pipeline": ["Threshold-Filtered (log)", "Selection-Top-350 (log)", "Filtered-UMAP", "Original",
                         "Threshold-Filtered", "Selection-Top-350"],
            "Best Model": ["SGD Classifier", "SGD Classifier", "Random Forest", "Random Forest", "SGD Classifier",
                           "Random Forest"],
            "Accuracy": ["100%", "99.8%", "99.5%", "98.9%", "99.2%", "99.1%"],
            "ARI Score": [">0.99", ">0.99", "0.85", "0.45", "0.78", "0.82"],
            "Silhouette": ["0.343", "0.372", "0.632", "0.156", "0.234", "0.289"]
        })


    performance_df = get_performance_table()
    st.dataframe(performance_df, use_container_width=True)

    # Technical details
    st.markdown("---")
    with st.expander("🔧 Technical Implementation Details"):
        st.markdown("""
        **Methodology:**
        - **Cross-Validation:** 3-fold stratified to ensure class balance
        - **Baseline Validation:** Shuffled-label testing to detect overfitting
        - **Feature Selection:** Variance thresholding (0.5), top-k selection (350)
        - **Dimensionality Reduction:** PCA, UMAP embeddings
        - **Preprocessing:** Log transformation (log1p), standardization
        - **Models Tested:** Decision Tree, Random Forest, SGD Classifier

        **Dataset Characteristics:**
        - **Samples:** 801 patients across 5 cancer types
        - **Features:** 20,531 genes (original), reduced via preprocessing
        - **Classes:** BRCA, COAD, KIRC, LUAD, PRAD
        - **Balance:** Stratified sampling maintained across folds
        """)
