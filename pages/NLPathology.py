import streamlit as st
import pandas as pd
import re
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Core imports for ClinicalBERT
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Imports for advanced analysis
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap # Import UMAP
import spacy
from spacy import displacy

st.set_page_config(layout="wide", page_title="NLP Pipeline for Pathology Reports")

# --- MODEL LOADING ---

# Global variables for ClinicalBERT model and tokenizer
tokenizer = None
base_clinicalbert_model = None

# Sample use for debugging
# CLINICALBERT_BASE_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

# Encoder LLM. Finetuned ModernBERT thing.
@st.cache_resource
def load_cancer_model():
    try:
        model_path = "models/answerdotai"
        cancer_tokenizer = AutoTokenizer.from_pretrained(model_path)
        cancer_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        cancer_model.eval()
        return cancer_tokenizer, cancer_model, True
    except Exception as e:
        st.error(f"Error loading cancer model: {e}")
        return None, None, False

cancer_tokenizer, cancer_model, cancer_model_loaded = load_cancer_model()

@st.cache_resource
def load_clinicalbert_base_model_cached():
    """Caches and loads the ClinicalBERT base model and tokenizer."""
    try:
        loaded_tokenizer = AutoTokenizer.from_pretrained(CLINICALBERT_BASE_MODEL_NAME)
        loaded_model = AutoModelForSequenceClassification.from_pretrained(CLINICALBERT_BASE_MODEL_NAME)
        loaded_model.eval()
        st.success(f"Standard ClinicalBERT base model '{CLINICALBERT_BASE_MODEL_NAME}' and tokenizer loaded successfully!")
        return loaded_tokenizer, loaded_model
    except Exception as e:
        st.error(f"Failed to load standard ClinicalBERT base model. Error: {e}")
        st.info("Falling back to dummy inference for demonstration.")
        return None, None

@st.cache_resource
def load_spacy_model_cached():
    """Caches and loads a spaCy model for NER demonstration."""
    try:
        # For a real clinical task, you'd use a specialized model like 'en_core_sci_sm'
        # For this demo, we use the general-purpose 'en_core_web_sm'
        nlp = spacy.load("en_core_web_sm")
        st.success("SpaCy model ('en_core_web_sm') for NER demonstration loaded successfully!")
        return nlp
    except OSError:
        st.error("SpaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm' in your terminal.")
        return None

# Attempt to load models once at the start
tokenizer, base_clinicalbert_model = load_clinicalbert_base_model_cached()
models_loaded_successfully = (tokenizer is not None) and (base_clinicalbert_model is not None)
nlp_spacy = load_spacy_model_cached()


# --- DATA LOADING & PREPARATION ---

@st.cache_data
def load_and_unify_data():
    """Loads, unifies, and caches all TCGA data from CSV files."""
    try:
        dtype_spec = {'patient_filename': str, 'text': str, 'patient_id': str, 'cancer_type': str,
                      'ajcc_pathologic_m': str, 'ajcc_pathologic_n': str, 'ajcc_pathologic_t': str,
                      'project_id': str, 'case_id': str, 'case_submitter_id': str}
        
        df_reports = pd.read_csv("Data/TCGA_Reports.csv", dtype=dtype_spec)
        df_cancer_type = pd.read_csv("Data/TCGA_patient_to_cancer_type.csv", dtype=dtype_spec)
        df_m01 = pd.read_csv("Data/TCGA_M01_patients.csv", dtype=dtype_spec)
        df_n03 = pd.read_csv("Data/TCGA_N03_patients.csv", dtype=dtype_spec)
        df_t14 = pd.read_csv("Data/TCGA_T14_patients.csv", dtype=dtype_spec)

    except FileNotFoundError as e:
        st.error(f"A dataset file was not found in the 'Data/' directory. Please check the file path. Error: {e}")
        return pd.DataFrame(), {}, 0

    df_reports['patient_id'] = df_reports['patient_filename'].apply(lambda x: x.split('.')[0])
    original_reports_rows = df_reports.shape[0]

    df_master = pd.merge(df_reports, df_cancer_type, on='patient_id', how='left')
    df_master = pd.merge(df_master, df_m01.rename(columns={'case_submitter_id': 'patient_id'}), on='patient_id', how='left')
    df_master = pd.merge(df_master, df_n03.rename(columns={'case_submitter_id': 'patient_id'}), on='patient_id', how='left')
    df_master = pd.merge(df_master, df_t14.rename(columns={'case_submitter_id': 'patient_id'}), on='patient_id', how='left')

    for col in ['patient_id', 'text', 'cancer_type', 'ajcc_pathologic_m', 'ajcc_pathologic_n', 'ajcc_pathologic_t']:
        if col in df_master.columns:
            df_master[col] = df_master[col].astype(str).fillna('Not Available')

    datasets_raw = {
        "TCGA_Reports.csv": df_reports, "TCGA_patient_to_cancer_type.csv": df_cancer_type,
        "TCGA_M01_patients.csv": df_m01, "TCGA_N03_patients.csv": df_n03, "TCGA_T14_patients.csv": df_t14,
    }
    return df_master, datasets_raw, original_reports_rows

def preprocess_text_for_eda(text):
    """Basic text preprocessing: lowercase, remove special chars, normalize whitespace."""
    if pd.isna(text): return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- NLP INFERENCE & ANALYSIS LOGIC ---

"""
def _dummy_inference_fallback(report_text):
    common_cancer_types = ["BRCA", "LUAD", "COAD", "KIRC", "GBM"]
    common_t_stages = ["T1", "T1a", "T1b", "T2", "T2a", "T2b", "T3", "T4", "TX"]
    common_n_stages = ["N0", "N1", "N1a", "N1b", "N2", "N3", "NX"]
    common_m_stages = ["M0", "M1", "M1a", "M1b", "M1c", "MX"]

    cancer_type = {"value": random.choice(common_cancer_types), "confidence": round(random.uniform(0.85, 0.99), 2), "status": "Assessed"}
    t_stage = {"value": random.choice(common_t_stages), "confidence": round(random.uniform(0.85, 0.99), 2), "status": "Assessed"}
    n_stage = {"value": random.choice(common_n_stages), "confidence": round(random.uniform(0.85, 0.99), 2), "status": "Assessed"}
    m_stage = {"value": random.choice(common_m_stages), "confidence": round(random.uniform(0.85, 0.99), 2), "status": "Assessed"}

    # Initialize features with default values
    tumor_size = {"value": "Not Specified", "unit": None, "confidence": None, "status": "Not Specified"}
    histologic_type = {"value": "Not Specified", "confidence": None, "status": "Not Specified"}
    margins_status = {"value": "Not Specified", "confidence": None, "status": "Not Specified"}
    angiolymphatic_invasion = {"value": "Not Specified", "confidence": None, "status": "Not Specified"}
    positive_lymph_nodes_count = {"value": "Not Specified", "total_examined": None, "confidence": None, "status": "Not Specified"}

    if pd.isna(report_text) or not report_text.strip():
        # Return a structure indicating insufficient information
        return {
            "cancer_type": {"value": "Insufficient Info", "confidence": None, "status": "Insufficient Info"},
            "tnm_staging": {k: {"value": "Insufficient Info", "confidence": None, "status": "Insufficient Info"} for k in ["t_stage", "n_stage", "m_stage"]},
            "key_features": {k: {"value": "Not Specified", "confidence": None, "status": "Not Specified"} for k in ["tumor_size_cm", "histologic_type", "margins_status", "angiolymphatic_invasion", "venous_invasion", "perineural_invasion", "positive_lymph_nodes_count"]},
            "report_status": "Insufficient Information in Text"
        }

    # Simplified regex extraction logic
    size_match = re.search(r'tumor size.*?(\d+\.?\d*)\s*(cm|mm)', report_text, re.IGNORECASE)
    if size_match:
        value, unit = float(size_match.group(1)), size_match.group(2).lower()
        if unit == 'mm': value /= 10
        tumor_size = {"value": str(value), "unit": 'cm', "confidence": round(random.uniform(0.9, 0.99), 2), "status": "Extracted"}

    return {
        "cancer_type": cancer_type, "tnm_staging": {"t_stage": t_stage, "n_stage": n_stage, "m_stage": m_stage},
        "key_features": {"tumor_size_cm": tumor_size, "histologic_type": histologic_type, "margins_status": margins_status,
                         "angiolymphatic_invasion": angiolymphatic_invasion, "venous_invasion": {"value": "Not Specified", "confidence": None, "status": "Not Specified"},
                         "perineural_invasion": {"value": "Not Specified", "confidence": None, "status": "Not Specified"}, "positive_lymph_nodes_count": positive_lymph_nodes_count},
        "report_status": "Processed Successfully"
    }
"""

def run_clinicalbert_inference(report_text):
    if not cancer_model_loaded:
        st.error("Cancer classification model failed to load. Pls fix.")
        return None

    try:
        def clean_text(text):
            if not isinstance(text, str):
                return ""
            text = text.lower()
            text = re.sub(r'[^a-z0-9\s\.]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        squeaky_clean_text = clean_text(report_text)

        inputs = cancer_tokenizer(
            squeaky_clean_text, 
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        device = torch.device("cuda" if torch.cuda_is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        cancer_model.to(device)

        with torch.no_grad():
            outputs = cancer_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        # TODO: Make function that gets cancer subtypes from TCGA_patient_to_cancer_type.csv, then plug them into this bad boy --> class_to_cancer = {}

        # TODO: Just have M, N, and T models output 1 until I get trained models

def plot_top_ngrams(text_series, n=2, top_k=20):
    """Calculates and plots the top K n-grams from a text series."""
    vec = CountVectorizer(ngram_range=(n, n), stop_words='english').fit(text_series)
    bag_of_words = vec.transform(text_series)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    top_df = pd.DataFrame(words_freq[:top_k], columns=['N-gram', 'Frequency'])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Frequency', y='N-gram', data=top_df, ax=ax)
    ax.set_title(f'Top {top_k} {"Bi-grams" if n==2 else "Tri-grams"}')
    return fig

# --- STREAMLIT PAGE-RENDERING FUNCTIONS ---

def explore_datasets():
    """Renders the EDA page with dataset overviews and advanced text analysis."""
    st.header("1. Exploratory Data Analysis of Datasets")

    df_master, datasets_raw, original_reports_rows = load_and_unify_data()

    if df_master.empty and not datasets_raw: # Check if data loading failed
        st.error("Data loading failed. Cannot perform EDA.")
        return

    for name, df in datasets_raw.items():
        with st.expander(f"Explore Raw Dataset: {name}"):
            st.write(f"**Shape:** `{df.shape}`")
            st.write("**First 5 rows:**")
            st.dataframe(df.head())
            
            # Create a string buffer to capture df.info() output
            from io import StringIO
            buffer = StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.write("**Column Information:**")
            st.text(s)

            st.write("**Missing Values:**")
            st.dataframe(df.isnull().sum().to_frame('missing_count'))
            st.write("**Descriptive Statistics:**")
            st.dataframe(df.describe(include='all'))

            if name == "TCGA_Reports.csv":
                df_temp_reports = df.copy()
                df_temp_reports['patient_id_extracted'] = df_temp_reports['patient_filename'].apply(lambda x: x.split('.')[0])
                st.write("**Extracted Patient IDs (Sample):**")
                st.dataframe(df_temp_reports[['patient_filename', 'patient_id_extracted']].head())
                st.write(f"Number of unique patient_ids in `{name}`: `{df_temp_reports['patient_id_extracted'].nunique()}`")
                st.write(f"Number of duplicate patient_ids in `{name}`: `{df_temp_reports['patient_id_extracted'].duplicated().sum()}`")

            if 'patient_id' in df.columns and name != "TCGA_Reports.csv":
                 st.write(f"Number of unique patient_ids in `{name}`: `{df['patient_id'].nunique()}`")
                 st.write(f"Number of duplicate patient_ids in `{name}`: `{df['patient_id'].duplicated().sum()}`")
            
            if 'case_submitter_id' in df.columns:
                 st.write(f"Number of unique patient_ids (case_submitter_id) in `{name}`: `{df['case_submitter_id'].nunique()}`")
                 st.write(f"Number of duplicate patient_ids (case_submitter_id) in `{name}`: `{df['case_submitter_id'].duplicated().sum()}`")

            if 'cancer_type' in df.columns:
                st.write("**Unique Cancer Types and their counts:**")
                st.dataframe(df['cancer_type'].value_counts())

            if 'ajcc_pathologic_m' in df.columns:
                st.write("**Unique AJCC Pathologic M values and their counts:**")
                st.dataframe(df['ajcc_pathologic_m'].value_counts())

            if 'ajcc_pathologic_n' in df.columns:
                st.write("**Unique AJCC Pathologic N values and their counts:**")
                st.dataframe(df['ajcc_pathologic_n'].value_counts())

            if 'ajcc_pathologic_t' in df.columns:
                st.write("**Unique AJCC Pathologic T values and their counts:**")
                st.dataframe(df['ajcc_pathologic_t'].value_counts())

    st.subheader("1.1. Data Unification") 
    st.write("All individual datasets have been unified into a single master DataFrame for comprehensive analysis.")
    st.write("Unified Master Dataset (first 5 rows with relevant columns):")
    st.dataframe(df_master[['patient_id', 'text', 'cancer_type', 'ajcc_pathologic_m', 'ajcc_pathologic_n', 'ajcc_pathologic_t']].head())
    st.write("Missing values after unification:")
    st.dataframe(df_master[['cancer_type', 'ajcc_pathologic_m', 'ajcc_pathologic_n', 'ajcc_pathologic_t']].isnull().sum())
    st.write(f"**Size of unified master dataset:** `{df_master.shape[0]}` rows. (Original `TCGA_Reports.csv` had `{original_reports_rows}` rows).")

    st.subheader("1.2. Dimensionality Reduction for Text Visualization")
    st.write("Visualize the high-dimensional text data in 2D to see if reports cluster by cancer type. We use TF-IDF for vectorization.")
    
    df_labeled_eda = df_master[df_master['cancer_type'].notna()].copy()
    df_labeled_eda['cleaned_text'] = df_labeled_eda['text'].apply(preprocess_text_for_eda)
    df_labeled_eda = df_labeled_eda[df_labeled_eda['cleaned_text'].str.strip() != '']
    if df_labeled_eda.empty:
        st.warning("No labeled data available for text analysis after preprocessing.")
        return

    # --- PCA Section ---
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    X_tfidf = tfidf.fit_transform(df_labeled_eda['cleaned_text'])
    
    # PCA Plot
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_tfidf.toarray())
    df_labeled_eda['PCA1'], df_labeled_eda['PCA2'] = X_pca[:, 0], X_pca[:, 1]
    
    # t-SNE Plot
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_tfidf.toarray())
    df_labeled_eda['TSNE1'], df_labeled_eda['TSNE2'] = X_tsne[:, 0], X_tsne[:, 1]

    # UMAP Plot
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_tfidf)
    df_labeled_eda['UMAP1'], df_labeled_eda['UMAP2'] = X_umap[:, 0], X_umap[:, 1]

    tab1, tab2, tab3 = st.tabs(["PCA Plot", "t-SNE Plot", "UMAP Plot"])
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(data=df_labeled_eda, x='PCA1', y='PCA2', hue='cancer_type', palette='tab10', s=50, alpha=0.7, ax=ax)
        ax.set_title("PCA of Pathology Reports")
        st.pyplot(fig)
    with tab2:
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(data=df_labeled_eda, x='TSNE1', y='TSNE2', hue='cancer_type', palette='tab10', s=50, alpha=0.7, ax=ax)
        ax.set_title("t-SNE of Pathology Reports")
        st.pyplot(fig)
    with tab3:
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(data=df_labeled_eda, x='UMAP1', y='UMAP2', hue='cancer_type', palette='tab10', s=50, alpha=0.7, ax=ax)
        ax.set_title("UMAP of Pathology Reports")
        st.pyplot(fig)

def run_ngram_analysis_page():
    """Renders the N-gram Frequency Analysis page."""
    st.header("N-gram Frequency Analysis")
    st.write("""
    Discover the most common two-word (bi-gram) and three-word (tri-gram) phrases to identify key clinical terminology. 
    You can filter the reports by Cancer Type or by TNM staging to compare linguistic patterns across different cohorts.
    """)

    df_master, _, _ = load_and_unify_data()
    if df_master.empty: return

    df_master['cleaned_text'] = df_master['text'].apply(preprocess_text_for_eda)
    df_master = df_master[df_master['cleaned_text'].str.strip() != '']

    # --- Filter selection ---
    filter_option = st.selectbox(
        "Choose a category to filter by:",
        ("Cancer Type", "T Stage", "N Stage", "M Stage")
    )

    df_filtered = df_master.copy() # Start with all data
    column_name = ''
    
    if filter_option == "Cancer Type":
        column_name = 'cancer_type'
    elif filter_option == "T Stage":
        column_name = 'ajcc_pathologic_t'
    elif filter_option == "N Stage":
        column_name = 'ajcc_pathologic_n'
    elif filter_option == "M Stage":
        column_name = 'ajcc_pathologic_m'

    if column_name:
        # Get unique, non-null, non-empty values for the selected category
        valid_values = df_master[column_name][df_master[column_name].notna() & (df_master[column_name] != 'Not Available')].unique()
        values = ["All"] + sorted(valid_values.tolist())
        selected_value = st.selectbox(f"Filter by {filter_option}:", values)
        
        if selected_value != "All":
            df_filtered = df_master[df_master[column_name] == selected_value]

    st.write(f"Analyzing `{len(df_filtered)}` reports.")

    if len(df_filtered) < 1:
        st.warning("No reports match the selected filter. Cannot generate N-gram plots.")
        return
        
    # --- Plotting ---
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_top_ngrams(df_filtered['cleaned_text'], n=2))
    with col2:
        st.pyplot(plot_top_ngrams(df_filtered['cleaned_text'], n=3))

def run_ner_analysis_page():
    """Renders the NER demonstration page."""
    st.header("Named Entity Recognition (NER) Demonstration")
    st.write("""
    This section demonstrates how a Named Entity Recognition (NER) model can automatically identify and classify key clinical entities in text. 
    **Note:** We are using a general-purpose spaCy model (`en_core_web_sm`). For real clinical applications, a model fine-tuned on biomedical data (like `en_core_sci_sm` or a custom-trained model) would be required for accurate results.
    """)
    if not nlp_spacy:
        st.error("NER analysis cannot proceed because the spaCy model is not loaded.")
        return

    df_master, _, _ = load_and_unify_data()
    if df_master.empty: return

    sample_options = {f"Patient {df_master['patient_id'].iloc[i]} ({df_master['cancer_type'].iloc[i]})": df_master['text'].iloc[i] for i in range(10)}
    selected_report_key = st.selectbox("Choose a sample report to analyze:", sample_options.keys())
    
    report_text = sample_options[selected_report_key]
    doc = nlp_spacy(report_text)
    
    # Generate HTML for displaCy
    html = displacy.render(doc, style="ent", jupyter=False)
    
    st.markdown("### Highlighted Entities")
    st.markdown(html, unsafe_allow_html=True)
    
    st.markdown("### Extracted Entities Table")
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    if entities:
        df_entities = pd.DataFrame(entities, columns=["Entity Text", "Entity Type"])
        st.dataframe(df_entities)
    else:
        st.info("No entities were identified by the general-purpose model in this report.")

def run_nlp_pipeline():
    """Renders the main NLP pipeline demonstration page."""
    st.header("2. NLP Pipeline (ClinicalBERT Integrated)")
    df_master, _, _ = load_and_unify_data()
    if df_master.empty: return
    st.subheader("2.1. Text Preprocessing")
    df_master['cleaned_text'] = df_master['text'].apply(preprocess_text_for_eda)
    st.text_area("Original", df_master['text'].iloc[0][:500] + "...", height=150)
    st.text_area("Cleaned", df_master['cleaned_text'].iloc[0][:500] + "...", height=150)
    st.subheader("2.2. ClinicalBERT Model Inference")
    
    sample_report_text = df_master['text'].iloc[0]
    sample_patient_id = df_master['patient_id'].iloc[0]
    st.write(f"**Processing Sample Patient ID:** `{sample_patient_id}`")
    generated_report = run_clinicalbert_inference(sample_report_text)
    st.json(generated_report)

def interactive_report_processing():
    """Renders the interactive report processing page."""
    st.header("3. Interactive Report Processing")
    st.write("Upload a pathology report file (.txt) or paste text directly to generate a structured summary report.")

    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    report_text_input = st.text_area("Or paste pathology report text here:", height=300)

    processed_text = ""
    if uploaded_file is not None:
        try:
            processed_text = uploaded_file.read().decode("utf-8")
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return
    elif report_text_input:
        processed_text = report_text_input

    if st.button("Generate Structured Report"):
        if processed_text:
            with st.spinner("Processing report..."):
                st.subheader("Generated Report:")
                patient_id_for_interactive = "USER-INPUT-001"
                generated_report = run_clinicalbert_inference(processed_text)
                generated_report["patient_id"] = patient_id_for_interactive

                st.markdown("### Patient Summary")
                col1, col2 = st.columns(2)
                col1.metric("Patient ID", generated_report['patient_id'])
                
                cancer_type_val = generated_report['cancer_type']['value']
                cancer_type_conf = generated_report['cancer_type'].get('confidence')
                cancer_type_status = generated_report['cancer_type']['status']
                
                if cancer_type_status != "Assessed":
                    col2.metric("Cancer Type", cancer_type_val, delta=cancer_type_status, delta_color="off")
                else:
                    col2.metric("Cancer Type", cancer_type_val, f"Confidence: {cancer_type_conf:.2f}")

                st.markdown("### TNM Staging")
                tnm_data = []
                for stage_key, stage_info in generated_report['tnm_staging'].items():
                    value = stage_info['value']
                    confidence = stage_info.get('confidence')
                    status = stage_info['status']
                    display_confidence = f"{confidence:.2f}" if confidence is not None else "N/A"
                    
                    tnm_data.append({
                        "Stage": stage_key.replace('_', ' ').title(),
                        "Value": value,
                        "Confidence": display_confidence,
                        "Status": status
                    })
                st.dataframe(pd.DataFrame(tnm_data))

                st.markdown("### Key Features Extracted")
                features_data = []
                for feature_key, feature_info in generated_report['key_features'].items():
                    value = feature_info['value']
                    confidence = feature_info.get('confidence')
                    status = feature_info['status']
                    unit = feature_info.get('unit', '')
                    total_examined = feature_info.get('total_examined', '')

                    display_value = str(value)
                    if unit:
                        display_value += f" {unit}"
                    if total_examined:
                        display_value += f" (of {total_examined})"
                    
                    display_confidence = f"{confidence:.2f}" if confidence is not None else "N/A"

                    features_data.append({
                        "Feature": feature_key.replace('_', ' ').title(),
                        "Value": display_value,
                        "Confidence": display_confidence,
                        "Status": status
                    })
                st.dataframe(pd.DataFrame(features_data))

                st.success(f"Overall Report Status: {generated_report['report_status']}")

                with st.expander("Show Raw JSON Output"):
                    st.json(generated_report)
        else:
            st.warning("Please upload a file or paste text to generate a report.")

def main():
    """Main function to set up the Streamlit application's navigation and content."""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to", 
        ["Dataset Exploration", "N-gram Frequency Analysis", "Named Entity Recognition (NER)", "NLP Pipeline", "Interactive Report Processing"]
    )

    if page == "Dataset Exploration":
        explore_datasets()
    elif page == "N-gram Frequency Analysis":
        run_ngram_analysis_page()
    elif page == "Named Entity Recognition (NER)":
        run_ner_analysis_page()
    elif page == "NLP Pipeline":
        run_nlp_pipeline()
    elif page == "Interactive Report Processing":
        interactive_report_processing()

if __name__ == "__main__":
    main()