import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import streamlit as st
import seaborn as sns
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE, KMeansSMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Breast Cancer Prognosis",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items=None,
)

# st.markdown(
#     """
#     <style>
#       /* In centered mode Streamlit still uses .block-container for your content */
#       .block-container {
#         max-width: 1200px !important;  /* bump up from its ~700px default */
#         padding-left: 1rem !important;
#         padding-right: 1rem !important;
#       }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )



# Load the dataset
file_path = 'Data/Breast_Cancer.csv'
data = pd.read_csv(file_path)

# # Navigation bar in the sidebar
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Modeling Results"])

############################## Training Model & Application #############################################
def trainModel():
    le = LabelEncoder()
    pdata = data.copy()
    encoders = {}

    for col in pdata.columns:
        if pdata[col].dtype == 'object':
            pdata[col] = le.fit_transform(pdata[col].astype(str))
            encoders[col] = le   # <-- store le, not enc
            le = LabelEncoder()  # re-instantiate for the next column

    # Copy of encoded dataset to use for survival prediction
    pdataS = pdata.copy()

    # 1. Create the new target column 'Survival Class' with 6 categories
    def categorize_survival(months):
        if months < 12:
            return 0
        elif 12 <= months < 24:
            return 1
        elif 24 <= months < 36:
            return 2
        elif 36 <= months < 48:
            return 3
        elif 48 <= months < 60:
            return 4
        elif 60 <= months < 72:
            return 5
        elif 72 <= months < 84:
            return 6
        elif 84 <= months < 96:
            return 7
        else:
            return 8

    pdataS['Survival Class'] = pdataS['Survival Months'].apply(categorize_survival)

    # Remove "Status" if present so it's not used as a feature
    if 'Status' in pdataS.columns:
        pdataS = pdataS.drop(columns=['Status'])

    # Define features (X1) and target (y1) for the classification task
    X1 = pdataS.drop(['Survival Months', 'Survival Class'], axis=1)
    y1 = pdataS['Survival Class']

    # Get unique class names for display purposes
    class_names = ['<1 Year', '1-2 Years', '2-3 Years', '3-4 Years', '4-5 Years', '5-6 Years', '7-8 Years', '8-9 Years', '9+ Years']

    original_counts = Counter(y1) 
    desired_sampling_strategy = {
        0: 600,
        1: 600,
        2: 600,
        3: 600,
        4: 1200,
        5: 1300,
        6: 1300,
        7: 1300,
        8: 1300
    }
    RandomSample_survival = RandomOverSampler(random_state=42, sampling_strategy=desired_sampling_strategy)
    X1_resampled, y1_resampled = RandomSample_survival.fit_resample(X1, y1)

    # Splitting data into training and testing sets
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1_resampled, y1_resampled, test_size=0.2, random_state=42) #resampled

    modelRFC_sm = RandomForestClassifier(random_state=42)
    modelRFC_sm.fit(X1_train, y1_train)

    y1_pred_sm = modelRFC_sm.predict(X1_test)

    # Use ENN to clean noisy points from resampled data
    enn = EditedNearestNeighbours(
        sampling_strategy='all',
        n_neighbors=2,
        kind_sel='all'
    )

    X1_cleaned, y1_cleaned = enn.fit_resample(X1_resampled, y1_resampled)

    # Split cleaned data into training and testing sets
    X1_train_cln, X1_test_cln, y1_train_cln, y1_test_cln = train_test_split(
        X1_cleaned, 
        y1_cleaned, 
        test_size=0.2, 
        stratify=y1_cleaned, 
        random_state=42
    )

    # Train a Random Forest model on cleaned data
    modelRFC_cln = RandomForestClassifier(random_state=42)
    modelRFC_cln.fit(X1_train_cln, y1_train_cln)
    y1_pred_cln = modelRFC_cln.predict(X1_test_cln)

    # Print metrics for cleaned data
    acc_cln = accuracy_score(y1_test_cln, y1_pred_cln)

    st.session_state["model_name"] = modelRFC_cln.__class__.__name__
    st.session_state["model_accuracy"] = acc_cln


    # st.write("### Random Forest with OverSampling & ENN Cleaning")
    # st.write(f"Accuracy: {acc_cln:.3f}")

    ### Store session state for interactive prediction
    st.session_state["clf"] = modelRFC_cln
    st.session_state["feature_cols"] = list(X1.columns)
    st.session_state["encoders"] = encoders
    st.session_state["class_names"] = class_names
    st.session_state["raw_data"] = data
    st.session_state["encoded_data"] = pdataS
    # keep cleaned test set around for sampling
    st.session_state["X_test"] = X1_test_cln.reset_index(drop=True)
    st.session_state["y_test"] = y1_test_cln.reset_index(drop=True)
    # also save the training split
    st.session_state["X_train"] = X1_train_cln.reset_index(drop=True)
    st.session_state["y_train"] = y1_train_cln.reset_index(drop=True)


    ### Interactive Risk Predictor
    st.subheader("Interactive: Predicted Risk Level")

    clf = st.session_state["clf"]
    feature_cols = st.session_state["feature_cols"]
    encs = st.session_state["encoders"]
    raw = st.session_state["raw_data"]
    encoded = st.session_state["encoded_data"]

    # # --- clear any previous patient input values so the form starts blank ---
    # for col in feature_cols:
    #     if col in st.session_state:
    #         del st.session_state[col]


        # --- only clear previous patient inputs once so form starts blank initially ---
    if "inputs_cleared" not in st.session_state:
        for col in st.session_state["feature_cols"]:
            st.session_state.pop(col, None)
            st.session_state.pop(f"{col}_raw", None)
        st.session_state["inputs_cleared"] = True

    # Callback to load random sample into session_state before widgets instantiate
    def load_random_sample():
        idx = np.random.randint(len(st.session_state["X_test"]))
        sample = st.session_state["X_test"].iloc[idx]
        feature_cols = st.session_state["feature_cols"]
        encs = st.session_state["encoders"]
        for col in feature_cols:
            if col in encs:
                st.session_state[col] = encs[col].inverse_transform([int(sample[col])])[0]
            else:
                st.session_state[col] = float(sample[col])
                # ensure the text_input shows it too
                st.session_state[f"{col}_raw"] = str(sample[col])
        st.session_state["loaded_idx"] = idx

    # Layout: left = input form, right = sample & prediction + result
    left_col, right_col = st.columns([5, 3.5])  # right slightly wider

    # RIGHT: Load sample & prediction
    with right_col:
        st.write("### ▶️ Prediction")
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            st.button("🔃 Load Random Sample", on_click=load_random_sample)
        with btn_col2:
            do_predict = st.button("Predict Risk Level")

        if "loaded_idx" in st.session_state:
            st.success(f"🔍 Loaded random patient at index **{st.session_state['loaded_idx']}**. Now click ▶️ Predict Risk Level.")

        # Prediction logic: collect inputs, validate, then predict
        feature_cols = st.session_state["feature_cols"]
        encs = st.session_state["encoders"]
        user_inputs = {}
        missing = []

        for col in feature_cols:
            if col in encs:
                val = st.session_state.get(col, "")
                if val == "" or val is None:
                    missing.append(col)
                else:
                    try:
                        user_inputs[col] = encs[col].transform([val])[0]
                    except Exception:
                        missing.append(col)
            else:
                val = st.session_state.get(col, None)
                if val is None:
                    missing.append(col)
                else:
                    user_inputs[col] = val

        if do_predict:
            if missing:
                st.warning(f"Cannot predict: missing inputs for {', '.join(missing)}.")
            else:
                X_new = pd.DataFrame([user_inputs])[feature_cols]
                clf = st.session_state["clf"]

                pred = clf.predict(X_new)[0]
                risk = "HIGH" if pred <= 2 else "MEDIUM" if pred <= 5 else "LOW"

                class_to_midpoint = {
                    0: 0.5, 1: 1.5, 2: 2.5, 3: 3.5,
                    4: 4.5, 5: 5.5, 6: 6.5, 7: 7.5, 8: 9.0,
                }
                class_to_range = {
                    0: "<1 year", 1: "1–2 years", 2: "2–3 years", 3: "3–4 years",
                    4: "4–5 years", 5: "5–6 years", 6: "6–7 years", 7: "7–8 years", 8: "8+ years",
                }
                est_years = class_to_midpoint.get(pred, None)
                range_str = class_to_range.get(pred, "Unknown")

                confidence_str = "N/A"
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(X_new)[0]
                    confidence_str = f"{probs[pred]*100:.1f}%"

                st.write("### Prediction Result")
                model_name = st.session_state.get("model_name", clf.__class__.__name__)
                model_acc = st.session_state.get("model_accuracy", None)
                info_line = f"**Model:** {model_name}"
                if model_acc is not None:
                    info_line += f"  •  **Test Accuracy:** {model_acc:.3f}"
                info_line += f"  •  **Confidence:** {confidence_str}"
                st.markdown(info_line)

                if risk == "HIGH":
                    st.error("🟥 High Risk (< 3 years)")
                elif risk == "MEDIUM":
                    st.warning("🟧 Medium Risk (3–6 years)")
                else:
                    st.success("🟩 Low Risk (> 6 years)")

                if est_years is not None:
                    st.markdown(f"**Estimated survival:** ~{est_years:.1f} years ({range_str})")
                else:
                    st.markdown(f"**Estimated survival range:** {range_str}")

                st.caption(f"Predicted class index: {pred}")

    # LEFT: Input form (starts blank unless a sample was loaded)
    with left_col:
        st.write("Enter patient characteristics:")
        feature_cols = st.session_state["feature_cols"]
        encs = st.session_state["encoders"]
        raw = st.session_state["raw_data"]
        encoded = st.session_state["encoded_data"]

        col1, col2 = st.columns(2)
        for i, col in enumerate(feature_cols):
            target = col1 if i % 2 == 0 else col2
            with target:
                if col in encs:
                    opts = [""] + list(encs[col].classes_)  # blank leading option
                    current = st.session_state.get(col, "")
                    index = opts.index(current) if current in opts else 0
                    st.selectbox(col, opts, index=index, key=col)
                else:
                    raw_key = f"{col}_raw"
                    existing = ""
                    if raw_key in st.session_state:
                        existing = st.session_state[raw_key]
                    inp = st.text_input(col, value=existing, key=raw_key, placeholder="e.g., 3.4")
                    try:
                        if inp.strip() == "":
                            st.session_state.pop(col, None)
                        else:
                            # sync numeric value for prediction
                            st.session_state[col] = float(inp)
                    except ValueError:
                        st.warning(f"Invalid number for {col}; please enter a numeric value.")

    st.markdown("---")
    st.subheader("📊 Dataset Sizes")

    train_size = len(st.session_state["X_train"])
    test_size  = len(st.session_state["X_test"])

    st.write(f"**Training set size:** {train_size} samples")
    st.write(f"**Testing set size:** {test_size} samples")

    ### Display the full testing‐set pool
    st.markdown("---")
    st.subheader("🗂️ Testing Set Pool")

    if st.checkbox("Show testing set pool"):
        # 1) grab encoded feature matrix + true labels
        df_pool = st.session_state["X_test"].copy()
        df_pool["Actual Survival Class"] = st.session_state["y_test"].values
        # map numeric class → human name
        class_names = st.session_state["class_names"]
        df_pool["Class Name"] = df_pool["Actual Survival Class"].map(lambda i: class_names[i])

        # 2) decode each encoded categorical column
        for col, le in st.session_state["encoders"].items():
            df_pool[col] = le.inverse_transform(df_pool[col].astype(int))

        # 3) show the decoded pool
        st.dataframe(df_pool)    

    ### 
############################################# Data Exploration #############################################

def dataExploration():
    st.subheader("Data Exploration")
    st.markdown("---")

    # Preview of the raw dataset
    st.subheader("Preview of Dataset")
    st.dataframe(data.head())

    # Shape of the dataset
    st.subheader("Shape of Dataset")
    st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

    # Summary statistics and null/unique counts
    def create_summary(df):
        return pd.DataFrame({
            "Null Count": df.isna().sum(),
            "Unique": df.nunique(),
            "Dtype": df.dtypes.astype(str)
        })
    summary_df = create_summary(data)
    st.subheader("Summary of Dataset")
    st.dataframe(summary_df, use_container_width=True)

    # Statistical description
    st.subheader("Dataset Statistical Information")
    st.dataframe(data.describe(), use_container_width=True)

    # Distribution plots
    st.subheader("Data Distribution by Column")
    def plot_distribution(df, col):
        fig, ax = plt.subplots()
        sns.histplot(data=df, x=col, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)
    selected_col = st.selectbox("Select a column to plot", data.columns)
    plot_distribution(data, selected_col)

    # Survival months by selected category
    st.subheader("Survival Months by Category")
    def plot_survival_by_category(df, cat):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(data=df, x=cat, hue=df['Survival Months'] // 12, palette='muted', ax=ax)
        ax.set_title(f"{cat} by Survival Years")
        st.pyplot(fig)
    cat_cols = data.columns.drop('Survival Months')
    sel_cat = st.selectbox("Select a category to plot Survival Months", cat_cols)
    plot_survival_by_category(data, sel_cat)

    # Status by selected category
    st.subheader("Status by Category")
    def plot_status_by_category(df, cat):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(data=df, x=cat, hue='Status', palette='muted', ax=ax)
        ax.set_title(f"{cat} by Status")
        st.pyplot(fig)
    status_cols = data.columns.drop('Status')
    sel_status_cat = st.selectbox("Select a category to plot Status", status_cols)
    plot_status_by_category(data, sel_status_cat)

    # Survival months distribution
    st.subheader("Survival Months Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data=data, x='Survival Months', hue=data['Survival Months'] // 12, palette='muted', ax=ax)
    ax.set_title("Survival Months Distribution")
    st.pyplot(fig)

    # Status distribution
    st.subheader("Status Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=data, x='Status', hue='Status', palette='muted', ax=ax)
    ax.set_title("Status Distribution")
    st.pyplot(fig)

def modelAnalysis():
    
    ############################################ Modeling Analysis #############################################
    
    # Data Preprocessing
    st.markdown("---")
    st.subheader("Data Preprocessing")
    st.markdown("---")

    # Label encode classification columns
    le = LabelEncoder()
    pdata = data.copy()
    for i in pdata.columns:
        if pdata[i].dtype == 'object':
            pdata[i] = le.fit_transform(pdata[i])

    st.subheader("Encoded Dataset")
    st.dataframe(pdata.head())

    # Copy of encoded dataset to use for survival prediction
    pdataS = pdata.copy()

    ################ Survival Model ################

    st.markdown("---")
    st.subheader("Survival Model (Multi-Class Classification)")
    st.markdown("---")

    #1. Create the new target column 'Survival Class' with 6 categories
    def categorize_survival(months):
        if months < 12:
            return 0
        elif 12 <= months < 24:
            return 1
        elif 24 <= months < 36:
            return 2
        elif 36 <= months < 48:
            return 3
        elif 48 <= months < 60:
            return 4
        elif 60 <= months < 72:
            return 5
        elif 72 <= months < 84:
            return 6
        elif 84 <= months < 96:
            return 7
        else:
            return 8
    
    # ---------------- common setup ----------------
    pdataS['Survival Class'] = pdataS['Survival Months'].apply(categorize_survival)
    X1 = pdataS.drop(['Survival Months', 'Survival Class'], axis=1)
    y1 = pdataS['Survival Class']
    class_names = ['<1', '1-2', '2-3', '3-4',
                   '4-5', '5-6', '7-8', '8-9', '9+']
    st.dataframe(X1.head())
    st.dataframe(y1.head())

    st.subheader("Survival Model (Multi-Class Classification)")
    st.write(f"Original class distribution: {Counter(y1)}")

    def eval_classifier(display_name, model, X_train, y_train, X_test, y_test, class_names):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        st.write(f"## {display_name}")
        st.write(f"Accuracy: {acc:.3f}  •  Balanced Accuracy: {bal_acc:.3f}  •  Precision: {precision:.3f}  •  Recall: {recall:.3f}  •  F1: {f1:.3f}")

        conf_mat = confusion_matrix(y_test, y_pred)
        # much smaller figure
        fig, ax = plt.subplots(figsize=(3.5, 2.5))  # shrink size
        sns.heatmap(
            pd.DataFrame(
                conf_mat,
                columns=[name for name in class_names],
                index=[name for name in class_names]
            ),
            annot=True,
            fmt="d",
            ax=ax,
            cmap="Blues",
            linewidths=0.2,
            linecolor="white",
            annot_kws={"fontsize": 5},               # tiny numbers
            cbar_kws={
                "shrink": 0.3,                       # make the bar shorter
                "aspect": 8                          # make it skinnier
            }
        )

        # Tidy styling
        ax.set_title(f"{display_name}", fontsize=9, pad=3)
        ax.set_xlabel("Predicted Survival Time (Years)", fontsize=7)
        ax.set_ylabel("True Survival Time (Years)", fontsize=7)
        ax.tick_params(axis='both', which='major', labelsize=6)
        
        # 4) Also shrink the colorbar’s tick labels
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=5)

        # 5) Tighten layout so there’s no extra padding
        plt.tight_layout(pad=0.5)
        # left, center, right = st.columns([1, 2, 1])
        # with center:
        st.pyplot(fig)

        st.write(f"Classification Report: {display_name}")
        report_dict = classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report_dict).transpose().round(2)
        #with center:
        st.table(report_df)

        return model

    # ------------------ 1. Baseline (raw/unbalanced) ------------------
    st.subheader("Baseline Models (Raw / Unbalanced Survival Class)")
    X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(
        X1, y1, test_size=0.2, stratify=y1, random_state=42
    )
    modelRFC_raw = RandomForestClassifier(random_state=42)
    modelXGB_raw = XGBClassifier(random_state=42, eval_metric='logloss')
    eval_classifier("Random Forest (Raw)", modelRFC_raw, X_raw_train, y_raw_train, X_raw_test, y_raw_test, class_names)
    eval_classifier("XGBoost (Raw)", modelXGB_raw, X_raw_train, y_raw_train, X_raw_test, y_raw_test, class_names)

    # ------------------ 2. Oversampled ------------------
    st.subheader("Oversampled Survival Class Pipeline")
    desired_sampling_strategy = {
        0: 600, 1: 600, 2: 600, 3: 600,
        4: 1200, 5: 1300, 6: 1300, 7: 1300, 8: 1300
    }
    ros = RandomOverSampler(random_state=42, sampling_strategy=desired_sampling_strategy)
    X_os, y_os = ros.fit_resample(X1, y1)
    st.write(f"Resampled class distribution: {Counter(y_os)}")

    X_os_train, X_os_test, y_os_train, y_os_test = train_test_split(
        X_os, y_os, test_size=0.2, random_state=42
    )
    modelRFC_os = RandomForestClassifier(random_state=42)
    modelXGB_os = XGBClassifier(random_state=42, eval_metric='logloss')
    eval_classifier("Random Forest (Oversampled)", modelRFC_os, X_os_train, y_os_train, X_os_test, y_os_test, class_names)
    eval_classifier("XGBoost (Oversampled)", modelXGB_os, X_os_train, y_os_train, X_os_test, y_os_test, class_names)

    # ------------------ 3. Oversampled + ENN cleaning ------------------
    st.subheader("After ENN Cleaning")
    enn = EditedNearestNeighbours(sampling_strategy='all', n_neighbors=2, kind_sel='all')
    X_cleaned, y_cleaned = enn.fit_resample(X_os, y_os)
    st.write(f"Class distribution after ENN: {Counter(y_cleaned)}")

    X_cln_train, X_cln_test, y_cln_train, y_cln_test = train_test_split(
        X_cleaned, y_cleaned, test_size=0.2, stratify=y_cleaned, random_state=42
    )
    modelRFC_cln = RandomForestClassifier(random_state=42)
    modelXGB_cln = XGBClassifier(random_state=42, eval_metric='logloss')
    eval_classifier("Random Forest (Oversampled + ENN)", modelRFC_cln, X_cln_train, y_cln_train, X_cln_test, y_cln_test, class_names)
    eval_classifier("XGBoost (Oversampled + ENN)", modelXGB_cln, X_cln_train, y_cln_train, X_cln_test, y_cln_test, class_names)

    # ---- Summary table ----
    st.subheader("Summary of Accuracy and Balanced Accuracy for all models")
    summary_rows = []
    def collect(name, model, X_test, y_test):
        y_pred = model.predict(X_test)
        summary_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        })

    # Baseline
    collect("RF Raw", modelRFC_raw, X_raw_test, y_raw_test)
    collect("XGB Raw", modelXGB_raw, X_raw_test, y_raw_test)
    # Oversampled
    collect("RF OS", modelRFC_os, X_os_test, y_os_test)
    collect("XGB OS", modelXGB_os, X_os_test, y_os_test)
    # Oversampled + ENN
    collect("RF OS+ENN", modelRFC_cln, X_cln_test, y_cln_test)
    collect("XGB OS+ENN", modelXGB_cln, X_cln_test, y_cln_test)

    summary_df = pd.DataFrame(summary_rows).round(3)

    # Sort by Accuracy desc, then Balanced Accuracy desc
    sorted_df = summary_df.sort_values(
        by=["Accuracy", "Balanced Accuracy"],
        ascending=[False, False]
    ).reset_index(drop=True)

    # Add rank (1 = best accuracy)
    sorted_df.insert(0, "Rank", range(1, len(sorted_df) + 1))

    # Display narrow
    st.dataframe(
        sorted_df.set_index("Rank"),
        width=400,
        use_container_width=False
    )




def main():
    """Main function to set up the Streamlit application's navigation and content."""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to", 
        ["Application", "Data Exploration", "Modeling Results"]
    )

    # Inject wide-mode CSS only on the Application page
    if page == "Application":
        st.markdown(
            """
            <style>
              /* bump up from ~700px to 1200px, still centered */
              .block-container {
                max-width: 1200px !important;
              }
            </style>
            """,
            unsafe_allow_html=True,
        )
        trainModel()
    elif page == "Data Exploration":
        dataExploration()
    elif page == "Modeling Results":
        modelAnalysis()

if __name__ == "__main__":
    main()





