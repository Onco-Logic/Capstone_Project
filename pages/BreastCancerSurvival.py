import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import streamlit as st
import seaborn as sns
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE, KMeansSMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
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
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Breast Cancer Prognosis",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Load the dataset
file_path = 'Data/Breast_Cancer.csv'
data = pd.read_csv(file_path)

# Label encode classification columns
le = LabelEncoder()
encoders = {}
pdata = data.copy()
for i in pdata.columns:
    if pdata[i].dtype == 'object':
        pdata[i] = le.fit_transform(pdata[i])
        encoders[i] = le   # Store the encoder for each column
        le = LabelEncoder()  # Re-instantiate for the next column

# Create the new target column 'Survival Class' with 6 categories
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

# Copy of encoded dataset to use for survival prediction
pdataS = pdata.copy()
# Apply the categorization function to create 'Survival Class'
pdataS['Survival Class'] = pdataS['Survival Months'].apply(categorize_survival)
X1 = pdataS.drop(['Survival Months', 'Survival Class', 'Status'], axis=1)
y1 = pdataS['Survival Class']
class_names = ['<1', '1-2', '2-3', '3-4',
                '4-5', '5-6', '7-8', '8-9', '9+']

# ------------------ 1. Baseline (raw/unbalanced) ------------------
X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(
        X1, y1, test_size=0.2, stratify=y1, random_state=42)

# ------------------ 2. Oversampled ------------------
desired_sampling_strategy = {
        0: 600, 1: 600, 2: 600, 3: 600,
        4: 1200, 5: 1300, 6: 1300, 7: 1300, 8: 1300
    }
ros = RandomOverSampler(random_state=42, sampling_strategy=desired_sampling_strategy)
X_os, y_os = ros.fit_resample(X1, y1)
X_os_train, X_os_test, y_os_train, y_os_test = train_test_split(
        X_os, y_os, test_size=0.2, random_state=42)

# ------------------ 3. Oversampled + ENN cleaning ------------------
enn = EditedNearestNeighbours(sampling_strategy='all', n_neighbors=2, kind_sel='all')
X_cleaned, y_cleaned = enn.fit_resample(X_os, y_os)
X_cln_train, X_cln_test, y_cln_train, y_cln_test = train_test_split(
        X_cleaned, y_cleaned, test_size=0.2, stratify=y_cleaned, random_state=42)

########################################## Train Models #############################################
@st.cache_resource
def trainModels():
    # Raw
    modelRFC_raw = RandomForestClassifier(random_state=42)
    modelXGB_raw = XGBClassifier(random_state=42, eval_metric='logloss')
    modelCAT_raw = CatBoostClassifier(random_state=42, verbose=0, train_dir="models")
    modelLGBM_raw = LGBMClassifier(random_state=42, verbose=0)
    y_pred_RFC_raw = modelRFC_raw.fit(X_raw_train, y_raw_train).predict(X_raw_test)
    y_pred_XGB_raw = modelXGB_raw.fit(X_raw_train, y_raw_train).predict(X_raw_test)
    y_pred_CAT_raw = modelCAT_raw.fit(X_raw_train, y_raw_train).predict(X_raw_test)
    y_pred_LGBM_raw = modelLGBM_raw.fit(X_raw_train, y_raw_train).predict(X_raw_test)

    # Oversampled
    modelRFC_os = RandomForestClassifier(random_state=42)
    modelXGB_os = XGBClassifier(random_state=42, eval_metric='logloss')
    modelCAT_os = CatBoostClassifier(random_state=42, verbose=0, train_dir="models")
    modelLGBM_os = LGBMClassifier(random_state=42, verbose=0)
    y_pred_RFC_os = modelRFC_os.fit(X_os_train, y_os_train).predict(X_os_test)
    y_pred_XGB_os = modelXGB_os.fit(X_os_train, y_os_train).predict(X_os_test)
    y_pred_CAT_os = modelCAT_os.fit(X_os_train, y_os_train).predict(X_os_test)
    y_pred_LGBM_os = modelLGBM_os.fit(X_os_train, y_os_train).predict(X_os_test)

    # Cleaned
    modelRFC_cln = RandomForestClassifier(random_state=42)
    modelXGB_cln = XGBClassifier(random_state=42, eval_metric='logloss')
    modelCAT_cln = CatBoostClassifier(random_state=42, verbose=0, train_dir="models")
    modelLGBM_cln = LGBMClassifier(random_state=42, verbose=0)
    y_pred_RFC_cln = modelRFC_cln.fit(X_cln_train, y_cln_train).predict(X_cln_test)
    y_pred_XGB_cln = modelXGB_cln.fit(X_cln_train, y_cln_train).predict(X_cln_test)
    y_pred_CAT_cln = modelCAT_cln.fit(X_cln_train, y_cln_train).predict(X_cln_test)
    y_pred_LGBM_cln = modelLGBM_cln.fit(X_cln_train, y_cln_train).predict(X_cln_test)

    return (
        modelRFC_raw, y_pred_RFC_raw, modelXGB_raw, y_pred_XGB_raw, modelCAT_raw, y_pred_CAT_raw, modelLGBM_raw, y_pred_LGBM_raw,
        modelRFC_os, y_pred_RFC_os, modelXGB_os, y_pred_XGB_os, modelCAT_os, y_pred_CAT_os, modelLGBM_os, y_pred_LGBM_os,  
        modelRFC_cln, y_pred_RFC_cln, modelXGB_cln, y_pred_XGB_cln, modelCAT_cln, y_pred_CAT_cln, modelLGBM_cln, y_pred_LGBM_cln
    )

########################################## Calibrate Models #############################################
def calibrateModels():
    """Calibrate each OS+ENN model once and stash into session_state."""
    cleaned_models = {
        "Random Forest (OS+ENN)": modelRFC_cln,
        "XGBoost (OS+ENN)"      : modelXGB_cln,
        "CatBoost (OS+ENN)"     : modelCAT_cln,
        "LightGBM (OS+ENN)"     : modelLGBM_cln
    }
    # 2. For each base model, create a CalibratedClassifierCV wrapper:
    #    - estimator=your already-trained model
    #    - cv=5: use 5-fold cross‐validation on the training set to learn the mapping
    #    - method="isotonic": isotonic regression calibration (non-parametric)
    for name, mdl in cleaned_models.items():
        calib = CalibratedClassifierCV(
            estimator=mdl,
            cv=5,
            method="isotonic"
        )
        # 3. Fit the calibrator using the same cleaned training data you used before
        calib.fit(X_cln_train, y_cln_train)
        # 4. Store the newly calibrated classifier in Streamlit’s session_state
        #    under a key like "cal_Random Forest (OS+ENN)"
        st.session_state[f"cal_{name}"] = calib

    # 5. Set the “default” classifier in session_state["clf"] if not already set.
    #    In your app you refer to session_state["clf"] whenever you need to predict.
    default_name = "Random Forest (OS+ENN)"
    st.session_state.setdefault("clf", st.session_state[f"cal_{default_name}"])
    # 6. Mark that calibration has been done so you don’t re-calibrate on every rerun
    st.session_state["calibrated"] = True

############################################# Data Exploration #############################################
def dataExploration():
    st.title("Data Exploration")
    # st.markdown("---")

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

############################################ Modeling Analysis #############################################
def modelAnalysis():
    st.title("Survival Model (Multi-Class Classification)")
    # Prepare two columns: left for detailed evals, right for the summary table
    left_col, right_col = st.columns([3, 2])

    def eval_classifier(display_name, X_train, y_train, X_test, y_test, y_pred, class_names):
        # model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        st.subheader(f" {display_name}")
        st.write(f"Accuracy: {acc:.3f}  •  Balanced Accuracy: {bal_acc:.3f}  •  Precision: {precision:.3f}  •  Recall: {recall:.3f}  •  F1: {f1:.3f}")

        conf_mat = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
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
        
        # Also shrink the colorbar’s tick labels
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=5)

        # Tighten layout so there’s no extra padding
        plt.tight_layout(pad=0.5)
        st.pyplot(fig)

        st.write(f"Classification Report:")
        report_dict = classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report_dict).transpose().round(2)
        st.table(report_df)

    def format_class_dist(counts, class_names, per_line=3):
        # Build one entry per class
        entries = [
            f"{class_names[i]} years: {int(counts.get(i, 0))} samples"
            for i in range(len(class_names))
        ]
        # Chunk into rows of `per_line` entries
        lines = [
            " | ".join(entries[i : i + per_line])
            for i in range(0, len(entries), per_line)
        ]
        # Join rows with a blank line in between for readability
        return "\n\n".join(lines)

# ------------------ display per selected dataset ------------------
    with left_col:
        dataset_option = st.selectbox(
            "Select dataset to display models for:",
            ["Raw", "Oversampled", "Oversampled + ENN"]
        )
        if dataset_option == "Raw":
            st.markdown("#### Original class distribution")
            raw_counts = pd.Series(y1).value_counts().sort_index()
            # st.markdown(format_class_dist(raw_counts, class_names))
            # ——— Bar chart of Raw distribution ———
            dist_df = pd.DataFrame({
                "Survival Class": class_names,
                "Count": [int(raw_counts.get(i, 0)) for i in range(len(class_names))]
            })
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.barplot(
                x="Survival Class", y="Count", data=dist_df,
                order=class_names, ax=ax
            )
            ax.set_xlabel("Survival Time (years)")
            ax.set_ylabel("Number of Samples")
            ax.set_title("Raw Samples per Survival Class")
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)
            # ————————————————————————————
            eval_classifier("Random Forest (Raw)",  X_raw_train, y_raw_train, X_raw_test, y_raw_test, y_pred_RFC_raw, class_names)
            eval_classifier("XGBoost (Raw)",         X_raw_train, y_raw_train, X_raw_test, y_raw_test, y_pred_XGB_raw, class_names)
            eval_classifier("CatBoost (Raw)",        X_raw_train, y_raw_train, X_raw_test, y_raw_test, y_pred_CAT_raw, class_names)
            eval_classifier("LightGBM (Raw)",        X_raw_train, y_raw_train, X_raw_test, y_raw_test, y_pred_LGBM_raw, class_names)

        elif dataset_option == "Oversampled":
            st.markdown("#### Oversampled class distribution")
            os_counts = pd.Series(y_os).value_counts().sort_index()
            # st.markdown(format_class_dist(os_counts, class_names))
            # ——— Bar chart of Oversampled distribution ———
            dist_df = pd.DataFrame({
                "Survival Class": class_names,
                "Count": [int(os_counts.get(i, 0)) for i in range(len(class_names))]
            })
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.barplot(
                x="Survival Class", y="Count", data=dist_df,
                order=class_names, ax=ax
            )
            ax.set_xlabel("Survival Time (years)")
            ax.set_ylabel("Number of Samples")
            ax.set_title("Oversampled Samples per Survival Class")
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)
            # ——————————————————————————————————————————
            eval_classifier("Random Forest (Oversampled)",    X_os_train,  y_os_train, X_os_test, y_os_test, y_pred_RFC_os, class_names)
            eval_classifier("XGBoost (Oversampled)",          X_os_train,  y_os_train, X_os_test, y_os_test, y_pred_XGB_os, class_names)
            eval_classifier("CatBoost (Oversampled)",         X_os_train,  y_os_train, X_os_test, y_os_test, y_pred_CAT_os, class_names)
            eval_classifier("LightGBM (Oversampled)",         X_os_train,  y_os_train, X_os_test, y_os_test, y_pred_LGBM_os, class_names)

        else:  # Oversampled + ENN
            st.markdown("#### ENN-cleaned class distribution")
            cln_counts = pd.Series(y_cleaned).value_counts().sort_index()
            # st.markdown(format_class_dist(cln_counts, class_names))
            # ——— Bar chart of ENN-cleaned distribution ———
            dist_df = pd.DataFrame({
                "Survival Class": class_names,
                "Count": [int(cln_counts.get(i, 0)) for i in range(len(class_names))]
            })
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.barplot(
                x="Survival Class", y="Count", data=dist_df,
                order=class_names, ax=ax
            )
            ax.set_xlabel("Survival Time (years)")
            ax.set_ylabel("Number of Samples")
            ax.set_title("ENN-Cleaned Samples per Survival Class")
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)
            # ——————————————————————————————————————————
            eval_classifier("Random Forest (Oversampled+ENN)", X_cln_train, y_cln_train, X_cln_test, y_cln_test, y_pred_RFC_cln, class_names)
            eval_classifier("XGBoost (Oversampled+ENN)",       X_cln_train, y_cln_train, X_cln_test, y_cln_test, y_pred_XGB_cln, class_names)
            eval_classifier("CatBoost (Oversampled+ENN)",      X_cln_train, y_cln_train, X_cln_test, y_cln_test, y_pred_CAT_cln, class_names)
            eval_classifier("LightGBM (Oversampled+ENN)",      X_cln_train, y_cln_train, X_cln_test, y_cln_test, y_pred_LGBM_cln, class_names)
 
    # ---- Summary table on the right ----
    with right_col:
        st.markdown("### Model Summary Results")
        summary_rows = []
        
        def collect(name, X_test, y_test, y_pred):
            # y_pred = model.predict(X_test)
            summary_rows.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
            })
        
        # Baseline
        collect("RF Raw", X_raw_test, y_raw_test, y_pred_RFC_raw)
        collect("XGB Raw", X_raw_test, y_raw_test, y_pred_XGB_raw)
        collect("CAT Raw", X_raw_test, y_raw_test, y_pred_CAT_raw)
        collect("LGBM Raw", X_raw_test, y_raw_test, y_pred_LGBM_raw)

        # Oversampled
        collect("RF OS", X_os_test, y_os_test, y_pred_RFC_os)
        collect("XGB OS", X_os_test, y_os_test, y_pred_XGB_os)
        collect("CAT OS", X_os_test, y_os_test, y_pred_CAT_os)
        collect("LGBM OS", X_os_test, y_os_test, y_pred_LGBM_os)

        # Oversampled + ENN
        collect("RF OS+ENN", X_cln_test, y_cln_test, y_pred_RFC_cln)
        collect("XGB OS+ENN", X_cln_test, y_cln_test, y_pred_XGB_cln)
        collect("CAT OS+ENN", X_cln_test, y_cln_test, y_pred_CAT_cln)
        collect("LGBM OS+ENN", X_cln_test, y_cln_test, y_pred_LGBM_cln)

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
            width=360,
            use_container_width=False,
            height=456       # ← set your desired vertical height here
        )

def application():
    # Print metrics for cleaned data
    acc_cln = accuracy_score(y_cln_test, y_pred_RFC_cln)

    # CHANGED: grouped all required objects for prediction; removed unused raw_data & encoded_data
    # st.session_state.setdefault("clf", modelRFC_cln)
    st.session_state.setdefault("feature_cols", list(X1.columns))
    st.session_state.setdefault("encoders", encoders)
    st.session_state.setdefault("class_names", class_names)
    st.session_state.setdefault("X_test", X_cln_test.reset_index(drop=True))
    st.session_state.setdefault("y_test", y_cln_test.reset_index(drop=True))
    st.session_state.setdefault("X_train", X_cln_train.reset_index(drop=True))
    st.session_state.setdefault("y_train", y_cln_train.reset_index(drop=True))

    ### Interactive Risk Predictor
    st.title("Breast Cancer Survival Risk Predictor")

    clf = st.session_state["clf"]
    feature_cols = st.session_state["feature_cols"]
    encs = st.session_state["encoders"]
    # CHANGED: rename local reference to session_class_names to avoid shadowing global
    session_class_names = st.session_state["class_names"]

    # Initialize form inputs once
    if "inputs_cleared" not in st.session_state:
        for col in feature_cols:
            st.session_state.pop(col, None)
            st.session_state.pop(f"{col}_raw", None)
        st.session_state["inputs_cleared"] = True

    # Callback to load random test sample
    def load_random_sample():
        idx = np.random.randint(len(st.session_state["X_test"]))
        sample = st.session_state["X_test"].iloc[idx]
        for col in feature_cols:
            if col in encs:
                val = encs[col].inverse_transform([int(sample[col])])[0]
                st.session_state[col] = val
            else:
                st.session_state[col] = float(sample[col])
                st.session_state[f"{col}_raw"] = str(sample[col])
        st.session_state["loaded_idx"] = idx

    left_col, right_col = st.columns([5, 4])

    # RIGHT: Load sample & prediction
    with right_col:
        st.subheader("Prediction")
        # now pulling from session_state
        model_options = {
            name: (st.session_state[f"cal_{name}"], st.session_state.get(f"acc_{name}", 0))
            for name in [
                "Random Forest (OS+ENN)",
                "XGBoost (OS+ENN)",
                "CatBoost (OS+ENN)",
                "LightGBM (OS+ENN)",
            ]
        }
        selected = st.selectbox("Select model pipeline", list(model_options.keys()), index=0)
        clf, acc = model_options[selected]
        st.session_state["clf"]           = clf
        st.session_state["model_name"]    = selected
        st.session_state["model_accuracy"]= acc

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            st.button("Load Random Sample", on_click=load_random_sample)
        with btn_col2:
            do_predict = st.button("Predict Risk Level")

        if "loaded_idx" in st.session_state:
            st.success(f"Loaded random patient at index **{st.session_state['loaded_idx']}**.")

        # Prediction logic: collect inputs, validate, then predict
        user_inputs, missing = {}, []
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
                raw_key = f"{col}_raw"
                inp = st.session_state.get(raw_key, "")
                # treat blank or whitespace-only as missing
                if not inp or not inp.strip():
                    missing.append(col)
                else:
                    try:
                        user_inputs[col] = float(inp)
                    except ValueError:
                        missing.append(col)
                # val = st.session_state.get(col, None)
                # if val is None:
                #     missing.append(col)
                # else:
                #     user_inputs[col] = val

        if do_predict:
            if missing:
                st.warning(f"Cannot predict: missing inputs for {', '.join(missing)}.")
            else:
                X_new = pd.DataFrame([user_inputs])[feature_cols]
                pred_arr = clf.predict(X_new)
                pred = int(pred_arr[0])
                risk = "HIGH" if pred <= 2 else "MEDIUM" if pred <= 5 else "LOW"
                est_label = session_class_names[pred]

                # Confidence calculation
                confidence_str = "N/A"
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(X_new)[0]
                    confidence_str = f"{probs[pred]*100:.1f}%"

                info = (
                    f"**Model:** {st.session_state['model_name']}  •  "
                    f"**Test Accuracy:** {st.session_state['model_accuracy']:.3f}  •  "
                    f"**Confidence:** {confidence_str}"
                )

                # Persist into session_state
                st.session_state["prediction_result"] = {
                    "info": info,
                    "risk": risk,
                    "est_label": est_label
                }

        # Always render the last prediction (until next Predict click)
        if "prediction_result" in st.session_state:
            res = st.session_state["prediction_result"]
            st.write("### Results")
            st.markdown(res["info"])
            if res["risk"] == "HIGH":
                st.error(f"🟥 High Risk ({res['est_label']} years of survival)")
            elif res["risk"] == "MEDIUM":
                st.warning(f"🟧 Medium Risk ({res['est_label']} years of survival)")
            else:
                st.success(f"🟩 Low Risk ({res['est_label']} years of survival)")

    # LEFT: Input form (starts blank unless a sample was loaded)
    with left_col:
        st.subheader("Enter patient characteristics:")
        col1, col2 = st.columns(2)
        for i, col in enumerate(feature_cols):
            target = col1 if i % 2 == 0 else col2
            with target:
                if col in encs:
                    opts = [""] + list(encs[col].classes_)  # blank leading option
                    st.selectbox(col, opts, key=col)
                else:
                    raw_key = f"{col}_raw"
                    inp = st.text_input(col, key=raw_key, placeholder="e.g., 3.4")
                    try:
                        st.session_state[col] = float(inp) if inp.strip() else None
                    except ValueError:
                        st.warning(f"Invalid number for {col}; please enter a numeric value.")
    

def main():
    """Main function to set up the Streamlit application's navigation and content."""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to", 
        ["Data Exploration", "Modeling Results", "Application"]
    )    
    # Train (once, cached) and unpack into globals
    global modelRFC_raw, y_pred_RFC_raw, modelXGB_raw, y_pred_XGB_raw, modelCAT_raw, y_pred_CAT_raw, modelLGBM_raw, y_pred_LGBM_raw
    global modelRFC_os, y_pred_RFC_os, modelXGB_os, y_pred_XGB_os, modelCAT_os, y_pred_CAT_os, modelLGBM_os, y_pred_LGBM_os
    global modelRFC_cln, y_pred_RFC_cln, modelXGB_cln, y_pred_XGB_cln, modelCAT_cln, y_pred_CAT_cln, modelLGBM_cln, y_pred_LGBM_cln
    (
        modelRFC_raw, y_pred_RFC_raw, modelXGB_raw, y_pred_XGB_raw, modelCAT_raw, y_pred_CAT_raw, modelLGBM_raw, y_pred_LGBM_raw,
        modelRFC_os, y_pred_RFC_os, modelXGB_os, y_pred_XGB_os, modelCAT_os, y_pred_CAT_os, modelLGBM_os, y_pred_LGBM_os,
        modelRFC_cln, y_pred_RFC_cln, modelXGB_cln, y_pred_XGB_cln, modelCAT_cln, y_pred_CAT_cln, modelLGBM_cln, y_pred_LGBM_cln
    ) = trainModels()

    # ——— Store each cleaned‐pipeline model’s test accuracy ———
    st.session_state["acc_Random Forest (OS+ENN)"] = accuracy_score(y_cln_test, y_pred_RFC_cln)
    st.session_state["acc_XGBoost (OS+ENN)"]       = accuracy_score(y_cln_test, y_pred_XGB_cln)
    st.session_state["acc_CatBoost (OS+ENN)"]      = accuracy_score(y_cln_test, y_pred_CAT_cln)
    st.session_state["acc_LightGBM (OS+ENN)"]      = accuracy_score(y_cln_test, y_pred_LGBM_cln)
    # ————————————————————————————————————————————————

    if "calibrated" not in st.session_state:
        calibrateModels()

    if page == "Data Exploration":
        dataExploration()
    elif page == "Modeling Results":
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
        modelAnalysis()
    elif page == "Application":
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
        application()

if __name__ == "__main__":
    main()