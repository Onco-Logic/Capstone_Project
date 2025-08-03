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

    st.write("### Random Forest with OverSampling & ENN Cleaning")
    st.write(f"Accuracy: {acc_cln:.3f}")

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
    st.markdown("---")
    st.subheader("🔮 Interactive: Predicted Risk Level")

    clf = st.session_state["clf"]
    feature_cols = st.session_state["feature_cols"]
    encs = st.session_state["encoders"]
    raw = st.session_state["raw_data"]
    encoded = st.session_state["encoded_data"]

    # 1) Load a random sample BEFORE building the form
    if st.button("🔃 Load Random Sample"):
        # draw a random index
        idx = np.random.randint(len(st.session_state["X_test"]))
        sample = st.session_state["X_test"].iloc[idx]
        # decode & store into session_state
        for col in feature_cols:
            if col in encs:
                st.session_state[col] = encs[col].inverse_transform([int(sample[col])])[0]
            else:
                st.session_state[col] = float(sample[col])
        # display the index
        st.success(f"🔍 Loaded random patient at index **{idx}**. Now click ▶️ Predict Risk Level.")


    # 2) Now build the form, using session_state[col] as your defaults
    with st.form("risk_form"):
        st.write("Enter patient characteristics:")
        for col in feature_cols:
            if col in encs:
                opts = list(encs[col].classes_)
                default = st.session_state.get(col, raw[col].mode().iloc[0])
                st.selectbox(col, opts, index=opts.index(default), key=col)
            else:
                cmin, cmax = float(encoded[col].min()), float(encoded[col].max())
                cmed = float(encoded[col].median())
                default = st.session_state.get(col, cmed)
                st.number_input(col, min_value=cmin, max_value=cmax, value=default, key=col)

        do_predict = st.form_submit_button("▶️ Predict Risk Level")

    # 3) Prediction logic
    if do_predict:
        user_inputs = {}
        for col in feature_cols:
            if col in encs:
                user_inputs[col] = encs[col].transform([st.session_state[col]])[0]
            else:
                user_inputs[col] = st.session_state[col]

        X_new = pd.DataFrame([user_inputs])[feature_cols]
        pred = clf.predict(X_new)[0]
        risk = "HIGH" if pred <= 2 else "MEDIUM" if pred <= 5 else "LOW"

        st.write("### Prediction Result")
        if risk == "HIGH":
            st.error("🟥 High Risk (< 3 years)")
        elif risk == "MEDIUM":
            st.warning("🟧 Medium Risk (3–6 years)")
        else:
            st.success("🟩 Low Risk (> 6 years)")
        st.caption(f"Predicted class index: {pred}")

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

    pdataS['Survival Class'] = pdataS['Survival Months'].apply(categorize_survival)

    # Define features (X1) and target (y1) for the classification task
    X1 = pdataS.drop(['Survival Months', 'Survival Class'], axis=1)
    y1 = pdataS['Survival Class']

    # Get unique class names for display purposes
    class_names = ['<1 Year', '1-2 Years', '2-3 Years', '3-4 Years', '4-5 Years', '5-6 Years', '7-8 Years', '8-9 Years', '9+ Years']

    st.subheader("Splitting data into X1 and y1 (Survival Class)")
    st.dataframe(X1.head())
    st.dataframe(y1.head())

    #################################################################################

    '''RandomOverSampler with modified parameters - This approach shows promise'''
    st.write(f"Original dataset shape: {Counter(y1)}")
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
    st.write(f"Resampled dataset shape after RandomOverSampler: {Counter(y1_resampled)}")

    #################################################################################

    # Splitting data into training and testing sets

    X1_train, X1_test, y1_train, y1_test = train_test_split(X1_resampled, y1_resampled, test_size=0.2, random_state=42) #resampled

    ################ Train Random Forest ################

    st.subheader("Random Forest Classifier (Oversampled Survival Pipeline)")

    modelRFC_sm = RandomForestClassifier(random_state=42)
    modelRFC_sm.fit(X1_train, y1_train)

    y1_pred_sm = modelRFC_sm.predict(X1_test)

    # Overall metrics
    accuracy_rfc = accuracy_score(y1_test, y1_pred_sm)
    balanced_accuracy_rfc = balanced_accuracy_score(y1_test, y1_pred_sm)
    precision_rfc = precision_score(y1_test, y1_pred_sm, average='weighted', zero_division=0)
    recall_rfc = recall_score(y1_test, y1_pred_sm, average='weighted', zero_division=0)
    f1_rfc = f1_score(y1_test, y1_pred_sm, average='weighted', zero_division=0)

    st.write(f"Accuracy: {accuracy_rfc:.3f}")
    st.write(f"Balanced Accuracy: {balanced_accuracy_rfc:.3f}")
    st.write(f"Precision (Weighted): {precision_rfc:.3f}")
    st.write(f"Recall (Weighted): {recall_rfc:.3f}")
    st.write(f"F1 Score (Weighted): {f1_rfc:.3f}")

    # Confusion matrix
    conf_mat_sm = confusion_matrix(y1_test, y1_pred_sm)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(conf_mat_sm,
        columns=[f"Pred {name}" for name in class_names],
        index=[f"Actual {name}" for name in class_names]),
        annot=True, cmap="Blues", fmt="d", ax=ax
    )
    ax.set_title("Random Forest Survival Class Confusion Matrix")
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    st.pyplot(fig)

    # Classification report
    st.write("Classification Report: Random Forest (Survival Class): ")
    report_rfc_dict = classification_report(y1_test, y1_pred_sm, target_names=class_names, output_dict=True, zero_division=0)
    report_rfc_df = pd.DataFrame(report_rfc_dict).transpose().round(2)
    st.table(report_rfc_df)

    ################ Train XGBoost ################

    st.subheader("XGBoost Classifier (Survival Class)")

    modelXGB_sm = XGBClassifier(random_state=42, eval_metric='logloss')
    modelXGB_sm.fit(X1_train, y1_train)

    y1_pred_xgb = modelXGB_sm.predict(X1_test)

    # Overall metrics
    accuracy_xgb = accuracy_score(y1_test, y1_pred_xgb)
    balanced_accuracy_xgb = balanced_accuracy_score(y1_test, y1_pred_xgb)
    precision_xgb = precision_score(y1_test, y1_pred_xgb, average='weighted', zero_division=0)
    recall_xgb = recall_score(y1_test, y1_pred_xgb, average='weighted', zero_division=0)
    f1_xgb = f1_score(y1_test, y1_pred_xgb, average='weighted', zero_division=0)

    st.write(f"Accuracy: {accuracy_xgb:.3f}")
    st.write(f"Balanced Accuracy: {balanced_accuracy_xgb:.3f}")
    st.write(f"Precision (Weighted): {precision_xgb:.3f}")
    st.write(f"Recall (Weighted): {recall_xgb:.3f}")
    st.write(f"F1 Score (Weighted): {f1_xgb:.3f}")

    # Confusion matrix
    conf_mat_xgb = confusion_matrix(y1_test, y1_pred_xgb)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(conf_mat_xgb,
        columns=[f"Pred {name}" for name in class_names],
        index=[f"Actual {name}" for name in class_names]),
        annot=True, cmap="Blues", fmt="d", ax=ax
    )
    ax.set_title("XGBoost Survival Class Confusion Matrix")
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    st.pyplot(fig)

    # Classification report
    st.write("Classification Report: XGBoost (Survival Class): ")
    report_xgb_dict = classification_report(y1_test, y1_pred_xgb, target_names=class_names, output_dict=True, zero_division=0)
    report_xgb_df = pd.DataFrame(report_xgb_dict).transpose().round(2)
    st.table(report_xgb_df)

    ################ Clean noisy points with ENN) ################

    #Clean noisy points with ENN (KNN‐based)

    #se ENN to clean noisy points from resampled data
    enn = EditedNearestNeighbours(
        sampling_strategy='all',
        n_neighbors=2,
        kind_sel='all'
    )

    X1_cleaned, y1_cleaned = enn.fit_resample(X1_resampled, y1_resampled)

    # Print shape of cleaned data
    st.write(f"Shape AFTER ENN cleaning: {Counter(y1_cleaned)}")

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
    bal_acc_cln = balanced_accuracy_score(y1_test_cln, y1_pred_cln)
    prec_cln = precision_score(y1_test_cln, y1_pred_cln, average='weighted', zero_division=0)
    recall_cln = recall_score(y1_test_cln, y1_pred_cln, average='weighted', zero_division=0)
    f1_cln = f1_score(y1_test_cln, y1_pred_cln, average='weighted', zero_division=0)

    st.write("### After ENN Cleaning → Random Forest Metrics")
    st.write(f"Accuracy: {acc_cln:.3f}")
    st.write(f"Balanced Accuracy: {bal_acc_cln:.3f}")
    st.write(f"Precision (Weighted): {prec_cln:.3f}")
    st.write(f"Recall (Weighted): {recall_cln:.3f}")
    st.write(f"F1 Score (Weighted): {f1_cln:.3f}")

    # Plot confusion matrix for cleaned data
    conf_mat_cln = confusion_matrix(y1_test_cln, y1_pred_cln)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pd.DataFrame(
            conf_mat_cln,
            columns=[f"Pred {name}" for name in class_names],
            index=[f"Actual {name}" for name in class_names]
        ),
        annot=True,
        cmap="Blues",
        fmt="d",
        ax=ax
    )
    ax.set_title("After ENN Cleaning: RF Survival Class Confusion Matrix")
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    st.pyplot(fig)

    #------------------- XGBoost on ENN-cleaned -------------------
    modelXGB_cln = XGBClassifier(random_state=42, eval_metric='logloss')
    modelXGB_cln.fit(X1_train_cln, y1_train_cln)
    y1_pred_xgb_cln = modelXGB_cln.predict(X1_test_cln)

    acc_xgb_cln = accuracy_score(y1_test_cln, y1_pred_xgb_cln)
    bal_acc_xgb_cln = balanced_accuracy_score(y1_test_cln, y1_pred_xgb_cln)
    prec_xgb_cln = precision_score(y1_test_cln, y1_pred_xgb_cln, average='weighted', zero_division=0)
    recall_xgb_cln = recall_score(y1_test_cln, y1_pred_xgb_cln, average='weighted', zero_division=0)
    f1_xgb_cln = f1_score(y1_test_cln, y1_pred_xgb_cln, average='weighted', zero_division=0)

    st.write("### After ENN Cleaning → XGBoost Metrics")
    st.write(f"Accuracy: {acc_xgb_cln:.3f}")
    st.write(f"Balanced Accuracy: {bal_acc_xgb_cln:.3f}")
    st.write(f"Precision (Weighted): {prec_xgb_cln:.3f}")
    st.write(f"Recall (Weighted): {recall_xgb_cln:.3f}")
    st.write(f"F1 Score (Weighted): {f1_xgb_cln:.3f}")

    # Confusion matrix XGBoost
    conf_mat_xgb_cln = confusion_matrix(y1_test_cln, y1_pred_xgb_cln)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pd.DataFrame(conf_mat_xgb_cln,
            columns=[f"Pred {name}" for name in class_names],
            index=[f"Actual {name}" for name in class_names]),
            annot=True, cmap="Blues", fmt="d", ax=ax
        )
    ax.set_title("After ENN Cleaning: XGBoost Survival Class Confusion Matrix")
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    st.pyplot(fig)

    # Optional: Full classification report
    st.write("Classification Report: XGBoost after ENN Cleaning")
    report_xgb_cln = classification_report(
        y1_test_cln, y1_pred_xgb_cln, target_names=class_names, output_dict=True, zero_division=0
    )
    st.table(pd.DataFrame(report_xgb_cln).transpose().round(2))


def main():
    """Main function to set up the Streamlit application's navigation and content."""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to", 
        ["Application", "Data Exploration", "Modeling Results"]
    )

    if page == "Application":
        trainModel()
    elif page == "Data Exploration":
        dataExploration()
    elif page == "Modeling Results":
        modelAnalysis()

if __name__ == "__main__":
    main()





