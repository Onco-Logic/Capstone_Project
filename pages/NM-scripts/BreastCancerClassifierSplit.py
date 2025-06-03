import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

# --- Configuration ---
# Number of K-folds
folds = 5

# File paths
train_data_path = '../../Data/NM-datasets/Breast_Cancer_train_undersampled.csv'
val_data_path = '../../Data/NM-datasets/Breast_Cancer_val.csv'

# --- Data Loading and Preparation ---
print(f"Loading training data from: {train_data_path}")
train_df = pd.read_csv(train_data_path)

print(f"Loading validation data from: {val_data_path}")
val_df = pd.read_csv(val_data_path)

le = LabelEncoder()

# Prepare Training Data
X_train_raw = train_df.drop('Status', axis=1)
Y_train = le.fit_transform(train_df['Status']) # 0 = Alive, 1 = Dead
X_train = pd.get_dummies(X_train_raw, drop_first=True)

# Prepare Validation Data
X_val_raw = val_df.drop('Status', axis=1)
Y_val = le.transform(val_df['Status'])
X_val = pd.get_dummies(X_val_raw, drop_first=True)

# Align columns between training and validation sets
common_cols = list(set(X_train.columns) & set(X_val.columns))
X_train = X_train[common_cols].reindex(columns=sorted(common_cols))
X_val = X_val[common_cols].reindex(columns=sorted(common_cols))

# Data Information
print(f"\nData split into {len(X_train)} training samples and {len(X_val)} validation samples.")

train_alive_count = (Y_train == 0).sum()
train_dead_count = (Y_train == 1).sum()
print(f"\nTotal Training Records: {len(Y_train)}")
print(f"Total Alive records: {train_alive_count} ({100 * train_alive_count / len(Y_train):.1f}%)")
print(f"Total Dead records: {train_dead_count} ({100 * train_dead_count / len(Y_train):.1f}%)")

val_alive_count = (Y_val == 0).sum()
val_dead_count = (Y_val == 1).sum()
print(f"\nTotal Validation Records: {len(Y_val)}")
print(f"Total Alive records: {val_alive_count} ({100 * val_alive_count / len(Y_val):.1f}%)")
print(f"Total Dead records: {val_dead_count} ({100 * val_dead_count / len(Y_val):.1f}%)")

# Check for duplicates between training and validation sets
X_train_tuples = [tuple(x) for x in X_train.values]
X_val_tuples = [tuple(x) for x in X_val.values]
overlap = set(X_train_tuples).intersection(set(X_val_tuples))
print(f"\nOverlap between train and validation sets: {len(overlap)} records")
print(f"  ({100 * len(overlap) / len(X_train) if len(X_train) > 0 else 0:.2f}% of training set)")
print(f"  ({100 * len(overlap) / len(X_val) if len(X_val) > 0 else 0:.2f}% of validation set)")

# --- K-Fold Cross-Validation ---
print(f"\n{folds}-Fold Cross-Validation")
kf = KFold(n_splits=folds, shuffle=True, random_state=100)

kfold_results = {
    'train_accuracy': [],
    'train_precision_alive': [], 'train_precision_dead': [],
    'train_recall_alive': [], 'train_recall_dead': [],
    'train_f1_alive': [], 'train_f1_dead': [],
    'train_cm': [],
    'test_accuracy': [],
    'test_precision_alive': [], 'test_precision_dead': [],
    'test_recall_alive': [], 'test_recall_dead': [],
    'test_f1_alive': [], 'test_f1_dead': [],
    'test_cm': []
}

for fold_num, (train_idx, test_idx) in enumerate(kf.split(X_train, Y_train)):
    X_fold_train, X_fold_test = X_train.iloc[train_idx], X_train.iloc[test_idx]
    Y_fold_train, Y_fold_test = Y_train[train_idx], Y_train[test_idx]

    model_fold = RandomForestClassifier(random_state=100)
    model_fold.fit(X_fold_train, Y_fold_train)

    # Evaluate on K-fold training subset
    Y_fold_train_pred = model_fold.predict(X_fold_train)
    kfold_results['train_accuracy'].append(accuracy_score(Y_fold_train, Y_fold_train_pred))
    
    precision_train_scores = precision_score(Y_fold_train, Y_fold_train_pred, average=None, zero_division=0)
    recall_train_scores = recall_score(Y_fold_train, Y_fold_train_pred, average=None, zero_division=0)
    f1_train_scores = f1_score(Y_fold_train, Y_fold_train_pred, average=None, zero_division=0)
    
    kfold_results['train_precision_alive'].append(precision_train_scores[0])
    kfold_results['train_precision_dead'].append(precision_train_scores[1])
    kfold_results['train_recall_alive'].append(recall_train_scores[0])
    kfold_results['train_recall_dead'].append(recall_train_scores[1])
    kfold_results['train_f1_alive'].append(f1_train_scores[0])
    kfold_results['train_f1_dead'].append(f1_train_scores[1])
    
    cm_train_fold = confusion_matrix(Y_fold_train, Y_fold_train_pred, labels=[0, 1])
    kfold_results['train_cm'].append(cm_train_fold)

    # Evaluate on K-fold test subset
    Y_fold_test_pred = model_fold.predict(X_fold_test)
    kfold_results['test_accuracy'].append(accuracy_score(Y_fold_test, Y_fold_test_pred))

    precision_test_scores = precision_score(Y_fold_test, Y_fold_test_pred, average=None, zero_division=0)
    recall_test_scores = recall_score(Y_fold_test, Y_fold_test_pred, average=None, zero_division=0)
    f1_test_scores = f1_score(Y_fold_test, Y_fold_test_pred, average=None, zero_division=0)

    kfold_results['test_precision_alive'].append(precision_test_scores[0])
    kfold_results['test_precision_dead'].append(precision_test_scores[1])
    kfold_results['test_recall_alive'].append(recall_test_scores[0])
    kfold_results['test_recall_dead'].append(recall_test_scores[1])
    kfold_results['test_f1_alive'].append(f1_test_scores[0])
    kfold_results['test_f1_dead'].append(f1_test_scores[1])

    cm_test_fold = confusion_matrix(Y_fold_test, Y_fold_test_pred, labels=[0, 1])
    kfold_results['test_cm'].append(cm_test_fold)

# Calculate average confusion matrices
avg_train_cm = np.mean(kfold_results['train_cm'], axis=0)
avg_test_cm = np.mean(kfold_results['test_cm'], axis=0)

def print_average_kfold_metrics(cm_avg, accuracy_mean,
                                precision_alive_mean, recall_alive_mean, f1_alive_mean,
                                precision_dead_mean, recall_dead_mean, f1_dead_mean,
                                title):
    
    print(f"\n{title}:")

    print(f"  Average Overall Accuracy: {100*accuracy_mean:.2f}%")
    print(f"  Alive:")
    print(f"     Precision: {100*precision_alive_mean:.2f}%, Recall: {100*recall_alive_mean:.2f}%, F1-Score: {100*f1_alive_mean:.2f}%")
    print(f"  Dead:")
    print(f"     Precision: {100*precision_dead_mean:.2f}%, Recall: {100*recall_dead_mean:.2f}%, F1-Score: {100*f1_dead_mean:.2f}%")

print_average_kfold_metrics(
    avg_train_cm,
    np.mean(kfold_results['train_accuracy']),
    np.mean(kfold_results['train_precision_alive']),
    np.mean(kfold_results['train_recall_alive']),
    np.mean(kfold_results['train_f1_alive']),
    np.mean(kfold_results['train_precision_dead']),
    np.mean(kfold_results['train_recall_dead']),
    np.mean(kfold_results['train_f1_dead']),
    f"Average {folds}-Fold Training Metrics")

print_average_kfold_metrics(
    avg_test_cm,
    np.mean(kfold_results['test_accuracy']),
    np.mean(kfold_results['test_precision_alive']),
    np.mean(kfold_results['test_recall_alive']),
    np.mean(kfold_results['test_f1_alive']),
    np.mean(kfold_results['test_precision_dead']),
    np.mean(kfold_results['test_recall_dead']),
    np.mean(kfold_results['test_f1_dead']),
    f"Average {folds}-Fold Test (Held-out Fold) Metrics")

# Final Model Training and Evaluation
print("\n--- Final Model Evaluation (Trained on Full Training Set) ---")
final_model = RandomForestClassifier(random_state=100)
final_model.fit(X_train, Y_train)

def print_evaluation_metrics(Y_true, Y_pred, dataset_name="Dataset"):
    cm = confusion_matrix(Y_true, Y_pred)
    
    print(f"\n{dataset_name} Evaluation")
    print(f"  True/False Alive: {cm[0,0]}, {cm[0,1]}")
    print(f"  True/False Dead: {cm[1,1]}, {cm[1,0]}")
    
    precision = precision_score(Y_true, Y_pred, average=None, zero_division=0)
    recall = recall_score(Y_true, Y_pred, average=None, zero_division=0)
    f1 = f1_score(Y_true, Y_pred, average=None, zero_division=0)
    acc = accuracy_score(Y_true, Y_pred)
    
    print(f"\nFinal {dataset_name} Model Metrics:")
    print(f"  Overall Accuracy: {100*acc:.2f}%")
    print(f"  Alive:")
    print(f"     Precision: {100*precision[0]:.2f}%, Recall: {100*recall[0]:.2f}%, F1-Score: {100*f1[0]:.2f}%")
    print(f"  Dead:")
    print(f"     Precision: {100*precision[1]:.2f}%, Recall: {100*recall[1]:.2f}%, F1-Score: {100*f1[1]:.2f}%")
    return acc, precision, recall, f1

# Training Set Evaluation
Y_train_pred_final = final_model.predict(X_train)
train_acc, train_precision, train_recall, train_f1 = print_evaluation_metrics(Y_train, Y_train_pred_final, "Training Set")

# Validation Set Evaluation
Y_val_pred_final = final_model.predict(X_val)
val_acc, val_precision, val_recall, val_f1 = print_evaluation_metrics(Y_val, Y_val_pred_final, "Validation Set")

# Overfitting Analysis
print("\nOverfitting Analysis")
print(f"Metric             Training    Validation   Difference")
print(f"------------------------------------------------------")
print(f"Overall Accuracy   {100*train_acc:.2f}%       {100*val_acc:.2f}%      {(100*(train_acc - val_acc)):.2f}%")
print(f"F1 Alive           {100*train_f1[0]:.2f}%       {100*val_f1[0]:.2f}%      {(100*(train_f1[0] - val_f1[0])):.2f}%")
print(f"F1 Dead            {100*train_f1[1]:.2f}%       {100*val_f1[1]:.2f}%      {(100*(train_f1[1] - val_f1[1])):.2f}%")
print(f"Precision Alive    {100*train_precision[0]:.2f}%       {100*val_precision[0]:.2f}%      {(100*(train_precision[0] - val_precision[0])):.2f}%")
print(f"Precision Dead     {100*train_precision[1]:.2f}%       {100*val_precision[1]:.2f}%      {(100*(train_precision[1] - val_precision[1])):.2f}%")
print(f"Recall Alive       {100*train_recall[0]:.2f}%       {100*val_recall[0]:.2f}%      {(100*(train_recall[0] - val_recall[0])):.2f}%")
print(f"Recall Dead        {100*train_recall[1]:.2f}%       {100*val_recall[1]:.2f}%      {(100*(train_recall[1] - val_recall[1])):.2f}%")

accuracy_diff = train_acc - val_acc
f1_alive_diff = train_f1[0] - val_f1[0]
f1_dead_diff = train_f1[1] - val_f1[1]

print(f"\nAccuracy Difference: {accuracy_diff * 100:.2f}%")
print(f"F1 Alive Difference: {f1_alive_diff * 100:.2f}%")
print(f"F1 Dead Difference: {f1_dead_diff * 100:.2f}%")