import pandas as pd
import random
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import os
import re

# File Paths
train_data_path = '../../Data/NM-datasets/Breast_Cancer_train.csv'
val_data_path = '../../Data/NM-datasets/Breast_Cancer_val.csv'

dataset_type = "normal"

best_f1_avg = -1.0
best_model_params = None

# Check if previous best metrics exist
if os.path.exists('best_model_metrics.txt'):
    try:
        with open('best_model_metrics.txt', 'r') as f:
            metrics_text = f.read()
            # Extract F1-Score using regex
            match = re.search(r'Average F1-Score: (\d+\.\d+)%', metrics_text)
            if match:
                prev_best_f1 = float(match.group(1)) / 100  # Convert from percentage
                best_f1_avg = prev_best_f1
                print(f"Found previous best model with Average F1-Score: {100*best_f1_avg:.2f}%")
                print("Will only save new models if they beat this score.")
            else:
                print("Could not find F1-Score in previous metrics file. Starting fresh.")
    except Exception as e:
        print(f"Error reading previous metrics: {e}. Starting fresh.")
else:
    print("No previous best model found. Starting fresh.")

# Menu
while True:
    try:
        num_trials = int(input("\nHow many times do you want to train the model? "))
        if num_trials > 0:
            break
        elif num_trials > 9999:
            print("Wow, that's a large number. Alright...")
            break
        else:
            print("Please enter a positive number.")
    except ValueError:
        print("Invalid input. Please enter a number.")

# Load Datasets
print(f"\nLoading training data from: {train_data_path}")
train_df = pd.read_csv(train_data_path)

print(f"Loading validation data from: {val_data_path}")
val_df = pd.read_csv(val_data_path)

# Prepare Training and Validation Data
X_train_raw = train_df.drop('Status', axis=1)
X_val_raw = val_df.drop('Status', axis=1)

le = LabelEncoder()
Y_train_encoded = le.fit_transform(train_df['Status']) # 0 = Alive, 1 = Dead
Y_val = le.transform(val_df['Status'])

X_train = pd.get_dummies(X_train_raw, drop_first=True)
X_val = pd.get_dummies(X_val_raw, drop_first=True)

# Align columns to ensure consistency
train_cols = X_train.columns
valid_cols = X_val.columns
missing_in_valid = set(train_cols) - set(valid_cols)
for c in missing_in_valid:
    X_val[c] = 0
missing_in_train = set(valid_cols) - set(train_cols)
for c in missing_in_train:
    X_train[c] = 0
X_val = X_val[train_cols]

# Balance Training Data via Oversampling
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, Y_train_resampled = oversampler.fit_resample(X_train, Y_train_encoded)

# Data & Class Distribution Info
val_alive_count = (Y_val == 0).sum()
val_dead_count = (Y_val == 1).sum()
train_resampled_alive_count = (Y_train_resampled == 0).sum()
train_resampled_dead_count = (Y_train_resampled == 1).sum()

print(f"\nData split into {len(X_train_resampled)} training samples and {len(X_val)} validation samples.")

print(f"\nTotal Alive records: {train_resampled_alive_count} ({100 * train_resampled_alive_count / len(Y_train_resampled):.1f}%)")
print(f"Total Dead records: {train_resampled_dead_count} ({100 * train_resampled_dead_count / len(Y_train_resampled):.1f}%)")
print(f"Total training records: {len(Y_train_resampled)}")

print(f"\nTotal Alive records: {val_alive_count} ({100 * val_alive_count / len(Y_val):.1f}%)")
print(f"Total Dead records: {val_dead_count} ({100 * val_dead_count / len(Y_val):.1f}%)")
print(f"Total validation records: {len(Y_val)}")


# Hyperparameter Tuning Loop
param_space = {
    'n_estimators': [10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 500],
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, None],
    'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8],
    'bootstrap': [True, False],
    'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],
    'class_weight': [None, 'balanced', 'balanced_subsample']
}

for trial in range(1, num_trials + 1):
    print(f"\n--- Starting Trial {trial}/{num_trials} ---")
    current_params = {key: random.choice(value) for key, value in param_space.items()}
    print("Testing with parameters:")
    print(current_params)

    clf = RandomForestClassifier(random_state=100, **current_params)
    print("\nTraining RandomForestClassifier...")
    clf.fit(X_train_resampled, Y_train_resampled)

    print("Testing on validation set...")
    Y_predict = clf.predict(X_val)

    # Validation Set Evaluation
    cm = confusion_matrix(Y_val, Y_predict)
    true_alive, false_alive, false_dead, true_dead = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    
    precision = precision_score(Y_val, Y_predict, average=None, zero_division=0)
    recall = recall_score(Y_val, Y_predict, average=None, zero_division=0)
    f1 = f1_score(Y_val, Y_predict, average=None, zero_division=0)

    print("\nValidation Set Evaluation:")
    print(f"-------------")
    print(f"True Alive: {true_alive}")
    print(f"False Alive: {false_alive}")
    print(f"False Dead: {false_dead}")
    print(f"True Dead: {true_dead}")
    print(f"-------------")
    
    dead_accuracy = recall[1]

    # Validation Set Evaluation
    print(f"Class          Precision    Recall       F1-Score")
    print(f"-----------------------------------------------")
    print(f"Alive (0)      {100 * precision[0]:.2f}%       {100 * recall[0]:.2f}%       {100 * f1[0]:.2f}%")
    print(f"Dead (1)       {100 * precision[1]:.2f}%       {100 * recall[1]:.2f}%       {100 * f1[1]:.2f}%")

    # Calculate average F1-score across both classes
    f1_avg = (f1[0] + f1[1]) / 2.0

    print(f"\nOverall Accuracy: {100 * accuracy_score(Y_val, Y_predict):.2f}%")
    print(f"Alive Accuracy: {100 * recall[0]:.2f}%")
    print(f"Dead Accuracy: {100 * recall[1]:.2f}%")
    print(f"Average F1-Score: {100 * f1_avg:.2f}%")

    # Check if this is the best model so far (using average F1)
    if f1_avg > best_f1_avg:
        best_f1_avg = f1_avg
        best_model_params = current_params
        
        # --- Calculate Metrics for the Best Model (for Overfitting Analysis) ---
        # Training set metrics
        Y_train_pred_for_best = clf.predict(X_train_resampled)
        train_acc = accuracy_score(Y_train_resampled, Y_train_pred_for_best)
        train_precision = precision_score(Y_train_resampled, Y_train_pred_for_best, average=None, zero_division=0)
        train_recall = recall_score(Y_train_resampled, Y_train_pred_for_best, average=None, zero_division=0)
        train_f1 = f1_score(Y_train_resampled, Y_train_pred_for_best, average=None, zero_division=0)

        # Validation set metrics
        val_acc = accuracy_score(Y_val, Y_predict)
        val_precision = precision
        val_recall = recall
        val_f1 = f1

        # --- Overfitting Analysis ---
        overfitting_analysis_output = "\n--- Overfitting Analysis for Best Model ---\n"
        overfitting_analysis_output += "Metric             Training    Validation   Difference\n"
        overfitting_analysis_output += "------------------------------------------------------\n"
        overfitting_analysis_output += f"Overall Accuracy   {100*train_acc:.2f}%       {100*val_acc:.2f}%      {(100*(train_acc - val_acc)):.2f}%\n"
        
        # Ensure metrics arrays have enough elements before accessing
        train_f1_alive = train_f1[0] if len(train_f1) > 0 else 0.0
        train_f1_dead = train_f1[1] if len(train_f1) > 1 else 0.0
        val_f1_alive = val_f1[0] if len(val_f1) > 0 else 0.0
        val_f1_dead = val_f1[1] if len(val_f1) > 1 else 0.0

        train_precision_alive = train_precision[0] if len(train_precision) > 0 else 0.0
        train_precision_dead = train_precision[1] if len(train_precision) > 1 else 0.0
        val_precision_alive = val_precision[0] if len(val_precision) > 0 else 0.0
        val_precision_dead = val_precision[1] if len(val_precision) > 1 else 0.0

        train_recall_alive = train_recall[0] if len(train_recall) > 0 else 0.0
        train_recall_dead = train_recall[1] if len(train_recall) > 1 else 0.0
        val_recall_alive = val_recall[0] if len(val_recall) > 0 else 0.0
        val_recall_dead = val_recall[1] if len(val_recall) > 1 else 0.0

        overfitting_analysis_output += f"F1 Alive (0)       {100*train_f1_alive:.2f}%       {100*val_f1_alive:.2f}%      {(100*(train_f1_alive - val_f1_alive)):.2f}%\n"
        overfitting_analysis_output += f"F1 Dead (1)        {100*train_f1_dead:.2f}%       {100*val_f1_dead:.2f}%      {(100*(train_f1_dead - val_f1_dead)):.2f}%\n"
        overfitting_analysis_output += f"Precision Alive(0) {100*train_precision_alive:.2f}%       {100*val_precision_alive:.2f}%      {(100*(train_precision_alive - val_precision_alive)):.2f}%\n"
        overfitting_analysis_output += f"Precision Dead(1)  {100*train_precision_dead:.2f}%       {100*val_precision_dead:.2f}%      {(100*(train_precision_dead - val_precision_dead)):.2f}%\n"
        overfitting_analysis_output += f"Recall Alive (0)   {100*train_recall_alive:.2f}%       {100*val_recall_alive:.2f}%      {(100*(train_recall_alive - val_recall_alive)):.2f}%\n"
        overfitting_analysis_output += f"Recall Dead (1)    {100*train_recall_dead:.2f}%       {100*val_recall_dead:.2f}%      {(100*(train_recall_dead - val_recall_dead)):.2f}%\n"
        
        print(overfitting_analysis_output) # Print to console
        
        # Create metrics text
        metrics_text = (
            f"--- BEST MODEL METRICS ---\n"
            f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"True Alive: {true_alive}\n"
            f"False Alive: {false_alive}\n"
            f"False Dead: {false_dead}\n"
            f"True Dead: {true_dead}\n\n"
            f"Class          Precision    Recall       F1-Score\n"
            f"-----------------------------------------------\n"
            f"Alive (0)      {100 * val_precision[0]:.2f}%       {100 * val_recall[0]:.2f}%       {100 * val_f1[0]:.2f}%\n"
            f"Dead (1)       {100 * val_precision[1]:.2f}%       {100 * val_recall[1]:.2f}%       {100 * val_f1[1]:.2f}%\n"
            f"Average        {100 * val_precision.mean():.2f}%   {100 * val_recall.mean():.2f}%   {100 * f1_avg:.2f}%\n\n" 
            f"Overall Val Accuracy: {100 * val_acc:.2f}%\n"
            f"Alive Val Recall: {100 * val_recall[0]:.2f}%\n"
            f"Dead Val Recall: {100 * val_recall[1]:.2f}%\n"
            f"Average F1-Score: {100 * f1_avg:.2f}%\n"
        )

        metrics_text += overfitting_analysis_output
        metrics_text += f"\nBest Model Parameters:\n"
        
        # Add parameters to txt
        for param, value in best_model_params.items():
            metrics_text += f"{param}: {value}\n"
        
        # Save metrics to text file
        with open(f'Autotunes/{dataset_type}_model_metrics.txt', 'w') as f:
            f.write(metrics_text)
        
        # Save the model ONLY when we find a better one
        joblib.dump(clf, f'Autotunes/{dataset_type}_random_forest_model.joblib')
        
        print(f"\nNew best model found!")
        print(f"New best Average F1-Score: {100*best_f1_avg:.2f}% 🥳")
        print(f"Model saved to 'Autotunes/{dataset_type}_random_forest_model.joblib'")
        print(f"Metrics saved to 'Autotunes/{dataset_type}_model_metrics.txt'")
    else:
        print(f"\nDid not beat the best Average F1-Score of {100*best_f1_avg:.2f}% :C")

print("\n--- Training complete ---")
print(f"The best model achieved an Average F1-Score of: {100*best_f1_avg:.2f}%")
print("The parameters for the best model were:")
print(best_model_params)
print("The best model is saved in 'random_forest_model.joblib'")