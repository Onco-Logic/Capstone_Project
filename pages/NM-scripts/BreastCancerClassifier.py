import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# File path
data_path = '../../Data/NM-datasets/Breast_CancerExtra_balanced_double_double.csv'

# Load the dataset
print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path)

# Prepare Data
X_raw = df.drop('Status', axis=1)
Y_raw = df['Status']

# Encode categorical features and the target variable
X = pd.get_dummies(X_raw, drop_first=True)
le = LabelEncoder()
Y = le.fit_transform(Y_raw) # 0 = Alive, 1 = Dead

# Split data into 80% training and 20% validation
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.20, random_state=100, stratify=Y
)

# Check for duplicates
X_train_tuples = [tuple(x) for x in X_train.values]
X_val_tuples = [tuple(x) for x in X_val.values]

# Insert counts
val_alive_count = (Y_val == 0).sum()
val_dead_count = (Y_val == 1).sum()
train_alive_count = (Y_train == 0).sum()
train_dead_count = (Y_train == 1).sum()

print(f"\nData split into {len(X_train)} training samples and {len(X_val)} validation samples.")

print(f"\nTotal Alive records: {train_alive_count} ({100 * train_alive_count / len(Y_train):.1f}%)")
print(f"Total Dead records: {train_dead_count} ({100 * train_dead_count / len(Y_train):.1f}%)")
print(f"Total training records: {len(Y_train)}")

print(f"\nTotal Alive records: {val_alive_count} ({100 * val_alive_count / len(Y_val):.1f}%)")
print(f"Total Dead records: {val_dead_count} ({100 * val_dead_count / len(Y_val):.1f}%)")
print(f"Total validation records: {len(Y_val)}")

# Random Forest Classifier
clf = RandomForestClassifier(random_state=100)
print("\nTraining RandomForestClassifier...")
clf.fit(X_train, Y_train)

print("Testing on validation set...")
Y_predict = clf.predict(X_val)

# Validation Set Evaluation
print("\nValidation Set Evaluation:")
cm = confusion_matrix(Y_val, Y_predict)

true_alive = cm[0, 0]
false_alive = cm[0, 1]
false_dead = cm[1, 0]
true_dead = cm[1, 1]
precision = precision_score(Y_val, Y_predict, average=None)
recall = recall_score(Y_val, Y_predict, average=None)
f1 = f1_score(Y_val, Y_predict, average=None)

print(f"-------------")
print(f"True Alive: {true_alive}")
print(f"False Alive: {false_alive}")
print(f"False Dead: {false_dead}")
print(f"True Dead: {true_dead}")
print(f"-------------")

print(f"Class          Precision    Recall       F1-Score")
print(f"-----------------------------------------------")
print(f"Alive (0)      {100 * precision[0]:.2f}%     {100 * recall[0]:.2f}%     {100 * f1[0]:.2f}%")
print(f"Dead (1)       {100 * precision[1]:.2f}%     {100 * recall[1]:.2f}%     {100 * f1[1]:.2f}%")

accuracy_percent = 100 * accuracy_score(Y_val, Y_predict)
alive_accuracy = 100 * (true_alive / (true_alive + false_alive))
dead_accuracy = 100 * (true_dead / (true_dead + false_dead))

print(f"\nOverall Accuracy: {accuracy_percent:.2f}%")
print(f"Alive Accuracy: {alive_accuracy:.2f}%")
print(f"Dead Accuracy: {dead_accuracy:.2f}%")

# Duplicate print stuff
overlap = set(X_train_tuples).intersection(set(X_val_tuples))
print(f"\nOverlap between train and validation sets: {len(overlap)} records")
print(f"  ({100 * len(overlap) / len(X_train):.2f}% of training set)")
print(f"  ({100 * len(overlap) / len(X_val):.2f}% of validation set)")