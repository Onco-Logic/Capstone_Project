import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# TODO: Tune model maybe (preferably an auto solution). Also maybe try different models besides Random Forest (a menu).

# File paths
train_data_path = '../../Data/NM-datasets/Breast_Cancer_train_balanced.csv'
val_data_path = '../../Data/NM-datasets/Breast_Cancer_val.csv'

# Load datasets
print(f"\nLoading training data from: {train_data_path}")
train_df = pd.read_csv(train_data_path)

print(f"Loading validation data from: {val_data_path}")
val_df = pd.read_csv(val_data_path)

# Prepare Training Data
X_train_raw = train_df.drop('Status', axis=1)
X_train = pd.get_dummies(X_train_raw, drop_first=True)
le = LabelEncoder()
Y_train = le.fit_transform(train_df['Status']) # 0 = Alive, 1 = Dead

# Prepare Validation Data
X_val_raw = val_df.drop('Status', axis=1)
X_val = pd.get_dummies(X_val_raw, drop_first=True)
Y_val = le.transform(val_df['Status'])

# Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
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

print(f"-------------")
print(f"True Alive: {true_alive}")
print(f"False Alive: {false_alive}")
print(f"False Dead: {false_dead}")
print(f"True Dead: {true_dead}")
print(f"-------------")

accuracy_percent = 100 * accuracy_score(Y_val, Y_predict)
alive_accuracy = 100 * (true_alive / (true_alive + false_alive))
dead_accuracy = 100 * (true_dead / (true_dead + false_dead))

print(f"\nOverall Accuracy: {accuracy_percent:.2f}%")
print(f"Alive Accuracy: {alive_accuracy:.2f}%")
print(f"Dead Accuracy: {dead_accuracy:.2f}%")