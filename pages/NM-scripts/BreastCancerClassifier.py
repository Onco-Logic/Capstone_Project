import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# File path
data_path = '../../Data/NM-datasets/Breast_Cancer.csv'

# Load the dataset
print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path)

# Prepare Data
# Separate features (X) and target (Y)
X_raw = df.drop('Status', axis=1)
Y_raw = df['Status']

# Encode categorical features and the target variable
X = pd.get_dummies(X_raw, drop_first=True)
le = LabelEncoder()
Y = le.fit_transform(Y_raw) # 0 = Alive, 1 = Dead

# Split data into 80% training and 20% validation
# stratify=Y ensures the split maintains the proportion of 'Alive' and 'Dead' records
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.20, random_state=100, stratify=Y
)

val_alive_count = (Y_val == 0).sum()  # Assuming 0 = Alive based on LabelEncoder
val_dead_count = (Y_val == 1).sum()   # Assuming 1 = Dead based on LabelEncoder

print(f"\nData split into {len(X_train)} training samples and {len(X_val)} validation samples.")
print(f"Total Alive records: {val_alive_count} ({100 * val_alive_count / len(Y_val):.1f}%)")
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

print(f"-------------")
print(f"True Alive: {true_alive}")
print(f"False Alive: {false_alive}")
print(f"False Dead: {false_dead}")
print(f"True Dead: {true_dead}")
print(f"-------------")

accuracy_percent = 100 * accuracy_score(Y_val, Y_predict)
# Handle division by zero case if a class has no samples in the validation set
alive_accuracy = 100 * (true_alive / (true_alive + false_alive)) if (true_alive + false_alive) > 0 else 0
dead_accuracy = 100 * (true_dead / (true_dead + false_dead)) if (true_dead + false_dead) > 0 else 0


print(f"\nOverall Accuracy: {accuracy_percent:.2f}%")
print(f"Alive Accuracy: {alive_accuracy:.2f}%")
print(f"Dead Accuracy: {dead_accuracy:.2f}%")