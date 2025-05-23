# Note: Make sure to run dead_extender before running this~

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load BreastCancer dataset
df = pd.read_csv('../../Data/Breast_Cancer_Dead_Extended.csv')

# Encode vars
X = df.drop('Status', axis = 1)
X = pd.get_dummies(X, drop_first = True)

le = LabelEncoder()
Y = le.fit_transform(df['Status']) # 0 is alive, 1 is dead

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)

# Random Forest
clf = RandomForestClassifier(random_state = 42)
clf.fit(X_train, Y_train)

Y_predict = clf.predict(X_test)

accuracy_percent = 100 * accuracy_score(Y_test, Y_predict)
print("\nAccuracy: " , accuracy_percent)

# Print matrix report
cm = confusion_matrix(Y_test, Y_predict)
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

print("Classification Report: \n", classification_report(Y_test, Y_predict))