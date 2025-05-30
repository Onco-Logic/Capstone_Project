import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Load the training data
try:
    df = pd.read_csv('../../Data/NM-datasets/Breast_Cancer_train.csv')
    print("Successfully loaded 'Breast_Cancer_train.csv'")
except FileNotFoundError:
    print("Error: 'Breast_Cancer_train.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# Separate features (X) and target (y)
X = df.drop('Status', axis=1)
y = df['Status']

# --- Preprocessing ---

# 1. Encode the target variable 'Status' into 0 and 1
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# 'Alive' is encoded as 0, 'Dead' is encoded as 1 (minority class)

print("\nOriginal class distribution:")
print(y.value_counts())

# 2. Apply one-hot encoding to categorical features
# This converts columns with text into a numerical format for the model
categorical_cols = X.select_dtypes(include=['object']).columns
X_processed = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print(f"\nOriginal number of columns: {len(X.columns)}")
print(f"Number of columns after one-hot encoding: {len(X_processed.columns)}")
print("This number is higher because categorical features were converted to a numerical format.")

# --- Applying SMOTE ---

# Initialize SMOTE. It will only oversample the minority class.
smote = SMOTE(random_state=42)

# Fit and apply the transform
X_resampled, y_resampled_encoded = smote.fit_resample(X_processed, y_encoded)

# --- Post-processing ---

# Convert the resampled target back to original labels ('Alive', 'Dead')
y_resampled = label_encoder.inverse_transform(y_resampled_encoded)

# Create a new balanced DataFrame
df_balanced = pd.DataFrame(X_resampled, columns=X_processed.columns)
df_balanced['Status'] = y_resampled

print("\nNew class distribution after SMOTE:")
print(df_balanced['Status'].value_counts())

# Save the new balanced dataset to a CSV file
output_filename = '../../Data/NM-datasets/Breast_Cancer_train_smote.csv'
df_balanced.to_csv(output_filename, index=False)

print(f"\nSuccessfully created balanced dataset and saved it as '{output_filename}'")
print(f"The new dataset will have {df_balanced.shape[0]} rows and {df_balanced.shape[1]} columns.")