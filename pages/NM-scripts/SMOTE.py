import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import os

train_input_path = '../../Data/NM-datasets/Breast_Cancer_train.csv'
smote_output_path = '../../Data/NM-datasets/Breast_Cancer_train_smote.csv'

# Load the training dataset
train_df = pd.read_csv(train_input_path)

print("\nOriginal Class Distribution:")
print(train_df['Status'].value_counts())

# Separate features (X) and target (y)
X = train_df.drop('Status', axis=1)
Y = train_df['Status']

# Encode target variable (Alive, Dead) to (0, 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(Y)

# One-hot encode categorical features in X, as SMOTE requires numerical input
X_categorical = X.select_dtypes(exclude=['number'])
x_encoded = pd.get_dummies(X, columns=X_categorical.columns, drop_first=True)

# Initialize SMOTE to oversample dead entries
smote = SMOTE(random_state=100)

# Apply SMOTE
X_resampled, y_resampled_encoded = smote.fit_resample(x_encoded, y_encoded)
X_resampled_df = pd.DataFrame(X_resampled, columns=x_encoded.columns)
y_resampled_decoded = label_encoder.inverse_transform(y_resampled_encoded)

# Combine resampled features and target
df_smote = X_resampled_df.copy()
df_smote['Status'] = y_resampled_decoded

print("\nClass distribution after SMOTE:")
print(df_smote['Status'].value_counts())

# Save SMOTE dataset
df_smote.to_csv(smote_output_path, index=False)

print(f"Original rows: {len(train_df)}")
print(f"SMOTE rows: {len(df_smote)}")