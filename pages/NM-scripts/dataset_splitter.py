# Takes a random 20% of all entries from the original dataset for validation
# The remaining 80% go to the training set

import pandas as pd
from sklearn.model_selection import train_test_split

input_path = '../../Data/NM-datasets/Breast_Cancer.csv'
train_output_path = '../../Data/NM-datasets/Breast_Cancer_train.csv'
val_output_path = '../../Data/NM-datasets/Breast_Cancer_val.csv'

# Load the dataset
try:
    df = pd.read_csv(input_path)
    print(f"Dataset loaded from: {input_path}")
    print(f"Total entries: {len(df)}")
    print(f"Class distribution in original dataset:")
    print(f"Alive: {sum(df['Status']=='Alive')} ({100*sum(df['Status']=='Alive')/len(df):.1f}%)")
    print(f"Dead: {sum(df['Status']=='Dead')} ({100*sum(df['Status']=='Dead')/len(df):.1f}%)")
except FileNotFoundError:
    print(f"Error: File not found at {input_path}")
    exit()
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

# Split the dataset
train_df, val_df = train_test_split(
    df, 
    test_size=0.2,
    random_state=100,
    stratify=df['Status']
)

# Save the new sets
try:
    val_df.to_csv(val_output_path, index=False)
    train_df.to_csv(train_output_path, index=False)
    print(f"\nFiles saved successfully:")
    print(f"Validation set: {val_output_path}")
    print(f"Training set: {train_output_path}")
except Exception as e:
    print(f"Error saving files: {e}")
    exit()

# Print summary
print(f"\nSplit results:")
print(f"Validation Set: {len(val_df)} rows ({100*len(val_df)/len(df):.1f}% of total)")
print(f"  Alive: {sum(val_df['Status']=='Alive')} ({100*sum(val_df['Status']=='Alive')/len(val_df):.1f}%)")
print(f"  Dead: {sum(val_df['Status']=='Dead')} ({100*sum(val_df['Status']=='Dead')/len(val_df):.1f}%)")
print(f"Training Set: {len(train_df)} rows ({100*len(train_df)/len(df):.1f}% of total)")
print(f"  Alive: {sum(train_df['Status']=='Alive')} ({100*sum(train_df['Status']=='Alive')/len(train_df):.1f}%)")
print(f"  Dead: {sum(train_df['Status']=='Dead')} ({100*sum(train_df['Status']=='Dead')/len(train_df):.1f}%)")