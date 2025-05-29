import pandas as pd
from sklearn.utils import shuffle

# Define input and output file paths
input_train_file_path = '../../Data/NM-datasets/Breast_Cancer_train.csv'
output_balanced_file_path = '../../Data/NM-datasets/Breast_Cancer_train_balanced.csv'

# Load the training dataset
try:
    df_train = pd.read_csv(input_train_file_path)
except FileNotFoundError:
    print(f"Error: Input training file not found at {input_train_file_path}")
    exit()
except Exception as e:
    print(f"Error reading training file: {e}")
    exit()

print(f"Original dataset loaded from: {input_train_file_path}")
print("Original class distribution:")
print(df_train['Status'].value_counts())

# Separate 'Alive' and 'Dead' entries
df_alive = df_train[df_train['Status'] == 'Alive']
df_dead = df_train[df_train['Status'] == 'Dead']

count_alive = len(df_alive)
count_dead = len(df_dead)

if count_dead == 0:
    print("Error: No 'Dead' entries found in the dataset. Cannot perform oversampling.")
    exit()

if count_alive == 0:
    print("Warning: No 'Alive' entries found. The dataset will only contain 'Dead' entries if oversampling proceeds.")
    # Depending on desired behavior, you might want to exit or handle this differently.
    # For now, we'll proceed, but the 'balanced' dataset will be just the dead entries.

df_balanced = df_train.copy() # Start with a copy

if count_dead < count_alive:
    print(f"\nOversampling 'Dead' entries from {count_dead} to match 'Alive' entries ({count_alive})...")
    
    # Calculate how many 'Dead' entries to add
    num_to_add = count_alive - count_dead
    
    # Duplicate 'Dead' entries. We use sample with replacement.
    # If num_to_add is very large, this might create many exact duplicates.
    # For a more sophisticated approach, one might cycle through df_dead.
    # However, simple duplication via sampling with replacement is common.
    
    oversampled_dead = df_dead.sample(n=num_to_add, replace=True, random_state=42) # random_state for reproducibility
    
    # Combine original dead with oversampled dead
    df_dead_balanced = pd.concat([df_dead, oversampled_dead], ignore_index=True)
    
    # Combine balanced dead with alive
    df_balanced = pd.concat([df_alive, df_dead_balanced], ignore_index=True)
    
    print("Oversampling complete.")
elif count_dead > count_alive and count_alive > 0:
    print(f"\n'Dead' entries ({count_dead}) are already more numerous than 'Alive' entries ({count_alive}).")
    print("No oversampling of 'Dead' entries performed. You might consider undersampling 'Dead' or oversampling 'Alive'.")
    # The script will save the original dataset as 'balanced' in this case.
elif count_dead == count_alive:
    print("\nDataset is already balanced between 'Alive' and 'Dead' entries.")
    # The script will save the original dataset as 'balanced' in this case.


# Shuffle the final balanced dataset
df_balanced = shuffle(df_balanced, random_state=42) # random_state for reproducibility

print("\nClass distribution in the new balanced dataset:")
print(df_balanced['Status'].value_counts())

# Save the balanced dataset
try:
    df_balanced.to_csv(output_balanced_file_path, index=False)
    print(f"\nSuccessfully saved balanced data to: {output_balanced_file_path}")
except Exception as e:
    print(f"\nError saving output file: {e}")