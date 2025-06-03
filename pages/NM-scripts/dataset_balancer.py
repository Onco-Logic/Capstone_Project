# This script oversamples dead entries by duplicating them until they're the same amount as alive entries

import pandas as pd
from sklearn.utils import shuffle

# Define input and output file paths
input_train_file_path = '../../Data/NM-datasets/Breast_Cancer.csv'
output_balanced_file_path = '../../Data/NM-datasets/Breast_CancerExtra_balanced.csv'

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

df_balanced = df_train.copy()

if count_dead < count_alive:
    print(f"\nOversampling 'Dead' entries from {count_dead} to match 'Alive' entries ({count_alive})...")
    
    # Calculate how many 'Dead' entries to add
    num_to_add = count_alive - count_dead
    oversampled_dead = df_dead.sample(n=num_to_add, replace=True, random_state=100)
    
    # Combine original dead with oversampled dead
    df_dead_balanced = pd.concat([df_dead, oversampled_dead], ignore_index=True)
    
    # Combine balanced dead with alive
    df_balanced = pd.concat([df_alive, df_dead_balanced], ignore_index=True)
    
    print("Oversampling complete.")

elif count_dead > count_alive and count_alive > 0:
    print(f"\n'Dead' entries ({count_dead}) are already more numerous than 'Alive' entries ({count_alive})!?")

elif count_dead == count_alive:
    print("\nDataset is already balanced between 'Alive' and 'Dead' entries.")

# Shuffle the final balanced dataset
df_balanced = shuffle(df_balanced, random_state=100)

print("\nClass distribution in the new balanced dataset:")
print(df_balanced['Status'].value_counts())

# Save the balanced dataset
df_balanced.to_csv(output_balanced_file_path, index=False)
print(f"\nSuccessfully saved balanced data to: {output_balanced_file_path}")