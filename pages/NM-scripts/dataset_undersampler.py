# This script removes alive entries until it meets the same number as dead entries

import pandas as pd
from sklearn.utils import shuffle

# Define input and output file paths
input_train_file_path = '../../Data/NM-datasets/Breast_Cancer.csv'
output_undersampled_file_path = '../../Data/NM-datasets/Breast_CancerExtra_undersampled.csv'

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

df_undersampled = df_train.copy()

if count_alive > count_dead and count_dead > 0:
    print(f"\nUndersampling 'Alive' entries from {count_alive} to match 'Dead' entries ({count_dead})...")
    
    # Randomly select a subset of 'Alive' entries equal to the number of 'Dead' entries
    undersampled_alive = df_alive.sample(n=count_dead, replace=False, random_state=100)
    
    # Combine undersampled alive with all dead entries
    df_undersampled = pd.concat([undersampled_alive, df_dead], ignore_index=True)
    
    print("Undersampling complete.")

elif count_dead > count_alive and count_alive > 0:
    print(f"\nUndersampling 'Dead' entries from {count_dead} to match 'Alive' entries ({count_alive})...")
    
    # Randomly select a subset of 'Dead' entries equal to the number of 'Alive' entries
    undersampled_dead = df_dead.sample(n=count_alive, replace=False, random_state=100)
    
    # Combine all alive with undersampled dead entries
    df_undersampled = pd.concat([df_alive, undersampled_dead], ignore_index=True)
    
    print("Undersampling complete.")

elif count_dead == count_alive:
    print("\nDataset is already balanced between 'Alive' and 'Dead' entries.")

# Shuffle the final undersampled dataset
df_undersampled = shuffle(df_undersampled, random_state=100)

print("\nClass distribution in the new undersampled dataset:")
print(df_undersampled['Status'].value_counts())
print(f"Alive: {len(df_undersampled[df_undersampled['Status'] == 'Alive'])}")
print(f"Dead: {len(df_undersampled[df_undersampled['Status'] == 'Dead'])}")

# Save the undersampled dataset
try:
    df_undersampled.to_csv(output_undersampled_file_path, index=False)
    print(f"\nSuccessfully saved undersampled data to: {output_undersampled_file_path}")
except Exception as e:
    print(f"Error saving undersampled data: {e}")