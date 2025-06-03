# This doubles all entries (both alive and dead)

import pandas as pd
from sklearn.utils import shuffle

# Define input and output file paths
input_train_file_path = '../../Data/NM-datasets/Breast_CancerExtra_balanced_double.csv'
output_doubled_file_path = '../../Data/NM-datasets/Breast_CancerExtra_balanced_double_double.csv'

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

# Make copies of both alive and dead entries to double them
print(f"\nDoubling 'Alive' entries from {count_alive} to {count_alive * 2}...")
print(f"Doubling 'Dead' entries from {count_dead} to {count_dead * 2}...")

# Sample without replacement (to include all original records)
oversampled_alive = df_alive.sample(n=count_alive, replace=False, random_state=42)
oversampled_dead = df_dead.sample(n=count_dead, replace=False, random_state=42)

# Combine original records with copies to double both classes
df_alive_doubled = pd.concat([df_alive, oversampled_alive], ignore_index=True)
df_dead_doubled = pd.concat([df_dead, oversampled_dead], ignore_index=True)

# Combine doubled alive with doubled dead
df_doubled = pd.concat([df_alive_doubled, df_dead_doubled], ignore_index=True)

# Shuffle the final dataset
df_doubled = shuffle(df_doubled, random_state=42)

print("\nClass distribution in the new dataset with doubled entries:")
print(df_doubled['Status'].value_counts())
print(f"Alive: {len(df_doubled[df_doubled['Status'] == 'Alive'])} (was {count_alive})")
print(f"Dead: {len(df_doubled[df_doubled['Status'] == 'Dead'])} (was {count_dead})")
print(f"Total records: {len(df_doubled)} (was {len(df_train)})")

# Save the dataset with doubled entries
try:
    df_doubled.to_csv(output_doubled_file_path, index=False)
    print(f"\nSuccessfully saved data with doubled entries to: {output_doubled_file_path}")
except Exception as e:
    print(f"\nError saving output file: {e}")