import os
import pandas as pd

# Hard-coded paths
input_path = 'Data/Breast_Cancer.csv'
output_path = 'Data/Breast_Cancer_Balanced.csv'

# Load data
try:
    df = pd.read_csv(input_path)
except FileNotFoundError:
    print(f"Error: Input training file not found at {input_path}")
    exit()
except Exception as e:
    print(f"Error reading training file: {e}")
    exit()

print(f"Original dataset loaded from: {input_path}")
print("Original class distribution:")
print(df['Status'].value_counts())

# Split by status
dead_df  = df[df['Status'] == 'Dead']
alive_df = df[df['Status'] == 'Alive']

# Report original counts
count_dead  = len(dead_df)
count_alive = len(alive_df)
print(f'Original counts -> Alive: {count_alive}, Dead: {count_dead}')

# Oversample dead rows if needed
if count_dead < count_alive:
    n_to_sample = count_alive - count_dead
    sampled_dead = dead_df.sample(n=n_to_sample, replace=True, random_state=42)
    df = pd.concat([df, sampled_dead], ignore_index=True)
    print(f'Duplicated {n_to_sample} dead rows.')
else:
    print('No duplication needed: dead ≥ alive.')

# Shuffle the DataFrame
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save balanced CSV
df.to_csv(output_path, index=False)
print(f'Balanced dataset saved to {output_path}')

# Report new counts
final_alive = len(df[df['Status'] == 'Alive'])
final_dead  = len(df[df['Status'] == 'Dead'])
print(
    f'New counts -> alive: {final_alive}, '
    f'dead: {final_dead}'
)