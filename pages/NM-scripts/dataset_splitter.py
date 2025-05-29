import pandas as pd

# Takes 20% of dead entries from the original dataset, adds in an equal amount of alive entries, then inserts them into a val_set.
# The rest of the entries go to the train_set.

input_path = '../../Data/Breast_Cancer.csv'
train_output_path = '../../Data/NM-datasets/Breast_Cancer_train.csv'
val_output_path = '../../Data/NM-datasets/Breast_Cancer_val.csv'

df = pd.read_csv(input_path)

alive_df = df[df['Status'] == 'Alive']
dead_df  = df[df['Status'] == 'Dead']

random_state = 100

# Total count (just in case)
total = len(dead_df + alive_df)

# Count dead entries that take up 20% of total set
val_dead_count = int(0.2 * len(dead_df))

# Sample the same number of dead and alive entries for the val_set
val_dead  = dead_df.sample(n=val_dead_count, random_state=random_state)
val_alive = alive_df.sample(n=val_dead_count, random_state=random_state)

# Put the rest in the training set
train_dead  = dead_df.drop(val_dead.index)
train_alive = alive_df.drop(val_alive.index)

# Combine and shuffle
val_df   = pd.concat([val_dead, val_alive],   ignore_index=True) \
             .sample(frac=1, random_state=random_state)
train_df = pd.concat([train_dead, train_alive], ignore_index=True) \
             .sample(frac=1, random_state=random_state)

# Save the new sets
val_df.to_csv(val_output_path,   index=False)
train_df.to_csv(train_output_path, index=False)

print(f"Total Entries: {total}")
print(f"Validation Set: {len(val_df)} rows "
      f"(Alive: {sum(val_df['Status']=='Alive')}, Dead: {sum(val_df['Status']=='Dead')})")
print(f"Training Set: {len(train_df)} rows "
      f"(Alive: {sum(train_df['Status']=='Alive')}, Dead: {sum(train_df['Status']=='Dead')})")