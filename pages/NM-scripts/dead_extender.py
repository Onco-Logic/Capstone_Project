# This script extends Breast_Cancer.csv's dead entries by duplicating existing ones to help balance the class distribution for model training! (｡◕‿‿◕｡)

import pandas as pd

# Load dataframe
path = '../../Data/Breast_Cancer2.csv'
df = pd.read_csv(path)

# Count ALIVE and DEAD entries
alive_count = 0
dead_count = 0

# Run for loop that counts alive and dead
for idx, row in df.iterrows():
    if row['Status'] == 'Alive':
        alive_count += 1
    elif row['Status'] == 'Dead':
        dead_count += 1

# Calculate difference
dead_needed = alive_count - dead_count

# Collect all dead entires
dead_rows = [row for idx, row in df.iterrows() if row['Status'] == 'Dead']

# Add dead rows until balanced
with open(path, 'a', newline='') as f:
    for i in range(dead_needed):
        row = dead_rows[i % len(dead_rows)]
        line = ','.join([str(row[col]) for col in df.columns]) + '\n'
        f.write(line)

print(f"IT'S DONE!! ヽ༼ຈل͜ຈ༽ﾉ")
