import pandas as pd
import sys

# Read both CSV files
df1 = pd.read_csv('../predictions/synapse-logreg-filtered-balanced.csv')
df2 = pd.read_csv('../predictions/synapse-logreg-filtered-balanced2.csv')

print(f'Shape 1: {df1.shape}, Shape 2: {df2.shape}')
print(f'Are IDs same? {(df1["Id"] == df2["Id"]).all()}')
print(f'Are Predictions same? {(df1["Prediction"] == df2["Prediction"]).all()}')

# Find differences
diff = df1[df1["Prediction"] != df2["Prediction"]]
print(f'\nNumber of differences: {len(diff)}')

if len(diff) > 0:
    print(f'\nFirst 20 differences:')
    for idx, row in diff.head(20).iterrows():
        pred1 = df1.loc[idx, "Prediction"]
        pred2 = df2.loc[idx, "Prediction"]
        print(f'  Id {row["Id"]}: {pred1} vs {pred2}')
    
    print(f'\nLast 20 differences:')
    for idx, row in diff.tail(20).iterrows():
        pred1 = df1.loc[idx, "Prediction"]
        pred2 = df2.loc[idx, "Prediction"]
        print(f'  Id {row["Id"]}: {pred1} vs {pred2}')
    
    # Show some statistics
    print(f'\nDifference statistics:')
    print(f'  Predictions changed from -1 to 1: {len(diff[diff["Prediction"] == 1])}')
    print(f'  Predictions changed from 1 to -1: {len(diff[diff["Prediction"] == -1])}')
    
    sys.exit(1)
else:
    print('\n✓ Files are identical!')
    sys.exit(0)

